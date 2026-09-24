//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file khi_diagnostics.cpp
//! \brief KHI whole-volume/half-volume histories and native-resolution planar profiles.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "khi_diagnostics.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
const char *labels[khi::nvar] = {
  "volume", "rho", "mom_x", "mom_y", "mom_z", "Etot", "Eint",
  "ux", "uy", "uz", "Bx", "By", "Bz", "ux2", "uy2", "uz2",
  "Bx2", "By2", "Bz2", "Kx", "Ky", "Kz", "uyBz", "uzBy", "uzBx",
  "uxBz", "uxBy", "uyBx", "omega_x", "omega_y", "omega_z", "Jx", "Jy", "Jz",
  "omega2", "J2", "u_dot_omega", "B_dot_J", "u_dot_B", "stretch",
  "compression", "ohmic_heat", "visc_heat", "div_u", "pressure"
};
const char *regions[] = {"global", "lower", "upper"};
const char *planes[] = {"yz", "xz", "xy"};

std::vector<Real> Extras(const std::vector<Real> &q) {
  Real ke = q[19]+q[20]+q[21], me = 0.5*(q[16]+q[17]+q[18]);
  Real mean_me = 0.0, k_reynolds = ke, k_favre = ke;
  for (int a = 0; a < 3; ++a) {
    mean_me += 0.5*SQR(q[10+a]);
    k_reynolds += -q[7+a]*q[2+a]+0.5*q[1]*SQR(q[7+a]);
    k_favre -= 0.5*SQR(q[2+a])/q[1];
  }
  return {q[22]-q[23]-(q[8]*q[12]-q[9]*q[11]),
          q[24]-q[25]-(q[9]*q[10]-q[7]*q[12]),
          q[26]-q[27]-(q[7]*q[11]-q[8]*q[10]),
          ke, me, mean_me, me-mean_me, k_reynolds, k_favre};
}
const char *extra_labels[] = {"emf_x", "emf_y", "emf_z", "Ekin", "Emag",
                              "Emag_mean", "Emag_fluct", "Kfluct_vol", "Kfluct_Favre"
};

void Header(std::ostream &out) {
  out << "# Volume-weighted means; lower: x<midpoint, upper: x>=midpoint.\n"
      << "# volume is an integral; remaining raw quantities are volume averages.\n"
      << "# Eint is energy density, not specific energy; J=curl(B), mu=1.\n"
      << "# stretch=B_i B^j partial_j u^i; compression=-B^2 div(u)/2.\n"
      << "# Dissipation uses centered diagnostic gradients, not solver fluxes.\n"
      << "# Subvolume energies/momenta exchange flux and are not separately conserved.\n";
}
}  // namespace

KhiDiagnosticsOutput::KhiDiagnosticsOutput(ParameterInput *pin, Mesh *pm,
                                           OutputParameters op)
    : BaseTypeOutput(pin, pm, op), profiles_(op.file_type == "khi_profiles") {
  auto *pack = pm->pmb_pack;
  if (!pm->three_d || pm->multilevel || !pm->strictly_periodic || pack->pmhd == nullptr ||
      pack->pcoord->is_special_relativistic || pack->pcoord->is_general_relativistic ||
      pack->pcoord->is_dynamical_relativistic || !pack->pmhd->peos->eos_data.is_ideal ||
      pm->mesh_indcs.nx1%2 != 0) {
    std::cerr << "KHI diagnostics require uniform periodic 3-D Newtonian ideal MHD "
              << "with even nx1.\n";
    std::exit(EXIT_FAILURE);
  }
  cells_[0] = pm->mesh_indcs.nx1;
  cells_[1] = pm->mesh_indcs.nx2;
  cells_[2] = pm->mesh_indcs.nx3;
  lower_[0] = pm->mesh_size.x1min;
  lower_[1] = pm->mesh_size.x2min;
  lower_[2] = pm->mesh_size.x3min;
  Real upper[] = {pm->mesh_size.x1max, pm->mesh_size.x2max, pm->mesh_size.x3max};
  nplanes_ = 0;
  for (int a = 0; a < 3; ++a) {
    offset_[a] = nplanes_;
    nplanes_ += cells_[a];
    spacing_[a] = (upper[a]-lower_[a])/cells_[a];
  }
  nu_ = pin->DoesParameterExist("mhd", "viscosity") ?
        pin->GetReal("mhd", "viscosity") : 0.0;
  eta_ = pin->DoesParameterExist("mhd", "ohmic_resistivity") ?
         pin->GetReal("mhd", "ohmic_resistivity") : 0.0;
  gm1_ = pack->pmhd->peos->eos_data.gamma-1.0;
  sums_.resize(2*nplanes_*khi::nvar);
}

void KhiDiagnosticsOutput::LoadOutputData(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto w = pack->pmhd->w0, u = pack->pmhd->u0, b = pack->pmhd->bcc0;
  auto size = pack->pmb->mb_size;
  auto idx = pm->mb_indcs;
  int nx = idx.nx1, ny = idx.nx2, nz = idx.nx3;
  int local_planes = nx+ny+nz, np = nplanes_;
  int gx = cells_[0], gy = cells_[1];
  Real x0 = lower_[0], y0 = lower_[1], z0 = lower_[2];
  Real dx = spacing_[0], dy = spacing_[1], dz = spacing_[2];
  Real nu = nu_, eta = eta_, gm1 = gm1_;
  DvceArray2D<Real> sums("khi_plane_sums", 2*np, khi::nvar);
  Kokkos::deep_copy(sums, 0.0);
  // Reduce each local plane cooperatively, then atomically add only one result per
  // block plane. Neither full grids nor cell data are gathered onto the host/rank zero.
  using Policy = Kokkos::TeamPolicy<DevExeSpace>;
  Kokkos::parallel_for("khi_planes",
      Policy(pack->nmb_thispack*local_planes*2, Kokkos::AUTO),
  KOKKOS_LAMBDA(const Policy::member_type &team) {
    int task = team.league_rank();
    int half = task%2, p = (task/2)%local_planes, m = task/(2*local_planes);
    int axis = (p < nx) ? 0 : ((p < nx+ny) ? 1 : 2);
    int plane = p-((axis == 0) ? 0 : ((axis == 1) ? nx : nx+ny));
    int count = (axis == 0) ? ny*nz : ((axis == 1) ? nx*nz : nx*ny);
    int ox = static_cast<int>((size.d_view(m).x1min-x0)/dx+0.5);
    int oy = static_cast<int>((size.d_view(m).x2min-y0)/dy+0.5);
    int oz = static_cast<int>((size.d_view(m).x3min-z0)/dz+0.5);
    array_sum::GlobalSum sum;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, count),
    [=](const int c, array_sum::GlobalSum &total) {
      int i = (axis == 0) ? plane : c%nx;
      int j = (axis == 1) ? plane : ((axis == 0) ? c%ny : c/nx);
      int k = (axis == 2) ? plane : ((axis == 0) ? c/ny : c/nx);
      if ((ox+i < gx/2 ? 0 : 1) != half) return;
      array_sum::GlobalSum values;
      khi::Cell(w, u, b, m, idx.ks+k, idx.js+j, idx.is+i,
                dx, dy, dz, nu, eta, gm1, values.the_array);
      for (int n = 0; n < khi::nvar; ++n) values.the_array[n] *= dx*dy*dz;
      total += values;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum));
    Kokkos::single(Kokkos::PerTeam(team), [=]() {
      int row = half*np + ((axis == 0) ? ox+plane :
                            ((axis == 1) ? gx+oy+plane : gx+gy+oz+plane));
      for (int n = 0; n < khi::nvar; ++n) {
        Kokkos::atomic_add(&sums(row,n), sum.the_array[n]);
      }
    });
  });
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), sums);
  for (int row = 0; row < 2*np; ++row) {
    for (int n = 0; n < khi::nvar; ++n) sums_[row*khi::nvar+n] = host(row,n);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums_.data(), static_cast<int>(sums_.size()),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
}

std::vector<Real> KhiDiagnosticsOutput::Plane(int region, int axis, int bin) const {
  std::vector<Real> q(khi::nvar, 0.0);
  for (int h = 0; h < 2; ++h) {
    if (region != 0 && region != h+1) continue;
    int row = h*nplanes_+offset_[axis]+bin;
    for (int n = 0; n < khi::nvar; ++n) q[n] += sums_[row*khi::nvar+n];
  }
  if (q[0] > 0.0) {
    for (int n = 1; n < khi::nvar; ++n) q[n] /= q[0];
  }
  return q;
}

void KhiDiagnosticsOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  if (global_variable::my_rank == 0) {
    if (profiles_) {
      std::ostringstream number;
      number << std::setw(5) << std::setfill('0') << out_params.file_number;
      std::ofstream out(out_params.file_basename+".khi_profiles."+number.str()+".tab");
      Header(out);
      out << std::scientific << std::setprecision(17)
          << "# time=" << pm->time << " cycle=" << pm->ncycle << '\n'
          << "# region: 0=global,1=lower,2=upper; axis: 0=yz,1=xz,2=xy\n"
          << "# region axis coordinate";
      for (auto label : labels) out << ' ' << label;
      for (auto label : extra_labels) out << ' ' << label;
      out << '\n';
      for (int r = 0; r < 3; ++r) for (int a = 0; a < 3; ++a) {
        for (int p = 0; p < cells_[a]; ++p) {
          auto q = Plane(r,a,p);
          if (q[0] == 0.0) continue;
          out << r << ' ' << a << ' ' << lower_[a]+(p+0.5)*spacing_[a];
          for (Real value : q) out << ' ' << value;
          for (Real value : Extras(q)) out << ' ' << value;
          out << '\n';
        }
      }
      if (!out) {
        std::cerr << "Failed to write KHI profiles\n";
        std::exit(EXIT_FAILURE);
      }
    } else {
      for (int r = 0; r < 3; ++r) {
        std::string name = out_params.file_basename+".khi_"+regions[r]+".hst";
        bool header = out_params.file_number == 0 || !std::ifstream(name).good();
        auto mode = out_params.file_number == 0 ? std::ios::out : std::ios::app;
        std::ofstream out(name, mode);
        if (header) {
          Header(out);
          out
              << "# Kfluct_vol: volume-mean velocity; Favre: mass-mean velocity.\n"
              << "# Bmean2_<plane>_<component> is <(plane-mean B)^2>, not <B^2>.\n"
              << "# time dt";
          for (auto label : labels) out << ' ' << label;
          for (auto label : extra_labels) out << ' ' << label;
          for (auto plane : planes) {
            for (int c = 0; c < 3; ++c) out << " Bmean2_" << plane << '_' << c+1;
            out << " Kfluct_vol_" << plane << " Kfluct_Favre_" << plane;
          }
          out << '\n';
        }
        std::vector<Real> q(khi::nvar, 0.0), planar(15, 0.0);
        for (int p = 0; p < cells_[0]; ++p) {
          auto row = Plane(r,0,p);
          q[0] += row[0];
          for (int n = 1; n < khi::nvar; ++n) q[n] += row[0]*row[n];
        }
        for (int n = 1; n < khi::nvar; ++n) q[n] /= q[0];
        for (int a = 0; a < 3; ++a) for (int p = 0; p < cells_[a]; ++p) {
          auto row = Plane(r,a,p);
          if (row[0] == 0.0) continue;
          auto ex = Extras(row);
          for (int c = 0; c < 3; ++c) planar[5*a+c] += row[0]*SQR(row[10+c])/q[0];
          planar[5*a+3] += row[0]*ex[7]/q[0];
          planar[5*a+4] += row[0]*ex[8]/q[0];
        }
        out << std::scientific << std::setprecision(17) << pm->time << ' ' << pm->dt;
        for (Real value : q) out << ' ' << value;
        for (Real value : Extras(q)) out << ' ' << value;
        for (Real value : planar) out << ' ' << value;
        out << '\n';
        if (!out) {
          std::cerr << "Failed to write KHI history\n";
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }
  ++out_params.file_number;
  out_params.last_time = out_params.last_time < 0 ? pm->time :
                         out_params.last_time+out_params.dt;
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
