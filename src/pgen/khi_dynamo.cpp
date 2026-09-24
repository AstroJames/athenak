//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file khi_dynamo.cpp
//! \brief Newtonian KHI dynamo profiles with native AthenaK spectral magnetic ICs.
// Build with -DPROBLEM=khi_dynamo. Uses existing shear viscosity and Ohmic diffusion.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "pgen.hpp"
#include "utils/spectral_ic_gen.hpp"

namespace {
void khi_dynamo_history(HistoryData *data, Mesh *mesh);

void invalid_khi_dynamo(const std::string &message) {
  std::cerr << "### FATAL ERROR in khi_dynamo: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = khi_dynamo_history;
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->pmhd == nullptr || pack->pcoord->is_special_relativistic ||
      pack->pcoord->is_general_relativistic || pack->pcoord->is_dynamical_relativistic) {
    invalid_khi_dynamo("requires Newtonian MHD");
  }
  if (!pack->pmhd->peos->eos_data.is_ideal || !pmy_mesh_->three_d) {
    invalid_khi_dynamo("requires a 3-D mesh and ideal-gas EOS");
  }
  const std::string field = pin->GetString("problem", "initial_field");
  const bool spectral = (field == "spectral");
  if (!spectral && field != "uniform_control") {
    invalid_khi_dynamo("initial_field must be spectral or uniform_control");
  }
  if (restart) return;

  const Real rho0 = pin->GetReal("problem", "sim_dens");
  const Real contrast = pin->GetReal("problem", "sim_dens_perturb");
  const Real pressure = pin->GetReal("problem", "sim_pres");
  const Real width = pin->GetReal("problem", "sim_layer_thickness");
  const Real sigma = pin->GetReal("problem", "sim_smoothing_size");
  const Real vx0 = pin->GetReal("problem", "sim_vx");
  const Real vy0 = pin->GetReal("problem", "sim_vy");
  const Real vz0 = pin->GetReal("problem", "sim_vz");
  const Real xlo = pin->GetReal("problem", "x_tilde_2");
  const Real xhi = pin->GetReal("problem", "x_tilde_1");
  const Real kz = pin->GetReal("problem", "k_z");
  const Real ly = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  const Real lz = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;
  const Real kmax = pin->GetReal("problem", "k_max");
  const Real bx = pin->GetReal("problem", "sim_bx");
  const Real by = pin->GetReal("problem", "sim_by");
  const Real bz = pin->GetReal("problem", "sim_bz");
  for (Real value : {rho0, contrast, pressure, width, sigma, vx0, vy0, vz0,
                     xlo, xhi, kz, kmax, bx, by, bz}) {
    if (!std::isfinite(value)) invalid_khi_dynamo("parameters must be finite");
  }
  if (rho0 <= 0.0 || contrast <= -1.0 || pressure <= 0.0 || width <= 0.0 ||
      sigma <= 0.0 || xhi <= xlo || kmax < 1.0 || kmax > 256.0 ||
      kmax != std::floor(kmax)) {
    invalid_khi_dynamo("invalid density, pressure, layer positions/widths, or k_max");
  }
  const int nmodes = static_cast<int>(kmax);
  DualArray1D<Real> phase("khi_dynamo_phase", nmodes + 1);
  for (int mode = 2; mode <= nmodes; ++mode) {
    // Fortran RANDOM_NUMBER is compiler-dependent: use supplied phase values in radians.
    const Real angle = pin->GetReal("problem", "phase_" + std::to_string(mode));
    if (!std::isfinite(angle)) invalid_khi_dynamo("phase values must be finite");
    phase.h_view(mode) = angle;
  }
  phase.template modify<HostMemSpace>();
  phase.template sync<DevExeSpace>();
  auto phases = phase.d_view;
  const auto &idx = pmy_mesh_->mb_indcs;
  const int is = idx.is, ie = idx.ie, js = idx.js, je = idx.je;
  const int ks = idx.ks, ke = idx.ke;
  const int nx1 = idx.nx1, nx2 = idx.nx2, nx3 = idx.nx3;
  auto size = pack->pmb->mb_size;
  auto u = pack->pmhd->u0;
  auto b = pack->pmhd->b0;
  const int nscalars = pack->pmhd->nscalars;
  const int nfluid = pack->pmhd->nmhd;
  const Real gm1 = pack->pmhd->peos->eos_data.gamma - 1.0;
  if (!(gm1 > 0.0)) invalid_khi_dynamo("gamma must exceed one");
  par_for("khi_dynamo_fluid", DevExeSpace(), 0, pack->nmb_thispack-1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    const Real layer = tanh((x-xlo)/width) - tanh((x-xhi)/width);
    const Real rho = rho0*(1.0 + 0.5*contrast*layer);
    Real perturb = 0.0;
    for (int mode = 2; mode <= nmodes; ++mode) {
      perturb += sin(2.0*M_PI*mode*y/ly + phases(mode))/sqrt(kmax-1.0);
    }
    const Real vx = vx0*perturb*(exp(-SQR((x-xlo)/sigma)) + exp(-SQR((x-xhi)/sigma)));
    const Real vy = vy0*(layer-1.0);
    const Real vz = vz0*sin(2.0*M_PI*kz*z/lz);
    u(m,IDN,k,j,i) = rho;
    u(m,IM1,k,j,i) = rho*vx;
    u(m,IM2,k,j,i) = rho*vy;
    u(m,IM3,k,j,i) = rho*vz;
    u(m,IEN,k,j,i) = pressure/gm1 + 0.5*rho*(SQR(vx)+SQR(vy)+SQR(vz))
                                    + 0.5*(SQR(bx)+SQR(by)+SQR(bz));
    for (int n = 0; n < nscalars; ++n) {
      u(m,nfluid+n,k,j,i) = 0.0;
    }
    b.x1f(m,k,j,i) = bx;
    b.x2f(m,k,j,i) = by;
    b.x3f(m,k,j,i) = bz;
    if (i == ie) b.x1f(m,k,j,i+1) = bx;
    if (j == je) b.x2f(m,k,j+1,i) = by;
    if (k == ke) b.x3f(m,k+1,j,i) = bz;
  });
  // Ensure the local phase allocation remains alive until initialization finishes.
  Kokkos::fence();
  if (spectral) {
    const Real rms_b = pin->GetReal("problem", "magnetic_rms");
    if (!std::isfinite(rms_b) || rms_b <= 0.0) {
      invalid_khi_dynamo("magnetic_rms must be finite and positive");
    }
    int nmb = pack->nmb_thispack;
    int ncells1 = idx.nx1 + 2*idx.ng;
    int ncells2 = idx.nx2 + 2*idx.ng;
    int ncells3 = idx.nx3 + 2*idx.ng;
    DvceArray4D<Real> ax("khi_dynamo_ax_spec", nmb, ncells3, ncells2, ncells1);
    DvceArray4D<Real> ay("khi_dynamo_ay_spec", nmb, ncells3, ncells2, ncells1);
    DvceArray4D<Real> az("khi_dynamo_az_spec", nmb, ncells3, ncells2, ncells1);
    par_for("pgen_khi_dynamo_zero_a", DevExeSpace(), 0, nmb - 1,
            0, ncells3 - 1, 0, ncells2 - 1, 0, ncells1 - 1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      ax(m, k, j, i) = 0.0;
      ay(m, k, j, i) = 0.0;
      az(m, k, j, i) = 0.0;
    });

    SpectralICGenerator gen(pack, pin);
    gen.GenerateVectorPotential(ax, ay, az);

    par_for("pgen_khi_dynamo_curl_a", DevExeSpace(), 0, nmb - 1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      b.x1f(m, k, j, i) =
          (az(m, k, j + 1, i) - az(m, k, j, i))/dx2
        - (ay(m, k + 1, j, i) - ay(m, k, j, i))/dx3;
      b.x2f(m, k, j, i) =
          (ax(m, k + 1, j, i) - ax(m, k, j, i))/dx3
        - (az(m, k, j, i + 1) - az(m, k, j, i))/dx1;
      b.x3f(m, k, j, i) =
          (ay(m, k, j, i + 1) - ay(m, k, j, i))/dx1
        - (ax(m, k, j + 1, i) - ax(m, k, j, i))/dx2;
      if (i == ie) {
        b.x1f(m, k, j, i + 1) =
            (az(m, k, j + 1, i + 1) - az(m, k, j, i + 1))/dx2
          - (ay(m, k + 1, j, i + 1) - ay(m, k, j, i + 1))/dx3;
      }
      if (j == je) {
        b.x2f(m, k, j + 1, i) =
            (ax(m, k + 1, j + 1, i) - ax(m, k, j + 1, i))/dx3
          - (az(m, k, j + 1, i + 1) - az(m, k, j + 1, i))/dx1;
      }
      if (k == ke) {
        b.x3f(m, k + 1, j, i) =
            (ay(m, k + 1, j, i + 1) - ay(m, k + 1, j, i))/dx1
          - (ax(m, k + 1, j + 1, i) - ax(m, k + 1, j, i))/dx2;
      }
    });

    SubtractGlobalMeanB(pack, b);
    NormalizeRmsB(pack, b, rms_b);

    // Add the requested mean after normalizing the zero-mean fluctuation.
    par_for("khi_dynamo_mean_b", DevExeSpace(), 0, pack->nmb_thispack-1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b.x1f(m,k,j,i) += bx;
      b.x2f(m,k,j,i) += by;
      b.x3f(m,k,j,i) += bz;
      if (i == ie) b.x1f(m,k,j,i+1) += bx;
      if (j == je) b.x2f(m,k,j+1,i) += by;
      if (k == ke) b.x3f(m,k+1,j,i) += bz;
    });
    par_for("khi_dynamo_magnetic_energy", DevExeSpace(), 0, pack->nmb_thispack-1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real b1 = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      const Real b2 = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      const Real b3 = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
      u(m,IEN,k,j,i) += 0.5*(SQR(b1)+SQR(b2)+SQR(b3)-SQR(bx)-SQR(by)-SQR(bz));
    });
    Kokkos::fence();
  }
}

namespace {
// Add net magnetic flux and a cell-centered energy split to the standard history.
// HistoryOutput performs the MPI sum; all quantities here are rank-local sums.
void khi_dynamo_history(HistoryData *data, Mesh *mesh) {
  data->nhist = 6;
  const char *labels[] = {"mean-B1", "mean-B2", "mean-B3", "ME-cc", "int-E",
                          "divB-L1"};
  for (int n = 0; n < data->nhist; ++n) data->label[n] = labels[n];
  auto *pack = mesh->pmb_pack;
  auto u = pack->pmhd->u0;
  auto b = pack->pmhd->b0;
  auto size = pack->pmb->mb_size;
  const auto &idx = mesh->mb_indcs;
  const int is = idx.is, js = idx.js, ks = idx.ks;
  const int nx1 = idx.nx1, nx2 = idx.nx2, nx3 = idx.nx3;
  const auto &box = mesh->mesh_size;
  const Real volume = (box.x1max-box.x1min)*(box.x2max-box.x2min)
                      *(box.x3max-box.x3min);
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("khi_dynamo_history",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, pack->nmb_thispack*nx1*nx2*nx3),
  KOKKOS_LAMBDA(const int &index, array_sum::GlobalSum &total) {
    const int m = index/(nx1*nx2*nx3);
    const int cell = index%(nx1*nx2*nx3);
    const int i = is + cell%nx1;
    const int j = js + (cell/nx1)%nx2;
    const int k = ks + cell/(nx1*nx2);
    const Real dx = size.d_view(m).dx1, dy = size.d_view(m).dx2;
    const Real dz = size.d_view(m).dx3;
    const Real dv = dx*dy*dz;
    const Real b1 = 0.5*(b.x1f(m,k,j,i)+b.x1f(m,k,j,i+1));
    const Real b2 = 0.5*(b.x2f(m,k,j,i)+b.x2f(m,k,j+1,i));
    const Real b3 = 0.5*(b.x3f(m,k,j,i)+b.x3f(m,k+1,j,i));
    const Real me = 0.5*(SQR(b1)+SQR(b2)+SQR(b3));
    const Real ke = 0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))
                        +SQR(u(m,IM3,k,j,i)))/u(m,IDN,k,j,i);
    const Real divb = (b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))/dx
                   + (b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))/dy
                   + (b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i))/dz;
    array_sum::GlobalSum values;
    for (int n = 0; n < NHISTORY_VARIABLES; ++n) values.the_array[n] = 0.0;
    values.the_array[0] = dv*b1/volume;
    values.the_array[1] = dv*b2/volume;
    values.the_array[2] = dv*b3/volume;
    values.the_array[3] = dv*me;
    values.the_array[4] = dv*(u(m,IEN,k,j,i)-ke-me);
    values.the_array[5] = dv*fabs(divb)/volume;
    total += values;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));
  Kokkos::fence();
  for (int n = 0; n < data->nhist; ++n) data->hdata[n] = sum.the_array[n];
}
}  // namespace
