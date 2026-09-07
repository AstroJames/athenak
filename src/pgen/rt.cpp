//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rt.cpp
//! \brief Problem generator for RT instabilty.
//!
//! Note the gravitational acceleration is hardwired to be 0.1. Density difference is
//! hardwired to be 3.0 and is set by the input parameter `problem/drat`.
//! To reproduces 2D results of Liska & Wendroff set it to 2.0,
//! while for the 3D results of Dimonte et al use 3.0.
//!
//! FOR 2D HYDRO:
//! Problem domain should be -1/6 < x < 1/6; -0.5 < y < 0.5 with gamma=1.4 to match Liska
//! & Wendroff. Interface is at y=0; perturbation added to Vy. Gravity acts in y-dirn.
//! Special reflecting boundary conditions added in x2 to improve hydrostatic eqm
//! (prevents launching of weak waves) Atwood number A=(d2-d1)/(d2+d1)=1/3. Options:
//!    - iprob = 1  -- Perturb V2 using single mode
//!    - iprob != 1 -- Perturb V2 using multiple mode
//!
//! FOR 3D:
//! Problem domain should be -0.5 < x < 0.5; -0.5 < y < 0.5, -1. < z < 1., gamma=5/3 to
//! match Dimonte et al.  Interface is at z=0; perturbation added to Vz. Gravity acts in
//! z-dirn. Special reflecting boundary conditions added in x3.  A=1/2.  Options:
//!    - iprob = 1 -- Perturb V3 using single mode
//!    - iprob = 2 -- Perturb V3 using multiple mode
//!    - iprob = 3 -- B rotated by "angle" at interface, multimode perturbation
//!
//! REFERENCE: R. Liska & B. Wendroff, SIAM J. Sci. Comput., 25, 995 (2003)
//!
//! The optional `problem/xu_shu=true` mode implements the inviscid two-dimensional
//! Rayleigh--Taylor setup of Xu & Shu, JCP 205, 458 (2005), as used for Fig. 19 of
//! Fu, Hu & Adams, JCP 305, 333 (2016).  This mode requires hydro, constant
//! acceleration +1 in x2, reflective x1 boundaries, and user x2 boundaries.

// C++ headers
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream> // cout
#include <string>
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/random.hpp"
#include "pgen.hpp"

#include <Kokkos_Random.hpp>

namespace {

void XuShuBoundary(Mesh *pm);
void XuShuSymmetryErrors(ParameterInput *pin, Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::RayleighTaylor()
//  \brief Problem Generator for the Rayleigh-Taylor instability test

#if defined(RT_USER_PROBLEM_ENABLED)
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
#else
void ProblemGenerator::RayleighTaylor(ParameterInput *pin, const bool restart) {
#endif
  bool xu_shu = pin->GetOrAddBoolean("problem", "xu_shu", false);
  if (xu_shu) {
    user_bcs_func = XuShuBoundary;
    if (pin->GetOrAddBoolean("problem", "check_symmetry", false)) {
      pgen_final_func = XuShuSymmetryErrors;
    }
  }
  if (restart) return;
  if (pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rti problem generator only works in 2D/3D" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real kx = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);
  Real ky = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min);
  Real kz = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Xu--Shu/Fu et al. 2D inviscid RT setup --------------------------------------
  if (xu_shu) {
    if (!(pmbp->pmesh->two_d) || pmbp->phydro == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "problem/xu_shu=true requires two-dimensional hydrodynamics"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!(pmbp->phydro->peos->eos_data.is_ideal)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "problem/xu_shu=true requires an ideal-gas EOS" << std::endl;
      exit(EXIT_FAILURE);
    }
    Real grav_acc = pin->GetReal("hydro", "const_accel_val");
    int grav_dir = pin->GetInteger("hydro", "const_accel_dir");
    if (grav_dir != 2 || std::abs(grav_acc - 1.0) > 1.0e-14) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "problem/xu_shu=true requires hydro/const_accel_val=1 and "
                << "hydro/const_accel_dir=2" << std::endl;
      exit(EXIT_FAILURE);
    }

    auto &u0 = pmbp->phydro->u0;
    Real gamma = pmbp->phydro->peos->eos_data.gamma;
    Real gm1 = gamma - 1.0;
    Real x1mesh_min = pmy_mesh_->mesh_size.x1min;
    Real x1mesh_max = pmy_mesh_->mesh_size.x1max;
    par_for("rt2d_xu_shu", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real den = (x2v < 0.5) ? 2.0 : 1.0;
      Real pgas = (x2v < 0.5) ? (1.0 + 2.0*x2v) : (x2v + 1.5);
      Real cs = sqrt(gamma*pgas/den);
      // Evaluate the perturbation at one canonical coordinate for each reflected pair.
      // This prevents independently rounded cos(theta) and cos(2*pi-theta) values from
      // seeding an asymmetric mode in this deliberately reflection-symmetric problem.
      Real x1mirror = x1mesh_min + (x1mesh_max - x1v);
      Real x1fold = fmin(x1v, x1mirror);
      Real vel2 = -0.025*cs*cos(8.0*M_PI*x1fold);

      u0(m,IDN,k,j,i) = den;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = den*vel2;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = pgas/gm1 + 0.5*den*SQR(vel2);
    });
    return;
  }

  // Read perturbation amplitude, problem switch, density ratio for the original setup.
  Real amp = pin->GetReal("problem","amp");
  int iprob = pin->GetInteger("problem","iprob");
  Real drat = pin->GetOrAddReal("problem","drat",3.0);
  bool smooth_interface = pin->GetOrAddBoolean("problem","smooth_interface",false);

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_;
  Real gm1, p0;
  Real grav_acc;
  if (pmbp->phydro != nullptr) {
    grav_acc = pin->GetReal("hydro","const_accel_val");
    u0_ = pmbp->phydro->u0;
    gm1 = (pmbp->phydro->peos->eos_data.gamma) - 1.0;
    p0 = 1.0/(pmbp->phydro->peos->eos_data.gamma);
    p0 = pin->GetOrAddReal("problem", "p0", p0);
  } else if (pmbp->pmhd != nullptr) {
    grav_acc = pin->GetReal("mhd","const_accel_val");
    u0_ = pmbp->pmhd->u0;
    gm1 = (pmbp->pmhd->peos->eos_data.gamma) - 1.0;
    p0 = 1.0/(pmbp->pmhd->peos->eos_data.gamma);
    p0 = pin->GetOrAddReal("problem", "p0", p0);
  }

  // Ensure that p0 is sufficiently large to avoid negative pressures
  if (pmbp->pmesh->two_d) {
    p0 -= grav_acc*pmy_mesh_->mesh_size.x2max;
  } else {
    p0 -= grav_acc*pmy_mesh_->mesh_size.x3max;
  }

  // 2D PROBLEM ----------------------------------------------------------------

  if (pmbp->pmesh->two_d) {
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("rt2d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real den=1.0;
      Real sigma = 0.01;
      if (smooth_interface) {
        den = 0.5*((drat + 1.0) + (drat - 1.0)*tanh(x2v/sigma));
      } else {
        if (x2v > 0.0) den *= drat;
      }

      if (iprob == 1) {
        u0_(m,IM2,k,j,i) = (1.0 + cos(kx*x1v))*(1.0 + cos(ky*x2v))/4.0;
      } else {
        auto rand_gen = rand_pool64.get_state();  // get random number state this thread
        u0_(m,IM2,k,j,i) = (rand_gen.frand()-0.5)*(1.0 + cos(ky*x2v))/4.0;
        rand_pool64.free_state(rand_gen);  // free state for use by other threads
      }

      u0_(m,IDN,k,j,i) = den;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) *= (den*amp);
      u0_(m,IM3,k,j,i) = 0.0;
      u0_(m,IEN,k,j,i) = (p0 + grav_acc*den*x2v)/gm1 + 0.5*SQR(u0_(m,IM2,k,j,i))/den;
    });

    // initialize magnetic fields if MHD
    if (pmbp->pmhd != nullptr) {
      // Read magnetic field strength
      Real bx = pin->GetReal("problem","b0");
      auto &b0 = pmbp->pmhd->b0;
      par_for("pgen_b0", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        b0.x1f(m,k,j,i) = bx;
        b0.x2f(m,k,j,i) = 0.0;
        b0.x3f(m,k,j,i) = 0.0;
        if (i==ie) b0.x1f(m,k,j,i+1) = bx;
        if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
        if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
        u0_(m,IEN,k,j,i) += 0.5*bx*bx;
      });
    }

  // 3D PROBLEM ----------------------------------------------------------------

  } else {
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("rt3d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real den=1.0;
      if (x3v > 0.0) den *= drat;

      if (iprob == 1) {
        u0_(m,IM3,k,j,i) = (1.0+cos(kx*x1v))*(1.0+cos(ky*x2v))*(1.0+cos(kz*x3v))/8.0;
      } else {
        auto rand_gen = rand_pool64.get_state();  // get random number state this thread
        Real r = 2.0*static_cast<Real>(rand_gen.frand()) - 1.0;
        u0_(m,IM3,k,j,i) = r * (1.0 + cos(kz*x3v))/2.0;
        rand_pool64.free_state(rand_gen);  // free state for use by other threads
      }

      u0_(m,IDN,k,j,i) = den;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) = 0.0;
      u0_(m,IM3,k,j,i) *= (den*amp);
      u0_(m,IEN,k,j,i) = (p0 + grav_acc*den*x3v)/gm1 + 0.5*SQR(u0_(m,IM3,k,j,i))/den;
    });
  }

  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \brief Measure exact x1-reflection parity of the final Xu--Shu hydro state.

void XuShuSymmetryErrors(ParameterInput *pin, Mesh *pm) {
  if (pm->multilevel) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Xu--Shu symmetry diagnostics require a uniform mesh"
                << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }

  constexpr int nfields = 10;
  const std::array<const char *, nfields> labels = {
      "u_d", "u_m1", "u_m2", "u_m3", "u_e",
      "w_d", "w_v1", "w_v2", "w_v3", "w_e"};
  const std::array<int, nfields> parity = {
      1, -1, 1, 1, 1, 1, -1, 1, 1, 1};

  auto *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), phydro->u0);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), phydro->w0);
  auto f1 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), phydro->uflx.x1f);
  auto f2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), phydro->uflx.x2f);
  const auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1_mb = indcs.nx1, nx2_mb = indcs.nx2, nx3_mb = indcs.nx3;
  const int nx1 = pm->mesh_indcs.nx1;
  const int nx2 = pm->mesh_indcs.nx2;
  const int nx3 = pm->mesh_indcs.nx3;
  const int ncells = nx1*nx2*nx3;
  const int nfaces1 = (nx1 + 1)*nx2*nx3;
  const int nfaces2 = nx1*(nx2 + 1)*nx3;
  std::vector<Real> state(nfields*ncells, 0.0);
  std::vector<Real> flux1(5*nfaces1, 0.0);
  std::vector<Real> flux2(5*nfaces2, 0.0);

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const int gid = pmbp->pmb->mb_gid.h_view(m);
    const LogicalLocation &lloc = pm->lloc_eachmb[gid];
    const int gi0 = lloc.lx1*nx1_mb;
    const int gj0 = lloc.lx2*nx2_mb;
    const int gk0 = lloc.lx3*nx3_mb;
    for (int k = ks; k < ks + nx3_mb; ++k) {
      const int gk = gk0 + k - ks;
      for (int j = js; j < js + nx2_mb; ++j) {
        const int gj = gj0 + j - js;
        for (int i = is; i < is + nx1_mb; ++i) {
          const int gi = gi0 + i - is;
          const int idx = (gk*nx2 + gj)*nx1 + gi;
          state[0*ncells + idx] = u(m, IDN, k, j, i);
          state[1*ncells + idx] = u(m, IM1, k, j, i);
          state[2*ncells + idx] = u(m, IM2, k, j, i);
          state[3*ncells + idx] = u(m, IM3, k, j, i);
          state[4*ncells + idx] = u(m, IEN, k, j, i);
          state[5*ncells + idx] = w(m, IDN, k, j, i);
          state[6*ncells + idx] = w(m, IVX, k, j, i);
          state[7*ncells + idx] = w(m, IVY, k, j, i);
          state[8*ncells + idx] = w(m, IVZ, k, j, i);
          state[9*ncells + idx] = w(m, IEN, k, j, i);

          const int iface1 = (gk*nx2 + gj)*(nx1 + 1) + gi;
          const int iface2 = (gk*(nx2 + 1) + gj)*nx1 + gi;
          for (int n = 0; n < 5; ++n) {
            flux1[n*nfaces1 + iface1] = f1(m, n, k, j, i);
            flux2[n*nfaces2 + iface2] = f2(m, n, k, j, i);
          }
        }
        if (gi0 + nx1_mb == nx1) {
          const int iface1 = (gk*nx2 + gj)*(nx1 + 1) + nx1;
          for (int n = 0; n < 5; ++n) {
            flux1[n*nfaces1 + iface1] = f1(m, n, k, j, is + nx1_mb);
          }
        }
      }
      if (gj0 + nx2_mb == nx2) {
        for (int i = is; i < is + nx1_mb; ++i) {
          const int gi = gi0 + i - is;
          const int iface2 = (gk*(nx2 + 1) + nx2)*nx1 + gi;
          for (int n = 0; n < 5; ++n) {
            flux2[n*nfaces2 + iface2] = f2(m, n, k, js + nx2_mb, i);
          }
        }
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, state.data(), nfields*ncells, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, flux1.data(), 5*nfaces1, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, flux2.data(), 5*nfaces2, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  if (global_variable::my_rank == 0) {
    std::array<std::int64_t, nfields> broken{};
    std::array<Real, nfields> max_residual{};
    std::array<std::int64_t, 10> broken_flux{};
    std::array<Real, 10> max_flux_residual{};
    for (int gk = 0; gk < nx3; ++gk) {
      for (int gj = 0; gj < nx2; ++gj) {
        for (int gi = 0; gi < nx1; ++gi) {
          const int idx = (gk*nx2 + gj)*nx1 + gi;
          const int reflected = (gk*nx2 + gj)*nx1 + (nx1 - 1 - gi);
          for (int n = 0; n < nfields; ++n) {
            const Real residual = state[n*ncells + idx]
                                - parity[n]*state[n*ncells + reflected];
            if (residual != 0.0) {
              ++broken[n];
              max_residual[n] = std::max(max_residual[n], std::abs(residual));
            }
          }
        }
      }
    }

    const std::array<int, 5> flux1_parity = {-1, 1, -1, -1, -1};
    const std::array<int, 5> flux2_parity = {1, -1, 1, 1, 1};
    for (int gk = 0; gk < nx3; ++gk) {
      for (int gj = 0; gj < nx2; ++gj) {
        for (int gf = 0; gf <= nx1; ++gf) {
          const int idx = (gk*nx2 + gj)*(nx1 + 1) + gf;
          const int reflected = (gk*nx2 + gj)*(nx1 + 1) + (nx1 - gf);
          for (int n = 0; n < 5; ++n) {
            const Real residual = flux1[n*nfaces1 + idx]
                                - flux1_parity[n]*flux1[n*nfaces1 + reflected];
            if (residual != 0.0) {
              ++broken_flux[n];
              max_flux_residual[n] = std::max(max_flux_residual[n],
                                               std::abs(residual));
            }
          }
        }
      }
      for (int gf = 0; gf < nx1; ++gf) {
        for (int gj = 0; gj <= nx2; ++gj) {
          const int idx = (gk*(nx2 + 1) + gj)*nx1 + gf;
          const int reflected = (gk*(nx2 + 1) + gj)*nx1 + (nx1 - 1 - gf);
          for (int n = 0; n < 5; ++n) {
            const Real residual = flux2[n*nfaces2 + idx]
                                - flux2_parity[n]*flux2[n*nfaces2 + reflected];
            if (residual != 0.0) {
              ++broken_flux[5+n];
              max_flux_residual[5+n] = std::max(max_flux_residual[5+n],
                                                 std::abs(residual));
            }
          }
        }
      }
    }

    const std::string basename = pin->GetString("job", "basename");
    std::ofstream file(basename + "-symmetry.dat");
    file << "# nx1 nx2 nx3 ncycle time";
    for (int n = 0; n < nfields; ++n) {
      file << " n_" << labels[n] << " max_" << labels[n];
    }
    for (const char *direction : {"f1", "f2"}) {
      for (const char *field : {"d", "m1", "m2", "m3", "e"}) {
        file << " n_" << direction << "_" << field
             << " max_" << direction << "_" << field;
      }
    }
    file << "\n" << std::setprecision(17) << nx1 << " " << nx2 << " " << nx3
         << " " << pm->ncycle << " " << pm->time;
    for (int n = 0; n < nfields; ++n) {
      file << " " << broken[n] << " " << max_residual[n];
    }
    for (int n = 0; n < 10; ++n) {
      file << " " << broken_flux[n] << " " << max_flux_residual[n];
    }
    file << std::endl;
  }
}

// Constant primitive states at the lower and upper x2 boundaries, as specified by
// Xu & Shu.  The x1 boundaries remain the standard reflective boundaries.
void XuShuBoundary(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &js = indcs.js;
  int &je = indcs.je;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto &u0 = pm->pmb_pack->phydro->u0;
  Real gm1 = pm->pmb_pack->phydro->peos->eos_data.gamma - 1.0;
  int nmb = pm->pmb_pack->nmb_thispack;

  par_for("rt2d_xu_shu_bc_x2", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        u0(m,IDN,k,js-j-1,i) = 2.0;
        u0(m,IM1,k,js-j-1,i) = 0.0;
        u0(m,IM2,k,js-j-1,i) = 0.0;
        u0(m,IM3,k,js-j-1,i) = 0.0;
        u0(m,IEN,k,js-j-1,i) = 1.0/gm1;
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        u0(m,IDN,k,je+j+1,i) = 1.0;
        u0(m,IM1,k,je+j+1,i) = 0.0;
        u0(m,IM2,k,je+j+1,i) = 0.0;
        u0(m,IM3,k,je+j+1,i) = 0.0;
        u0(m,IEN,k,je+j+1,i) = 2.5/gm1;
      }
    }
  });
}

} // namespace
