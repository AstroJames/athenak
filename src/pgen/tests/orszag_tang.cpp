//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file orszag-tang.c
//  \brief Problem generator for Orszag-Tang vortex problem.
//
// REFERENCE: For example, see: G. Toth,  "The div(B)=0 constraint in shock capturing
//   MHD codes", JCP, 161, 605 (2000)
//========================================================================================

// C++ headers
#include <math.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {
void OrszagTangSymmetryErrors(ParameterInput *pin, Mesh *pm);
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential

KOKKOS_INLINE_FUNCTION
Real A3(const Real x1, const Real x2, const Real B0) {
  return (B0/(4.0*M_PI))*(std::cos(4.0*M_PI*x1) - 2.0*std::cos(2.0*M_PI*x2));
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::OrszagTang_(ParameterInput *pin)
//  \brief Problem Generator for the Orszag-Tang test.  The initial conditions are
//  constructed assuming the domain extends over [-0.5x0.5, -0.5x0.5], so that exact
//  symmetry can be enforced across x=0 and y=0.  An optional 3-D extension preserves
//  exact point-inversion parity and the discrete divergence-free constraint.

void ProblemGenerator::OrszagTang(ParameterInput *pin, const bool restart) {
  if (pin->GetOrAddBoolean("problem", "check_symmetry", false)) {
    pgen_final_func = OrszagTangSymmetryErrors;
  }
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Orszag-Tang test can only be run in MHD, but no <mhd> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  const bool three_d = pin->GetOrAddBoolean("problem", "three_dimensional", false);
  const Real three_d_amp = pin->GetOrAddReal("problem", "three_d_amplitude", 0.2);
  if (three_d && pmy_mesh_->mesh_indcs.nx3 <= 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 3-D Orszag-Tang extension requires mesh/nx3 > 1" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real B0 = 1.0/std::sqrt(4.0*M_PI);
  Real d0 = 25.0/(36.0*M_PI);
  Real v0 = 1.0;
  Real p0 = 5.0/(12.0*M_PI);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &size = pmbp->pmb->mb_size;

  par_for("pgen_ot1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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
    Real perturb = three_d ? three_d_amp : 0.0;

    // compute cell-centered conserved variables
    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) =  d0*v0*std::sin(2.0*M_PI*x2v);
    u0(m,IM2,k,j,i) = -d0*v0*std::sin(2.0*M_PI*x1v);
    u0(m,IM3,k,j,i) =  d0*v0*perturb*std::sin(2.0*M_PI*x3v);

    // Compute face-centered fields from curl(A).
    Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);
    Real x1fp1 = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);
    Real x2fp1 = LeftEdgeX(j+1-js, nx2, x2min, x2max);
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;

    b0.x1f(m,k,j,i) =  (A3(x1f,  x2fp1,B0) - A3(x1f,x2f,B0))/dx2
                          + B0*perturb*std::sin(2.0*M_PI*x3v);
    b0.x2f(m,k,j,i) = -(A3(x1fp1,x2f  ,B0) - A3(x1f,x2f,B0))/dx1
                          + B0*perturb*std::sin(2.0*M_PI*x3v);
    b0.x3f(m,k,j,i) = B0*perturb*(std::sin(2.0*M_PI*x1v)
                                          + std::sin(2.0*M_PI*x2v));

    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) =  (A3(x1fp1,x2fp1,B0) - A3(x1fp1,x2f,B0))/dx2
                              + B0*perturb*std::sin(2.0*M_PI*x3v);
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = -(A3(x1fp1,x2fp1,B0) - A3(x1f,x2fp1,B0))/dx1
                              + B0*perturb*std::sin(2.0*M_PI*x3v);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = B0*perturb*(std::sin(2.0*M_PI*x1v)
                                              + std::sin(2.0*M_PI*x2v));
    }
  });

  // initialize total energy (requires B to be defined across entire grid first)
  par_for("pgen_ot2", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IEN,k,j,i) = p0/gm1 + (0.5/u0(m,IDN,k,j,i))*
         (SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i))) +
          0.5*(SQR(0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1))) +
               SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i))) +
               SQR(0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i))));
  });

  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \brief Measure exact half-turn or point-inversion parity of the final MHD state.

void OrszagTangSymmetryErrors(ParameterInput *pin, Mesh *pm) {
  if (pm->multilevel) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Orszag-Tang symmetry diagnostics require a uniform mesh"
                << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }

  constexpr int nfields = 13;
  const std::array<const char *, nfields> labels = {
      "u_d", "u_m1", "u_m2", "u_m3", "u_e",
      "w_d", "w_v1", "w_v2", "w_v3", "w_e", "b1", "b2", "b3"};
  const bool three_d = pin->GetOrAddBoolean("problem", "three_dimensional", false);
  std::array<int, nfields> parity = {
      1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1};
  if (three_d) {
    parity = {1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1};
  }

  auto *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u0);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto bcc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bcc0);
  auto b1 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x1f);
  auto b2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x2f);
  auto b3 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x3f);
  auto e1 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->efld.x1e);
  auto e2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->efld.x2e);
  auto e3 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->efld.x3e);
  const auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1_mb = indcs.nx1, nx2_mb = indcs.nx2, nx3_mb = indcs.nx3;
  const int nx1 = pm->mesh_indcs.nx1;
  const int nx2 = pm->mesh_indcs.nx2;
  const int nx3 = pm->mesh_indcs.nx3;
  const int ncells = nx1*nx2*nx3;
  const int nfaces1 = (nx1 + 1)*nx2*nx3;
  const int nfaces2 = nx1*(nx2 + 1)*nx3;
  const int nfaces3 = nx1*nx2*(nx3 + 1);
  const int nedges1 = nx1*(nx2 + 1)*(nx3 + 1);
  const int nedges2 = (nx1 + 1)*nx2*(nx3 + 1);
  const int nedges3 = (nx1 + 1)*(nx2 + 1)*nx3;
  std::vector<Real> state(nfields*ncells, 0.0);
  std::vector<Real> face1(three_d ? nfaces1 : 0, 0.0);
  std::vector<Real> face2(three_d ? nfaces2 : 0, 0.0);
  std::vector<Real> face3(three_d ? nfaces3 : 0, 0.0);
  std::vector<Real> edge1(three_d ? nedges1 : 0, 0.0);
  std::vector<Real> edge2(three_d ? nedges2 : 0, 0.0);
  std::vector<Real> edge3(three_d ? nedges3 : 0, 0.0);

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const int gid = pmbp->pmb->mb_gid.h_view(m);
    const LogicalLocation &lloc = pm->lloc_eachmb[gid];
    const int gi0 = lloc.lx1*nx1_mb;
    const int gj0 = lloc.lx2*nx2_mb;
    const int gk0 = lloc.lx3*nx3_mb;
    for (int k = ks; k < ks + nx3_mb; ++k) {
      for (int j = js; j < js + nx2_mb; ++j) {
        for (int i = is; i < is + nx1_mb; ++i) {
          const int gi = gi0 + i - is;
          const int gj = gj0 + j - js;
          const int gk = gk0 + k - ks;
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
          state[10*ncells + idx] = bcc(m, IBX, k, j, i);
          state[11*ncells + idx] = bcc(m, IBY, k, j, i);
          state[12*ncells + idx] = bcc(m, IBZ, k, j, i);
        }
      }
    }

    if (three_d) {
      const int ni_face = nx1_mb + (gi0 + nx1_mb == nx1 ? 1 : 0);
      const int nj_face = nx2_mb + (gj0 + nx2_mb == nx2 ? 1 : 0);
      const int nk_face = nx3_mb + (gk0 + nx3_mb == nx3 ? 1 : 0);

      // Each interior shared face or edge is owned by the block on its upper side.
      // The block at the global upper boundary additionally owns the duplicated endpoint.
      for (int lk = 0; lk < nx3_mb; ++lk) {
        const int gk = gk0 + lk;
        for (int lj = 0; lj < nx2_mb; ++lj) {
          const int gj = gj0 + lj;
          for (int li = 0; li < ni_face; ++li) {
            const int gf = gi0 + li;
            const int idx = (gk*nx2 + gj)*(nx1 + 1) + gf;
            face1[idx] = b1(m, ks + lk, js + lj, is + li);
          }
        }
      }
      for (int lk = 0; lk < nx3_mb; ++lk) {
        const int gk = gk0 + lk;
        for (int lj = 0; lj < nj_face; ++lj) {
          const int gf = gj0 + lj;
          for (int li = 0; li < nx1_mb; ++li) {
            const int gi = gi0 + li;
            const int idx = (gk*(nx2 + 1) + gf)*nx1 + gi;
            face2[idx] = b2(m, ks + lk, js + lj, is + li);
          }
        }
      }
      for (int lk = 0; lk < nk_face; ++lk) {
        const int gf = gk0 + lk;
        for (int lj = 0; lj < nx2_mb; ++lj) {
          const int gj = gj0 + lj;
          for (int li = 0; li < nx1_mb; ++li) {
            const int gi = gi0 + li;
            const int idx = (gf*nx2 + gj)*nx1 + gi;
            face3[idx] = b3(m, ks + lk, js + lj, is + li);
          }
        }
      }

      for (int lk = 0; lk < nk_face; ++lk) {
        const int gf3 = gk0 + lk;
        for (int lj = 0; lj < nj_face; ++lj) {
          const int gf2 = gj0 + lj;
          for (int li = 0; li < nx1_mb; ++li) {
            const int gi = gi0 + li;
            const int idx = (gf3*(nx2 + 1) + gf2)*nx1 + gi;
            edge1[idx] = e1(m, ks + lk, js + lj, is + li);
          }
        }
      }
      for (int lk = 0; lk < nk_face; ++lk) {
        const int gf3 = gk0 + lk;
        for (int lj = 0; lj < nx2_mb; ++lj) {
          const int gj = gj0 + lj;
          for (int li = 0; li < ni_face; ++li) {
            const int gf1 = gi0 + li;
            const int idx = (gf3*nx2 + gj)*(nx1 + 1) + gf1;
            edge2[idx] = e2(m, ks + lk, js + lj, is + li);
          }
        }
      }
      for (int lk = 0; lk < nx3_mb; ++lk) {
        const int gk = gk0 + lk;
        for (int lj = 0; lj < nj_face; ++lj) {
          const int gf2 = gj0 + lj;
          for (int li = 0; li < ni_face; ++li) {
            const int gf1 = gi0 + li;
            const int idx = (gk*(nx2 + 1) + gf2)*(nx1 + 1) + gf1;
            edge3[idx] = e3(m, ks + lk, js + lj, is + li);
          }
        }
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, state.data(), nfields*ncells, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  if (three_d) {
    MPI_Allreduce(MPI_IN_PLACE, face1.data(), nfaces1, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, face2.data(), nfaces2, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, face3.data(), nfaces3, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, edge1.data(), nedges1, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, edge2.data(), nedges2, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, edge3.data(), nedges3, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif

  if (global_variable::my_rank == 0) {
    std::array<std::int64_t, nfields> broken{};
    std::array<Real, nfields> max_residual{};
    std::array<std::int64_t, 3> broken_face{};
    std::array<Real, 3> max_face_residual{};
    std::array<std::int64_t, 3> broken_edge{};
    std::array<Real, 3> max_edge_residual{};
    std::array<Real, 3> max_edge_magnitude{};
    for (int gk = 0; gk < nx3; ++gk) {
      for (int gj = 0; gj < nx2; ++gj) {
        for (int gi = 0; gi < nx1; ++gi) {
          const int idx = (gk*nx2 + gj)*nx1 + gi;
          const int reflected_k = three_d ? nx3 - 1 - gk : gk;
          const int opposite = (reflected_k*nx2 + (nx2 - 1 - gj))*nx1
                             + (nx1 - 1 - gi);
          for (int n = 0; n < nfields; ++n) {
            const Real residual = state[n*ncells + idx]
                                - parity[n]*state[n*ncells + opposite];
            if (residual != 0.0) {
              ++broken[n];
              max_residual[n] = std::max(max_residual[n], std::abs(residual));
            }
          }
        }
      }
    }

    if (three_d) {
      for (int gk = 0; gk < nx3; ++gk) {
        for (int gj = 0; gj < nx2; ++gj) {
          for (int gf = 0; gf <= nx1; ++gf) {
            const int idx = (gk*nx2 + gj)*(nx1 + 1) + gf;
            const int reflected = ((nx3 - 1 - gk)*nx2 + (nx2 - 1 - gj))
                                *(nx1 + 1) + (nx1 - gf);
            const Real residual = face1[idx] + face1[reflected];
            if (residual != 0.0) {
              ++broken_face[0];
              max_face_residual[0] = std::max(max_face_residual[0],
                                               std::abs(residual));
            }
          }
        }
      }
      for (int gk = 0; gk < nx3; ++gk) {
        for (int gf = 0; gf <= nx2; ++gf) {
          for (int gi = 0; gi < nx1; ++gi) {
            const int idx = (gk*(nx2 + 1) + gf)*nx1 + gi;
            const int reflected = ((nx3 - 1 - gk)*(nx2 + 1) + (nx2 - gf))*nx1
                                + (nx1 - 1 - gi);
            const Real residual = face2[idx] + face2[reflected];
            if (residual != 0.0) {
              ++broken_face[1];
              max_face_residual[1] = std::max(max_face_residual[1],
                                               std::abs(residual));
            }
          }
        }
      }
      for (int gf = 0; gf <= nx3; ++gf) {
        for (int gj = 0; gj < nx2; ++gj) {
          for (int gi = 0; gi < nx1; ++gi) {
            const int idx = (gf*nx2 + gj)*nx1 + gi;
            const int reflected = ((nx3 - gf)*nx2 + (nx2 - 1 - gj))*nx1
                                + (nx1 - 1 - gi);
            const Real residual = face3[idx] + face3[reflected];
            if (residual != 0.0) {
              ++broken_face[2];
              max_face_residual[2] = std::max(max_face_residual[2],
                                               std::abs(residual));
            }
          }
        }
      }

      for (int gf3 = 0; gf3 <= nx3; ++gf3) {
        for (int gf2 = 0; gf2 <= nx2; ++gf2) {
          for (int gi = 0; gi < nx1; ++gi) {
            const int idx = (gf3*(nx2 + 1) + gf2)*nx1 + gi;
            const int reflected = ((nx3 - gf3)*(nx2 + 1) + (nx2 - gf2))*nx1
                                + (nx1 - 1 - gi);
            const Real residual = edge1[idx] - edge1[reflected];
            max_edge_magnitude[0] = std::max(max_edge_magnitude[0],
                                              std::abs(edge1[idx]));
            if (residual != 0.0) {
              ++broken_edge[0];
              max_edge_residual[0] = std::max(max_edge_residual[0],
                                               std::abs(residual));
            }
          }
        }
      }
      for (int gf3 = 0; gf3 <= nx3; ++gf3) {
        for (int gj = 0; gj < nx2; ++gj) {
          for (int gf1 = 0; gf1 <= nx1; ++gf1) {
            const int idx = (gf3*nx2 + gj)*(nx1 + 1) + gf1;
            const int reflected = ((nx3 - gf3)*nx2 + (nx2 - 1 - gj))*(nx1 + 1)
                                + (nx1 - gf1);
            const Real residual = edge2[idx] - edge2[reflected];
            max_edge_magnitude[1] = std::max(max_edge_magnitude[1],
                                              std::abs(edge2[idx]));
            if (residual != 0.0) {
              ++broken_edge[1];
              max_edge_residual[1] = std::max(max_edge_residual[1],
                                               std::abs(residual));
            }
          }
        }
      }
      for (int gk = 0; gk < nx3; ++gk) {
        for (int gf2 = 0; gf2 <= nx2; ++gf2) {
          for (int gf1 = 0; gf1 <= nx1; ++gf1) {
            const int idx = (gk*(nx2 + 1) + gf2)*(nx1 + 1) + gf1;
            const int reflected = ((nx3 - 1 - gk)*(nx2 + 1) + (nx2 - gf2))
                                *(nx1 + 1) + (nx1 - gf1);
            const Real residual = edge3[idx] - edge3[reflected];
            max_edge_magnitude[2] = std::max(max_edge_magnitude[2],
                                              std::abs(edge3[idx]));
            if (residual != 0.0) {
              ++broken_edge[2];
              max_edge_residual[2] = std::max(max_edge_residual[2],
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
    if (three_d) {
      for (const char *field : {"b1f", "b2f", "b3f"}) {
        file << " n_" << field << " max_" << field;
      }
      for (const char *field : {"e1", "e2", "e3"}) {
        file << " n_" << field << " max_" << field << " mag_" << field;
      }
    }
    file << "\n" << std::setprecision(17) << nx1 << " " << nx2 << " " << nx3
         << " " << pm->ncycle << " " << pm->time;
    for (int n = 0; n < nfields; ++n) {
      file << " " << broken[n] << " " << max_residual[n];
    }
    if (three_d) {
      for (int n = 0; n < 3; ++n) {
        file << " " << broken_face[n] << " " << max_face_residual[n];
      }
      for (int n = 0; n < 3; ++n) {
        file << " " << broken_edge[n] << " " << max_edge_residual[n]
             << " " << max_edge_magnitude[n];
      }
    }
    file << std::endl;
  }
}

} // namespace
