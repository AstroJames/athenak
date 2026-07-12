//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_ect.cpp
//! \brief Manufactured compatible-operator test for charge-conserving dual CT.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "mhd/dual_ct.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
Real Electric1(const Real x, const Real y, const Real z) {
  return 0.3 + x*x + 0.125*y - 0.0625*z;
}

KOKKOS_INLINE_FUNCTION
Real Electric2(const Real x, const Real y, const Real z) {
  return -0.2 + y*y - 0.0625*x + 0.03125*z;
}

KOKKOS_INLINE_FUNCTION
Real Electric3(const Real x, const Real y, const Real z) {
  return 0.1 + z*z + 0.03125*x - 0.015625*y;
}

KOKKOS_INLINE_FUNCTION
Real Current1(const Real x, const Real y, const Real z) {
  return 0.7*x*x + 0.1*y + 0.05*z;
}

KOKKOS_INLINE_FUNCTION
Real Current2(const Real x, const Real y, const Real z) {
  return -0.4*y*y + 0.07*x - 0.03*z;
}

KOKKOS_INLINE_FUNCTION
Real Current3(const Real x, const Real y, const Real z) {
  return 0.2*z*z - 0.02*x + 0.04*y;
}

KOKKOS_INLINE_FUNCTION
Real DynamicElectric1(const Real x, const Real amplitude) {
  constexpr Real two_pi = 6.283185307179586476925286766559;
  return amplitude*sin(two_pi*x);
}

} // namespace

void SRRMHDECTErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDECTDynamicErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ResistiveSRMHDECT()
//! \brief Initialize face E, edge B, and one manufactured Ampere update.

void ProblemGenerator::ResistiveSRMHDECT(ParameterInput *pin, const bool restart) {
  pgen_final_func = pmy_mesh_->three_d ? SRRMHDECTErrors : SRRMHDECTDynamicErrors;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  if (!(pmy_mesh_->three_d)) {
    if (pmhd == nullptr || !(pmhd->is_resistive_rel) || !(pmhd->use_electric_ct) ||
        pmy_mesh_->multi_d) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dynamic rsrmhd_ect requires one-dimensional "
                << "resistive SRMHD with <mhd>/electric_ct=true" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    auto &indcs = pmy_mesh_->mb_indcs;
    const int is = indcs.is, ie = indcs.ie;
    const int js = indcs.js, ks = indcs.ks;
    const int n1 = indcs.nx1 + 2*indcs.ng;
    const int nmb = pmbp->nmb_thispack;
    const Real amplitude = pin->GetOrAddReal("problem", "electric_amplitude", 0.05);
    auto &mbsize = pmbp->pmb->mb_size;
    auto e = pmhd->e0;
    auto b = pmhd->b0;
    auto bcc = pmhd->bcc0;
    auto w = pmhd->w0;
    auto u = pmhd->u0;

    Kokkos::deep_copy(b.x1f, 0.0);
    Kokkos::deep_copy(b.x2f, 0.0);
    Kokkos::deep_copy(b.x3f, 0.0);
    Kokkos::deep_copy(e.x1f, 0.0);
    Kokkos::deep_copy(e.x2f, 0.0);
    Kokkos::deep_copy(e.x3f, 0.0);

    // Initialize normal E on every x1 face, including periodic ghost faces.
    par_for("pgen_ect_dynamic_e1", DevExeSpace(), 0, nmb-1, 0, n1,
    KOKKOS_LAMBDA(int m, int i) {
      const Real x = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
      e.x1f(m,ks,js,i) = DynamicElectric1(x, amplitude);
    });

    const Real gamma = pmhd->peos->eos_data.gamma;
    par_for("pgen_ect_dynamic_fluid", DevExeSpace(), 0, nmb-1, ks, ks, js, js,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real ec1, ec2, ec3;
      srrmhd::ElectricFaceToCell(e, m, k, j, i, ec1, ec2, ec3);
      w(m,IDN,k,j,i) = 1.0;
      w(m,IVX,k,j,i) = 0.0;
      w(m,IVY,k,j,i) = 0.0;
      w(m,IVZ,k,j,i) = 0.0;
      w(m,IEN,k,j,i) = 1.0/(gamma - 1.0);
      w(m,srrmhd::IRE1,k,j,i) = ec1;
      w(m,srrmhd::IRE2,k,j,i) = ec2;
      w(m,srrmhd::IRE3,k,j,i) = ec3;
      bcc(m,IBX,k,j,i) = 0.0;
      bcc(m,IBY,k,j,i) = 0.0;
      bcc(m,IBZ,k,j,i) = 0.0;
    });
    pmhd->peos->PrimToCons(w, bcc, u, is, ie, js, js, ks, ks);
    return;
  }

  if (pmhd == nullptr || !(pmhd->is_resistive_rel) || !(pmhd->use_electric_ct) ||
      !(pmy_mesh_->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_ect requires three-dimensional resistive SRMHD "
              << "with <mhd>/electric_ct=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real dt = pin->GetOrAddReal("problem", "operator_dt", 0.125);
  auto &mbsize = pmbp->pmb->mb_size;
  auto e0 = pmhd->e0;
  auto e1 = pmhd->e1;
  auto bedge = pmhd->bfld;
  auto b = pmhd->b0;
  auto bcc = pmhd->bcc0;
  auto w = pmhd->w0;
  auto u = pmhd->u0;

  Kokkos::deep_copy(b.x1f, 0.0);
  Kokkos::deep_copy(b.x2f, 0.0);
  Kokkos::deep_copy(b.x3f, 0.0);

  par_for("pgen_ect_e1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                               mbsize.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                               mbsize.d_view(m).x3max);
    e0.x1f(m,k,j,i) = Electric1(x, y, z);
  });
  par_for("pgen_ect_e2", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    const Real y = LeftEdgeX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                             mbsize.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                               mbsize.d_view(m).x3max);
    e0.x2f(m,k,j,i) = Electric2(x, y, z);
  });
  par_for("pgen_ect_e3", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                               mbsize.d_view(m).x2max);
    const Real z = LeftEdgeX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                             mbsize.d_view(m).x3max);
    e0.x3f(m,k,j,i) = Electric3(x, y, z);
  });

  // Manufactured line-averaged B: curl(B)=(2*x, -3*y, z), so div(curl(B))=0.
  par_for("pgen_ect_b1edge", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real y = LeftEdgeX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                             mbsize.d_view(m).x2max);
    const Real z = LeftEdgeX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                             mbsize.d_view(m).x3max);
    bedge.x1e(m,k,j,i) = 2.0*y*z;
  });
  par_for("pgen_ect_b2edge", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
    const Real z = LeftEdgeX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                             mbsize.d_view(m).x3max);
    bedge.x2e(m,k,j,i) = 3.0*z*x;
  });
  par_for("pgen_ect_b3edge", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
    const Real y = LeftEdgeX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                             mbsize.d_view(m).x2max);
    bedge.x3e(m,k,j,i) = 5.0*x*y;
  });

  par_for("pgen_ect_update_e1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                               mbsize.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                               mbsize.d_view(m).x3max);
    const Real curl = srrmhd::EdgeCurl1(bedge, m, k, j, i,
        1.0/mbsize.d_view(m).dx2, 1.0/mbsize.d_view(m).dx3, true, true);
    e1.x1f(m,k,j,i) = e0.x1f(m,k,j,i) + dt*(curl - Current1(x, y, z));
  });
  par_for("pgen_ect_update_e2", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    const Real y = LeftEdgeX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                             mbsize.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                               mbsize.d_view(m).x3max);
    const Real curl = srrmhd::EdgeCurl2(bedge, m, k, j, i,
        1.0/mbsize.d_view(m).dx1, 1.0/mbsize.d_view(m).dx3, true);
    e1.x2f(m,k,j,i) = e0.x2f(m,k,j,i) + dt*(curl - Current2(x, y, z));
  });
  par_for("pgen_ect_update_e3", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                               mbsize.d_view(m).x2max);
    const Real z = LeftEdgeX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                             mbsize.d_view(m).x3max);
    const Real curl = srrmhd::EdgeCurl3(bedge, m, k, j, i,
        1.0/mbsize.d_view(m).dx1, 1.0/mbsize.d_view(m).dx2, true);
    e1.x3f(m,k,j,i) = e0.x3f(m,k,j,i) + dt*(curl - Current3(x, y, z));
  });

  const Real gamma = pmhd->peos->eos_data.gamma;
  par_for("pgen_ect_fluid", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real ec1, ec2, ec3;
    srrmhd::ElectricFaceToCell(e0, m, k, j, i, ec1, ec2, ec3);
    w(m, IDN, k, j, i) = 1.0;
    w(m, IVX, k, j, i) = 0.0;
    w(m, IVY, k, j, i) = 0.0;
    w(m, IVZ, k, j, i) = 0.0;
    w(m, IEN, k, j, i) = 1.0/(gamma - 1.0);
    w(m, srrmhd::IRE1, k, j, i) = ec1;
    w(m, srrmhd::IRE2, k, j, i) = ec2;
    w(m, srrmhd::IRE3, k, j, i) = ec3;
    bcc(m, IBX, k, j, i) = 0.0;
    bcc(m, IBY, k, j, i) = 0.0;
    bcc(m, IBZ, k, j, i) = 0.0;
  });
  pmhd->peos->PrimToCons(w, bcc, u, is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDECTDynamicErrors()
//! \brief Check a complete forward-Euler face-Ampere charge-continuity update.

void SRRMHDECTDynamicErrors(ParameterInput *pin, Mesh *pm) {
  auto *pmhd = pm->pmb_pack->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, ks = indcs.ks;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const Real amplitude = pin->GetOrAddReal("problem", "electric_amplitude", 0.05);
  const Real eta = pmhd->resistivity;
  const Real dt = pm->time;
  const Real h = dt/eta;
  const Real diagonal = 1.0 + 0.5*h;
  const Real stage0_factor = 1.0/diagonal;
  const Real stage1_factor = stage0_factor*(1.0 + h)/diagonal;
  const Real stage2_factor = (1.0 - 0.5*h*stage0_factor)/diagonal;
  const Real final_factor = 0.5*(stage2_factor + 1.0)
      - 0.25*h*(stage1_factor + stage2_factor);
  auto &mbsize = pm->pmb_pack->pmb->mb_size;
  auto e = pmhd->e0;
  auto en = pmhd->e1;
  auto re = pmhd->ect_src;
  auto w = pmhd->w0;

  Real charge_error = 0.0;
  Kokkos::parallel_reduce("ect_dynamic_charge_error",
  Kokkos::MDRangePolicy<Kokkos::Rank<2>>(DevExeSpace(), {0, is}, {nmb, ie+1}),
  KOKKOS_LAMBDA(int m, int i, Real &max_error) {
    const Real idx1 = 1.0/mbsize.d_view(m).dx1;
    const Real q1 = (e.x1f(m,ks,js,i+1) - e.x1f(m,ks,js,i))*idx1;
    const Real stage2_left = en.x1f(m,ks,js,i)
        + 0.5*dt*re.x1f(m,0,ks,js,i) + 0.5*dt*re.x1f(m,2,ks,js,i);
    const Real stage2_right = en.x1f(m,ks,js,i+1)
        + 0.5*dt*re.x1f(m,0,ks,js,i+1) + 0.5*dt*re.x1f(m,2,ks,js,i+1);
    const Real pred_left = 0.5*(stage2_left + en.x1f(m,ks,js,i))
        + 0.25*dt*(re.x1f(m,1,ks,js,i) + re.x1f(m,2,ks,js,i));
    const Real pred_right = 0.5*(stage2_right + en.x1f(m,ks,js,i+1))
        + 0.25*dt*(re.x1f(m,1,ks,js,i+1) + re.x1f(m,2,ks,js,i+1));
    const Real qpred = (pred_right - pred_left)*idx1;
    max_error = fmax(max_error, fabs(q1 - qpred));
  }, Kokkos::Max<Real>(charge_error));

  Real face_error = 0.0;
  Kokkos::parallel_reduce("ect_dynamic_face_error",
  Kokkos::MDRangePolicy<Kokkos::Rank<2>>(DevExeSpace(), {0, is}, {nmb, ie+1}),
  KOKKOS_LAMBDA(int m, int i, Real &max_error) {
    const Real x = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
    const Real einit = DynamicElectric1(x, amplitude);
    max_error = fmax(max_error,
        fabs(e.x1f(m,ks,js,i) - final_factor*einit));
  }, Kokkos::Max<Real>(face_error));

  Real source_error = 0.0;
  Kokkos::parallel_reduce("ect_dynamic_source_error",
  Kokkos::MDRangePolicy<Kokkos::Rank<2>>(DevExeSpace(), {0, is}, {nmb, ie+1}),
  KOKKOS_LAMBDA(int m, int i, Real &max_error) {
    const Real x = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
    const Real einit = DynamicElectric1(x, amplitude);
    Real error = fabs(re.x1f(m,0,ks,js,i) + stage0_factor*einit/eta);
    error = fmax(error,
        fabs(re.x1f(m,1,ks,js,i) + stage1_factor*einit/eta));
    error = fmax(error,
        fabs(re.x1f(m,2,ks,js,i) + stage2_factor*einit/eta));
    max_error = fmax(max_error, error);
  }, Kokkos::Max<Real>(source_error));

  Real mirror_error = 0.0;
  Kokkos::parallel_reduce("ect_dynamic_mirror_error",
  Kokkos::MDRangePolicy<Kokkos::Rank<2>>(DevExeSpace(), {0, is}, {nmb, ie+1}),
  KOKKOS_LAMBDA(int m, int i, Real &max_error) {
    const Real ec1 = 0.5*(e.x1f(m,ks,js,i) + e.x1f(m,ks,js,i+1));
    max_error = fmax(max_error, fabs(w(m,srrmhd::IRE1,ks,js,i) - ec1));
  }, Kokkos::Max<Real>(mirror_error));

#if MPI_PARALLEL_ENABLED
  Real errors[4] = {charge_error, face_error, source_error, mirror_error};
  MPI_Allreduce(MPI_IN_PLACE, errors, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  charge_error = errors[0];
  face_error = errors[1];
  source_error = errors[2];
  mirror_error = errors[3];
#endif

  if (global_variable::my_rank == 0) {
    const std::string basename = pin->GetString("job", "basename");
    std::ofstream file(basename + "-errs.dat");
    file << "# max_charge_residual max_face_update_error max_source_error "
         << "max_cell_mirror_error recovery_failures cycles\n";
    file << std::setprecision(17) << charge_error << " " << face_error << " "
         << source_error << " " << mirror_error << " "
         << pm->ecounter.neos_fail << " " << pm->ncycle << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDECTErrors()
//! \brief Verify div(curl)=0, conservative charge update, and face-to-cell averaging.

void SRRMHDECTErrors(ParameterInput *pin, Mesh *pm) {
  auto *pmhd = pm->pmb_pack->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const Real dt = pin->GetOrAddReal("problem", "operator_dt", 0.125);
  auto &mbsize = pm->pmb_pack->pmb->mb_size;
  auto e0 = pmhd->e0;
  auto e1 = pmhd->e1;
  auto bedge = pmhd->bfld;
  auto w = pmhd->w0;

  Real max_charge_residual = 0.0;
  Kokkos::parallel_reduce("ect_charge_residual",
  Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0, ks, js, is},
                                         {nmb, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &max_error) {
    const Real idx1 = 1.0/mbsize.d_view(m).dx1;
    const Real idx2 = 1.0/mbsize.d_view(m).dx2;
    const Real idx3 = 1.0/mbsize.d_view(m).dx3;
    const Real q0 = srrmhd::FaceDivergence(e0, m, k, j, i, idx1, idx2, idx3,
                                           true, true);
    const Real q1 = srrmhd::FaceDivergence(e1, m, k, j, i, idx1, idx2, idx3,
                                           true, true);
    const Real x0 = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                              mbsize.d_view(m).x1max);
    const Real x1 = LeftEdgeX(i-is+1, indcs.nx1, mbsize.d_view(m).x1min,
                              mbsize.d_view(m).x1max);
    const Real y0 = LeftEdgeX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                              mbsize.d_view(m).x2max);
    const Real y1 = LeftEdgeX(j-js+1, indcs.nx2, mbsize.d_view(m).x2min,
                              mbsize.d_view(m).x2max);
    const Real z0 = LeftEdgeX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                              mbsize.d_view(m).x3max);
    const Real z1 = LeftEdgeX(k-ks+1, indcs.nx3, mbsize.d_view(m).x3min,
                              mbsize.d_view(m).x3max);
    const Real xc = 0.5*(x0 + x1);
    const Real yc = 0.5*(y0 + y1);
    const Real zc = 0.5*(z0 + z1);
    const Real divj = (Current1(x1,yc,zc) - Current1(x0,yc,zc))*idx1
        +(Current2(xc,y1,zc) - Current2(xc,y0,zc))*idx2
        +(Current3(xc,yc,z1) - Current3(xc,yc,z0))*idx3;
    const Real error = fabs(q1 - q0 + dt*divj);
    max_error = (error > max_error) ? error : max_error;
  }, Kokkos::Max<Real>(max_charge_residual));

  Real max_curl_error = 0.0;
  Kokkos::parallel_reduce("ect_curl_error",
  Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0, ks, js, is},
                                         {nmb, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &max_error) {
    const Real x = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
    const Real y = LeftEdgeX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                             mbsize.d_view(m).x2max);
    const Real z = LeftEdgeX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                             mbsize.d_view(m).x3max);
    const Real c1 = srrmhd::EdgeCurl1(bedge, m, k, j, i,
        1.0/mbsize.d_view(m).dx2, 1.0/mbsize.d_view(m).dx3, true, true);
    const Real c2 = srrmhd::EdgeCurl2(bedge, m, k, j, i,
        1.0/mbsize.d_view(m).dx1, 1.0/mbsize.d_view(m).dx3, true);
    const Real c3 = srrmhd::EdgeCurl3(bedge, m, k, j, i,
        1.0/mbsize.d_view(m).dx1, 1.0/mbsize.d_view(m).dx2, true);
    Real error = fabs(c1 - 2.0*x);
    error = fmax(error, fabs(c2 + 3.0*y));
    error = fmax(error, fabs(c3 - z));
    max_error = (error > max_error) ? error : max_error;
  }, Kokkos::Max<Real>(max_curl_error));

  Real max_average_error = 0.0;
  Kokkos::parallel_reduce("ect_average_error",
  Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0, ks, js, is},
                                         {nmb, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &max_error) {
    Real ec1, ec2, ec3;
    srrmhd::ElectricFaceToCell(e0, m, k, j, i, ec1, ec2, ec3);
    Real error = fabs(ec1 - w(m,srrmhd::IRE1,k,j,i));
    error = fmax(error, fabs(ec2 - w(m,srrmhd::IRE2,k,j,i)));
    error = fmax(error, fabs(ec3 - w(m,srrmhd::IRE3,k,j,i)));
    max_error = (error > max_error) ? error : max_error;
  }, Kokkos::Max<Real>(max_average_error));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &max_charge_residual, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_curl_error, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_average_error, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif

  if (global_variable::my_rank == 0) {
    const std::string basename = pin->GetString("job", "basename");
    std::ofstream file(basename + "-errs.dat");
    file << "# max_charge_residual max_curl_error max_average_error electric_ct "
         << "recovery_failures\n";
    file << std::setprecision(17) << max_charge_residual << " " << max_curl_error
         << " " << max_average_error << " " << static_cast<int>(pmhd->use_electric_ct)
         << " " << pm->ecounter.neos_fail << std::endl;
  }
}
