//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_current_sheet.cpp
//! \brief Self-similar one-dimensional resistive relativistic current-sheet test.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

void SRRMHDCurrentSheetErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ResistiveSRMHDCurrentSheet()
//! \brief Initialize B_y and E_z from the high-pressure self-similar solution.

void ProblemGenerator::ResistiveSRMHDCurrentSheet(ParameterInput *pin,
                                                   const bool restart) {
  pgen_final_func = SRRMHDCurrentSheetErrors;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  if (pmhd == nullptr || !(pmhd->is_resistive_rel) || !(pmy_mesh_->one_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_current_sheet requires one-dimensional "
              << "resistive SRMHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real t0 = pin->GetOrAddReal("problem", "t0", 1.0);
  const Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real pressure = pin->GetOrAddReal("problem", "pressure", 5000.0);
  // For a nonuniform model, initial_eta controls only the resolved seed sheet.
  // The implicit stages subsequently evaluate and freeze their own local eta.
  const Real eta = pin->GetOrAddReal("problem", "initial_eta", pmhd->resistivity);
  if (eta <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_current_sheet requires "
              << "<problem>/initial_eta > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real gamma = pmhd->peos->eos_data.gamma;
  auto &mbsize = pmbp->pmb->mb_size;
  auto w = pmhd->w0;
  auto u = pmhd->u0;
  auto b = pmhd->b0;
  auto bcc = pmhd->bcc0;

  Kokkos::deep_copy(b.x1f, 0.0);
  Kokkos::deep_copy(b.x2f, 0.0);
  Kokkos::deep_copy(b.x3f, 0.0);
  par_for("pgen_srr_current_sheet_by", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    b.x2f(m, k, j, i) = erf(x/(2.0*sqrt(eta*t0)));
  });

  par_for("pgen_srr_current_sheet", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    w(m, IDN, k, j, i) = rho;
    w(m, IVX, k, j, i) = 0.0;
    w(m, IVY, k, j, i) = 0.0;
    w(m, IVZ, k, j, i) = 0.0;
    w(m, IEN, k, j, i) = pressure/(gamma - 1.0);
    w(m, srrmhd::IRE1, k, j, i) = 0.0;
    w(m, srrmhd::IRE2, k, j, i) = 0.0;
    w(m, srrmhd::IRE3, k, j, i) = sqrt(eta/(M_PI*t0))
        *exp(-SQR(x)/(4.0*eta*t0));
    bcc(m, IBX, k, j, i) = 0.0;
    bcc(m, IBY, k, j, i) = erf(x/(2.0*sqrt(eta*t0)));
    bcc(m, IBZ, k, j, i) = 0.0;
  });
  if (pmhd->use_electric_ct) {
    auto e = pmhd->e0;
    Kokkos::deep_copy(e.x1f, 0.0);
    Kokkos::deep_copy(e.x2f, 0.0);
    Kokkos::deep_copy(e.x3f, 0.0);
    par_for("pgen_srr_current_sheet_e3", DevExeSpace(), 0, nmb-1, is, ie,
            KOKKOS_LAMBDA(int m, int i) {
      const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                                 mbsize.d_view(m).x1max);
      const Real e3 = sqrt(eta/(M_PI*t0))*exp(-SQR(x)/(4.0*eta*t0));
      e.x3f(m, ks, js, i) = e3;
      e.x3f(m, ks+1, js, i) = e3;
    });
  }
  pmhd->peos->PrimToCons(w, bcc, u, is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDCurrentSheetErrors()
//! \brief Compare cell-centered B_y and E_z with the analytic profile at t0+t.

void SRRMHDCurrentSheetErrors(ParameterInput *pin, Mesh *pm) {
  auto *pmhd = pm->pmb_pack->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const Real eta = pin->GetOrAddReal("problem", "initial_eta", pmhd->resistivity);
  const Real analytic_time = pin->GetOrAddReal("problem", "t0", 1.0) + pm->time;
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto bcc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bcc0);
  auto mbsize = pm->pmb_pack->pmb->mb_size.h_view;

  Real l1_b = 0.0, linf_b = 0.0;
  Real l1_e = 0.0, linf_e = 0.0;
  int ncells = 0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real x = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                     mbsize(m).x1max);
          const Real by_exact = erf(x/(2.0*sqrt(eta*analytic_time)));
          const Real ez_exact = sqrt(eta/(M_PI*analytic_time))
              *exp(-SQR(x)/(4.0*eta*analytic_time));
          const Real b_error = fabs(bcc(m, IBY, k, j, i) - by_exact);
          const Real e_error = fabs(w(m, srrmhd::IRE3, k, j, i) - ez_exact);
          l1_b += b_error;
          l1_e += e_error;
          linf_b = std::max(linf_b, b_error);
          linf_e = std::max(linf_e, e_error);
          ++ncells;
        }
      }
    }
  }
  Real l1_errors[2] = {l1_b, l1_e};
  Real linf_errors[2] = {linf_b, linf_e};
  int failures = pm->ecounter.neos_fail;
  int max_iterations = pm->ecounter.maxit_c2p;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, l1_errors, 2, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, linf_errors, 2, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ncells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_iterations, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  l1_b = l1_errors[0];
  l1_e = l1_errors[1];
  linf_b = linf_errors[0];
  linf_e = linf_errors[1];
  l1_b /= ncells;
  l1_e /= ncells;

  // Assemble the complete global profile on every rank.  This makes the diagnostic
  // deterministic across one or many MeshBlocks and MPI decompositions.
  const int nx1_global = pm->mesh_indcs.nx1;
  std::vector<Real> profile_data(4*nx1_global, 0.0);
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int i = is; i <= ie; ++i) {
      const Real x = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                 mbsize(m).x1max);
      int gi = static_cast<int>((x - pm->mesh_size.x1min)/pm->mesh_size.dx1);
      gi = std::max(0, std::min(nx1_global-1, gi));
      const Real by_exact = erf(x/(2.0*sqrt(eta*analytic_time)));
      const Real ez_exact = sqrt(eta/(M_PI*analytic_time))
          *exp(-SQR(x)/(4.0*eta*analytic_time));
      profile_data[gi] = bcc(m, IBY, ks, js, i);
      profile_data[nx1_global + gi] = by_exact;
      profile_data[2*nx1_global + gi] = w(m, srrmhd::IRE3, ks, js, i);
      profile_data[3*nx1_global + gi] = ez_exact;
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, profile_data.data(), 4*nx1_global,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (global_variable::my_rank == 0) {
    const std::string basename = pin->GetString("job", "basename");
    const std::string error_name = basename + "-errs.dat";
    std::ifstream old_error_file(error_name);
    const bool error_file_exists = old_error_file.good();
    old_error_file.close();
    std::ofstream file(error_name, std::ios::app);
    if (!error_file_exists) {
      file << "# Nx1 Ncycle L1_By Linf_By L1_Ez Linf_Ez recovery_failures "
           << "max_c2p_iterations dtnew time\n";
    }
    file << std::setprecision(17) << nx1_global << " " << pm->ncycle << " "
         << l1_b << " " << linf_b << " " << l1_e << " " << linf_e << " "
         << failures << " " << max_iterations << " "
         << pmhd->dtnew << " " << pm->time << std::endl;

    std::ofstream profile(basename + "-profile.dat");
    profile << "# x By_numerical By_analytic Ez_numerical Ez_analytic\n";
    profile << std::setprecision(17);
    for (int i = 0; i < nx1_global; ++i) {
      const Real x = CellCenterX(i, nx1_global, pm->mesh_size.x1min,
                                 pm->mesh_size.x1max);
      profile << x << " " << profile_data[i] << " "
              << profile_data[nx1_global + i] << " "
              << profile_data[2*nx1_global + i] << " "
              << profile_data[3*nx1_global + i] << "\n";
    }
  }

  if (pmhd->resistivity_data.model ==
      srrmhd::ResistivityModel::charge_starvation) {
    std::vector<Real> eta_profile(nx1_global, 0.0);
    std::vector<Real> jz_profile(nx1_global, 0.0);
    if (pmhd->use_electric_ct) {
      auto eta_face = Kokkos::create_mirror_view_and_copy(
          HostMemSpace(), pmhd->eta_face.x3f);
      for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
        for (int i = is; i <= ie; ++i) {
          const Real x = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                     mbsize(m).x1max);
          int gi = static_cast<int>((x - pm->mesh_size.x1min)/pm->mesh_size.dx1);
          gi = std::max(0, std::min(nx1_global-1, gi));
          eta_profile[gi] = eta_face(m, ks, js, i);
          jz_profile[gi] = (bcc(m, IBY, ks, js, i+1)
                            - bcc(m, IBY, ks, js, i-1))/(2.0*mbsize(m).dx1);
        }
      }
    } else {
      auto eta_cell = Kokkos::create_mirror_view_and_copy(
          HostMemSpace(), pmhd->eta_cell);
      for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
        for (int i = is; i <= ie; ++i) {
          const Real x = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                     mbsize(m).x1max);
          int gi = static_cast<int>((x - pm->mesh_size.x1min)/pm->mesh_size.dx1);
          gi = std::max(0, std::min(nx1_global-1, gi));
          eta_profile[gi] = eta_cell(m, ks, js, i);
          jz_profile[gi] = (bcc(m, IBY, ks, js, i+1)
                            - bcc(m, IBY, ks, js, i-1))/(2.0*mbsize(m).dx1);
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, eta_profile.data(), nx1_global,
                  MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, jz_profile.data(), nx1_global,
                  MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      const std::string basename = pin->GetString("job", "basename");
      std::ofstream profile(basename + "-eta-profile.dat");
      profile << "# x By Ez Jz eta_frozen\n";
      profile << std::setprecision(17);
      for (int i = 0; i < nx1_global; ++i) {
        const Real x = CellCenterX(i, nx1_global, pm->mesh_size.x1min,
                                   pm->mesh_size.x1max);
        profile << x << " " << profile_data[i] << " "
                << profile_data[2*nx1_global + i] << " " << jz_profile[i]
                << " " << eta_profile[i] << "\n";
      }
    }
  }
}
