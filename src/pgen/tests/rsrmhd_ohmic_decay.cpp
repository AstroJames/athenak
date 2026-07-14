//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_ohmic_decay.cpp
//! \brief Ohmic spreading of a strong-guide-field relativistic Harris sheet.

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

namespace {

void SRRMHDOhmicDecayErrors(ParameterInput *pin, Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! \brief Initialize the guide-field-balanced Harris sheet of Grehan et al. (2025).

void ProblemGenerator::ResistiveSRMHDOhmicDecay(ParameterInput *pin,
                                                 const bool restart) {
  pgen_final_func = SRRMHDOhmicDecayErrors;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  if (pmhd == nullptr || !(pmhd->is_resistive_rel) || !(pmy_mesh_->one_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_ohmic_decay requires one-dimensional "
              << "resistive SRMHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmhd->resistivity_data.model != srrmhd::ResistivityModel::uniform) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_ohmic_decay requires uniform resistivity"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real sheet_width = pin->GetOrAddReal("problem", "sheet_width", 0.02);
  const Real field = pin->GetOrAddReal("problem", "field", 1.0);
  const Real guide_field = pin->GetOrAddReal("problem", "guide_field", 10.0);
  const Real hot_magnetization = pin->GetOrAddReal(
      "problem", "hot_magnetization", 10.0);
  const Real temperature = pin->GetOrAddReal("problem", "temperature", 1.0);
  if (sheet_width <= 0.0 || field <= 0.0 || guide_field <= field
      || hot_magnetization <= 0.0 || temperature <= 0.0
      || pmhd->resistivity <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_ohmic_decay requires sheet_width, field, "
              << "hot_magnetization, temperature, and resistivity > 0, with "
              << "guide_field > field" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real gamma = pmhd->peos->eos_data.gamma;
  const Real enthalpy_per_density = 1.0 + gamma*temperature/(gamma - 1.0);
  const Real rho = (SQR(field) + SQR(guide_field))
      /(hot_magnetization*enthalpy_per_density);
  const Real pressure = rho*temperature;
  auto &mbsize = pmbp->pmb->mb_size;
  auto w = pmhd->w0;
  auto u = pmhd->u0;
  auto b = pmhd->b0;
  auto bcc = pmhd->bcc0;

  Kokkos::deep_copy(b.x1f, 0.0);
  Kokkos::deep_copy(b.x2f, 0.0);
  Kokkos::deep_copy(b.x3f, 0.0);
  par_for("pgen_srr_ohmic_by", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    b.x2f(m, k, j, i) = field*tanh(x/sheet_width);
  });
  par_for("pgen_srr_ohmic_bz", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    const Real by = field*tanh(x/sheet_width);
    b.x3f(m, k, j, i) = sqrt(SQR(guide_field) + SQR(field) - SQR(by));
  });

  par_for("pgen_srr_ohmic_decay", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    const Real by = field*tanh(x/sheet_width);
    w(m, IDN, k, j, i) = rho;
    w(m, IVX, k, j, i) = 0.0;
    w(m, IVY, k, j, i) = 0.0;
    w(m, IVZ, k, j, i) = 0.0;
    w(m, IEN, k, j, i) = pressure/(gamma - 1.0);
    w(m, srrmhd::IRE1, k, j, i) = 0.0;
    w(m, srrmhd::IRE2, k, j, i) = 0.0;
    w(m, srrmhd::IRE3, k, j, i) = 0.0;
    bcc(m, IBX, k, j, i) = 0.0;
    bcc(m, IBY, k, j, i) = by;
    bcc(m, IBZ, k, j, i) = sqrt(
        SQR(guide_field) + SQR(field) - SQR(by));
  });
  if (pmhd->use_electric_ct) {
    Kokkos::deep_copy(pmhd->e0.x1f, 0.0);
    Kokkos::deep_copy(pmhd->e0.x2f, 0.0);
    Kokkos::deep_copy(pmhd->e0.x3f, 0.0);
  }
  pmhd->peos->PrimToCons(w, bcc, u, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! \brief Measure peak current, current-squared width, and the final profile.

void SRRMHDOhmicDecayErrors(ParameterInput *pin, Mesh *pm) {
  auto *pmhd = pm->pmb_pack->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, ks = indcs.ks;
  const int nx1_global = pm->mesh_indcs.nx1;
  const Real sheet_width = pin->GetOrAddReal("problem", "sheet_width", 0.02);
  const Real field = pin->GetOrAddReal("problem", "field", 1.0);
  const Real eta = pmhd->resistivity;
  const Real shifted_time = pm->time + SQR(sheet_width)/(M_PI*eta);
  const Real peak_model = field/sqrt(M_PI*eta*shifted_time);
  const Real moment_model = eta*shifted_time;
  auto bcc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bcc0);
  auto mbsize = pm->pmb_pack->pmb->mb_size.h_view;

  std::vector<Real> profile(5*nx1_global, 0.0);
  Real peak_current = 0.0;
  Real current2 = 0.0;
  Real x2_current2 = 0.0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    const Real dx = mbsize(m).dx1;
    for (int i = is; i <= ie; ++i) {
      const Real x = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                 mbsize(m).x1max);
      const Real jz = (bcc(m, IBY, ks, js, i+1)
                      - bcc(m, IBY, ks, js, i-1))/(2.0*dx);
      int gi = static_cast<int>((x - pm->mesh_size.x1min)/pm->mesh_size.dx1);
      gi = std::max(0, std::min(nx1_global-1, gi));
      const Real jz_model = peak_model*exp(-SQR(x)/(4.0*eta*shifted_time));
      profile[gi] = bcc(m, IBY, ks, js, i);
      profile[nx1_global + gi] = bcc(m, IBZ, ks, js, i);
      profile[2*nx1_global + gi] = jz;
      profile[3*nx1_global + gi] = jz_model;
      profile[4*nx1_global + gi] = x;
      peak_current = std::max(peak_current, jz);
      current2 += SQR(jz)*dx;
      x2_current2 += SQR(x*jz)*dx;
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, profile.data(), 5*nx1_global,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &peak_current, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &current2, 1, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &x2_current2, 1, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  const Real second_moment = x2_current2/current2;
  int failures = pm->ecounter.neos_fail;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (global_variable::my_rank == 0) {
    const std::string basename = pin->GetString("job", "basename");
    std::ofstream errors(basename + "-errs.dat");
    errors << "# Nx1 Ncycle Jz_max Jz_model x2_Jz2 x2_model "
           << "recovery_failures time\n";
    errors << std::setprecision(17) << nx1_global << " " << pm->ncycle << " "
           << peak_current << " " << peak_model << " " << second_moment << " "
           << moment_model << " " << failures << " " << pm->time << "\n";

    std::ofstream output(basename + "-profile.dat");
    output << "# x By Bz Jz Jz_model\n" << std::setprecision(17);
    for (int i = 0; i < nx1_global; ++i) {
      output << profile[4*nx1_global + i] << " " << profile[i] << " "
             << profile[nx1_global + i] << " "
             << profile[2*nx1_global + i] << " "
             << profile[3*nx1_global + i] << "\n";
    }
  }
}

} // namespace
