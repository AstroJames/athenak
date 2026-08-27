//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_maxwell_ohm.cpp
//! \brief Linear one-dimensional Maxwell--Ohm slow-mode regression.

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
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {

void SRRMHDMaxwellOhmErrors(ParameterInput *pin, Mesh *pm);

Real SlowEigenvalue(const Real eta, const Real wavenumber) {
  const Real discriminant = 1.0 - 4.0*SQR(eta*wavenumber);
  if (!(discriminant > 0.0)) return 0.0;
  return -2.0*eta*SQR(wavenumber)/(1.0 + sqrt(discriminant));
}

} // namespace

//----------------------------------------------------------------------------------------
//! \brief Initialize a well-prepared slow eigenmode of Maxwell plus scalar Ohm.

void ProblemGenerator::ResistiveSRMHDMaxwellOhm(ParameterInput *pin,
                                                 const bool restart) {
  pgen_final_func = SRRMHDMaxwellOhmErrors;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  if (pmhd == nullptr || !(pmhd->is_resistive_rel) || !(pmy_mesh_->one_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_maxwell_ohm requires one-dimensional "
              << "resistive SRMHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmhd->use_electric_ct) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_maxwell_ohm currently diagnoses the "
              << "cell-centred electric-field path only" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmhd->resistivity_data.model != srrmhd::ResistivityModel::uniform
      || !(pmhd->resistivity > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_maxwell_ohm requires positive uniform "
              << "resistivity" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real amplitude = pin->GetOrAddReal("problem", "amplitude", 1.0e-4);
  const Real density = pin->GetOrAddReal("problem", "density", 1.0);
  const Real pressure = pin->GetOrAddReal("problem", "pressure", 1.0);
  const int mode = pin->GetOrAddInteger("problem", "mode_number", 1);
  const Real length = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  const Real wavenumber = 2.0*M_PI*mode/length;
  const Real lambda = SlowEigenvalue(pmhd->resistivity, wavenumber);
  if (!(amplitude > 0.0) || !(density > 0.0) || !(pressure > 0.0)
      || mode <= 0 || !(lambda < 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_maxwell_ohm requires positive amplitude, "
              << "density, pressure, and mode, with 2*eta*k < 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real gamma = pmhd->peos->eos_data.gamma;
  const Real xmin = pmy_mesh_->mesh_size.x1min;
  auto &mbsize = pmbp->pmb->mb_size;
  auto w = pmhd->w0;
  auto u = pmhd->u0;
  auto b = pmhd->b0;
  auto bcc = pmhd->bcc0;

  Kokkos::deep_copy(b.x1f, 0.0);
  Kokkos::deep_copy(b.x2f, 0.0);
  Kokkos::deep_copy(b.x3f, 0.0);
  par_for("pgen_srr_maxwell_ohm_by", DevExeSpace(), 0, nmb-1, ks, ke,
          js, je+1, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    b.x2f(m, k, j, i) = amplitude*cos(wavenumber*(x - xmin));
  });

  par_for("pgen_srr_maxwell_ohm", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    const Real phase = wavenumber*(x - xmin);
    const Real by = amplitude*cos(phase);
    const Real ez = (lambda/wavenumber)*amplitude*sin(phase);
    w(m, IDN, k, j, i) = density;
    w(m, IVX, k, j, i) = 0.0;
    w(m, IVY, k, j, i) = 0.0;
    w(m, IVZ, k, j, i) = 0.0;
    w(m, IEN, k, j, i) = pressure/(gamma - 1.0);
    w(m, srrmhd::IRE1, k, j, i) = 0.0;
    w(m, srrmhd::IRE2, k, j, i) = 0.0;
    w(m, srrmhd::IRE3, k, j, i) = ez;
    bcc(m, IBX, k, j, i) = 0.0;
    bcc(m, IBY, k, j, i) = by;
    bcc(m, IBZ, k, j, i) = 0.0;
  });
  pmhd->peos->PrimToCons(w, bcc, u, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! \brief Project the accepted fields onto the analytic Fourier eigenmode.

void SRRMHDMaxwellOhmErrors(ParameterInput *pin, Mesh *pm) {
  auto *pmhd = pm->pmb_pack->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, ks = indcs.ks;
  const int nx1_global = pm->mesh_indcs.nx1;
  const Real amplitude = pin->GetOrAddReal("problem", "amplitude", 1.0e-4);
  const Real density = pin->GetOrAddReal("problem", "density", 1.0);
  const Real pressure = pin->GetOrAddReal("problem", "pressure", 1.0);
  const int mode = pin->GetOrAddInteger("problem", "mode_number", 1);
  const Real xmin = pm->mesh_size.x1min;
  const Real length = pm->mesh_size.x1max - xmin;
  const Real wavenumber = 2.0*M_PI*mode/length;
  const Real eta = pmhd->resistivity;
  const Real lambda = SlowEigenvalue(eta, wavenumber);
  const Real gamma = pmhd->peos->eos_data.gamma;
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto bcc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bcc0);
  auto mbsize = pm->pmb_pack->pmb->mb_size.h_view;

  Real b_amplitude = 0.0;
  Real e_amplitude = 0.0;
  Real velocity_max = 0.0;
  Real density_error = 0.0;
  Real pressure_error = 0.0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    const Real dx = mbsize(m).dx1;
    const Real projection_weight = 2.0*dx/length;
    for (int i = is; i <= ie; ++i) {
      const Real x = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                 mbsize(m).x1max);
      const Real phase = wavenumber*(x - xmin);
      b_amplitude += projection_weight*bcc(m, IBY, ks, js, i)*cos(phase);
      e_amplitude += projection_weight*w(m, srrmhd::IRE3, ks, js, i)*sin(phase);
      velocity_max = std::max(velocity_max, std::abs(w(m, IVX, ks, js, i)));
      velocity_max = std::max(velocity_max, std::abs(w(m, IVY, ks, js, i)));
      velocity_max = std::max(velocity_max, std::abs(w(m, IVZ, ks, js, i)));
      density_error = std::max(density_error,
                               std::abs(w(m, IDN, ks, js, i) - density));
      const Real local_pressure = (gamma - 1.0)*w(m, IEN, ks, js, i);
      pressure_error = std::max(pressure_error,
                                std::abs(local_pressure - pressure));
    }
  }
#if MPI_PARALLEL_ENABLED
  Real projections[2] = {b_amplitude, e_amplitude};
  Real maxima[3] = {velocity_max, density_error, pressure_error};
  MPI_Allreduce(MPI_IN_PLACE, projections, 2, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, maxima, 3, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  b_amplitude = projections[0];
  e_amplitude = projections[1];
  velocity_max = maxima[0];
  density_error = maxima[1];
  pressure_error = maxima[2];
#endif
  int failures = pm->ecounter.neos_fail;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  const Real b_exact = amplitude*exp(lambda*pm->time);
  const Real e_exact = (lambda/wavenumber)*b_exact;
  const Real h = pm->time/eta;
  const Real b_ratio = b_amplitude/b_exact;
  const Real e_ratio = e_amplitude/e_exact;
  const Real ohm_ratio = e_amplitude/(-eta*wavenumber*b_amplitude);
  if (global_variable::my_rank == 0) {
    const std::string basename = pin->GetString("job", "basename");
    std::ofstream errors(basename + "-errs.dat");
    errors << "# Nx1 Ncycle time eta h k lambda B B_exact E E_exact "
           << "B_ratio E_ratio E_over_etaJ vmax drho dp failures\n";
    errors << std::setprecision(17) << nx1_global << " " << pm->ncycle << " "
           << pm->time << " " << eta << " " << h << " " << wavenumber << " "
           << lambda << " " << b_amplitude << " " << b_exact << " "
           << e_amplitude << " " << e_exact << " " << b_ratio << " "
           << e_ratio << " " << ohm_ratio << " " << velocity_max << " "
           << density_error << " " << pressure_error << " " << failures << "\n";
  }
}

} // namespace
