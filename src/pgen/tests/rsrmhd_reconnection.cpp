//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_reconnection.cpp
//! \brief Pressure-balanced Harris-sheet reconnection with relativistic resistivity.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "mhd/mhd.hpp"
#include "mhd/resistivity_model.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"

namespace {

struct ReconnectionParameters {
  Real field = 1.0;
  Real guide_field = 0.0;
  Real sheet_width = 0.02;
  Real density = 0.1;
  Real sheet_density_contrast = 3.0;
  Real pressure = 1.0e-3;
  Real perturbation = 0.15;
  Real pinch_steepness = 200.0;
  Real pinch_along_offset = 10.0;
  Real pinch_cross_offset = 2.0;
};

ReconnectionParameters reconnection_parameters;

void SRRMHDReconnectionHistory(HistoryData *pdata, Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! \brief Initialize the pressure-balanced Harris sheet used by Grehan et al. (2025)
//! and Ripperda et al. (2026), with x1 along and x2 across the sheet.

void ProblemGenerator::ResistiveSRMHDReconnection(ParameterInput *pin,
                                                    const bool restart) {
  user_hist_func = SRRMHDReconnectionHistory;

  ReconnectionParameters params;
  params.field = pin->GetOrAddReal("problem", "field", 1.0);
  params.guide_field = pin->GetOrAddReal("problem", "guide_field", 0.0);
  params.sheet_width = pin->GetOrAddReal("problem", "sheet_width", 0.02);
  params.density = pin->GetOrAddReal("problem", "density", 0.1);
  params.sheet_density_contrast =
      pin->GetOrAddReal("problem", "sheet_density_contrast", 3.0);
  params.pressure = pin->GetOrAddReal("problem", "pressure", 1.0e-3);
  params.perturbation = pin->GetOrAddReal("problem", "perturbation", 0.15);
  params.pinch_steepness =
      pin->GetOrAddReal("problem", "pinch_steepness", 200.0);
  params.pinch_along_offset =
      pin->GetOrAddReal("problem", "pinch_along_offset", 10.0);
  params.pinch_cross_offset =
      pin->GetOrAddReal("problem", "pinch_cross_offset", 2.0);
  reconnection_parameters = params;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  if (pmhd == nullptr || !(pmhd->is_resistive_rel) || !(pmy_mesh_->two_d)
      || !pmbp->pcoord->is_special_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_reconnection requires two-dimensional "
              << "resistive SRMHD in Cartesian Minkowski spacetime" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (params.field <= 0.0 || params.guide_field < 0.0
      || params.sheet_width <= 0.0 || params.density <= 0.0
      || params.sheet_density_contrast < 0.0 || params.pressure <= 0.0
      || params.perturbation < 0.0 || params.perturbation >= 0.25
      || params.pinch_steepness <= 0.0 || params.pinch_along_offset <= 0.0
      || params.pinch_cross_offset <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_reconnection requires positive field, "
              << "sheet_width, density, pressure, and pinch scales; non-negative "
              << "guide_field and sheet_density_contrast; and 0 <= perturbation < 0.25"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real gamma = pmhd->peos->eos_data.gamma;
  auto &mbsize = pmbp->pmb->mb_size;
  auto w = pmhd->w0;
  auto u = pmhd->u0;
  auto b = pmhd->b0;
  auto bcc = pmhd->bcc0;

  Kokkos::deep_copy(b.x1f, 0.0);
  Kokkos::deep_copy(b.x2f, 0.0);
  Kokkos::deep_copy(b.x3f, params.guide_field);

  const Real field = params.field;
  const Real guide_field = params.guide_field;
  const Real width = params.sheet_width;
  const Real density = params.density;
  const Real density_contrast = params.sheet_density_contrast;
  const Real pressure = params.pressure;
  const Real perturbation = params.perturbation;
  const Real pinch_steepness = params.pinch_steepness;
  const Real pinch_along_offset = params.pinch_along_offset;
  const Real pinch_cross_offset = params.pinch_cross_offset;

  par_for("pgen_srr_reconnection_b1", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
          is, ie+1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real y = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                               mbsize.d_view(m).x2max);
    b.x1f(m, k, j, i) = field*tanh(y/width);
  });

  par_for("pgen_srr_reconnection", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                               mbsize.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                               mbsize.d_view(m).x2max);
    const Real tanh_sheet = tanh(y/width);
    const Real sech2 = 1.0 - SQR(tanh_sheet);
    const Real along = tanh(pinch_steepness*x - pinch_along_offset)
                     + tanh(-pinch_along_offset - pinch_steepness*x);
    const Real across = tanh(pinch_steepness*y + pinch_cross_offset)
                      + tanh(pinch_cross_offset - pinch_steepness*y);
    const Real pinch = 1.0 + perturbation*along*across;
    const Real pgas = pressure + 0.5*SQR(field)*sech2*pinch;

    w(m, IDN, k, j, i) = density*(1.0 + density_contrast*sech2);
    w(m, IVX, k, j, i) = 0.0;
    w(m, IVY, k, j, i) = 0.0;
    w(m, IVZ, k, j, i) = 0.0;
    w(m, IEN, k, j, i) = pgas/(gamma - 1.0);
    w(m, srrmhd::IRE1, k, j, i) = 0.0;
    w(m, srrmhd::IRE2, k, j, i) = 0.0;
    w(m, srrmhd::IRE3, k, j, i) = 0.0;
    bcc(m, IBX, k, j, i) = field*tanh_sheet;
    bcc(m, IBY, k, j, i) = 0.0;
    bcc(m, IBZ, k, j, i) = guide_field;
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
//! \brief Record the kinetic and electric reconnection rates, Alfv\'enic outflow,
//! accepted-state resistivity, density depletion, and eta*J convergence diagnostic.

void SRRMHDReconnectionHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 8;
  pdata->label[0] = "vin";
  pdata->label[1] = "vout";
  pdata->label[2] = "beta_kin";
  pdata->label[3] = "uout_max";
  pdata->label[4] = "Ez_B0";
  pdata->label[5] = "eta_max";
  pdata->label[6] = "rho_min";
  pdata->label[7] = "etaJ_B0";

  auto *pmhd = pm->pmb_pack->pmhd;
  auto w = pmhd->w0;
  auto bcc = pmhd->bcc0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int ncell = pm->pmb_pack->nmb_thispack*nkji;
  const auto eta_data = pmhd->resistivity_data;
  const Real field = reconnection_parameters.field;

  array_sum::GlobalSum sum;
  Real vout_max = 0.0;
  Real uout_max = 0.0;
  Real ez_max = 0.0;
  Real eta_max = 0.0;
  Real eta_j_max = 0.0;
  Real rho_min = 0.0;
  Kokkos::parallel_reduce("srr_reconnection_history",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &result,
                Real &local_vout_max, Real &local_uout_max, Real &local_ez_max,
                Real &local_eta_max, Real &local_eta_j_max, Real &local_rho_min) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k-ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k-ks)*nji - (j-js)*nx1 + is;
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real volume = size.d_view(m).dx1*size.d_view(m).dx2
                      * size.d_view(m).dx3;
    const Real rho = w(m, IDN, k, j, i);
    const Real u1 = w(m, IVX, k, j, i);
    const Real u2 = w(m, IVY, k, j, i);
    const Real u3 = w(m, IVZ, k, j, i);
    const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
    const Real v1 = u1/lor;
    const Real v2 = u2/lor;
    const Real e1 = w(m, srrmhd::IRE1, k, j, i);
    const Real e2 = w(m, srrmhd::IRE2, k, j, i);
    const Real e3 = w(m, srrmhd::IRE3, k, j, i);
    const Real b1 = bcc(m, IBX, k, j, i);
    const Real b2 = bcc(m, IBY, k, j, i);
    const Real b3 = bcc(m, IBZ, k, j, i);
    const Real eta = srrmhd::EvaluateResistivity(
        eta_data, rho, u1, u2, u3, e1, e2, e3, b1, b2, b3);
    const Real jz = (bcc(m, IBY, k, j, i+1)
                   - bcc(m, IBY, k, j, i-1))/(2.0*size.d_view(m).dx1)
                  - (bcc(m, IBX, k, j+1, i)
                   - bcc(m, IBX, k, j-1, i))/(2.0*size.d_view(m).dx2);

    const bool inflow_region = fabs(x) <= 0.5
        && ((y >= -0.8 && y <= -0.3) || (y >= 0.3 && y <= 0.8));
    const Real inward_speed = y < 0.0 ? v2 : -v2;
    if (inflow_region && inward_speed > 0.0) {
      result.the_array[0] += volume*inward_speed;
      result.the_array[1] += volume;
    }
    if (fabs(x) <= 1.5) local_vout_max = fmax(local_vout_max, fabs(v1));
    local_uout_max = fmax(local_uout_max, fabs(u1));
    local_ez_max = fmax(local_ez_max, fabs(e3)/field);
    local_eta_max = fmax(local_eta_max, eta);
    local_eta_j_max = fmax(local_eta_j_max, eta*fabs(jz)/field);
    local_rho_min = fmin(local_rho_min, rho);
  }, Kokkos::Sum<array_sum::GlobalSum>(sum), Kokkos::Max<Real>(vout_max),
     Kokkos::Max<Real>(uout_max), Kokkos::Max<Real>(ez_max),
     Kokkos::Max<Real>(eta_max), Kokkos::Max<Real>(eta_j_max),
     Kokkos::Min<Real>(rho_min));
  Kokkos::fence();

  Real sums[2] = {sum.the_array[0], sum.the_array[1]};
  Real maxima[5] = {vout_max, uout_max, ez_max, eta_max, eta_j_max};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, maxima, 5, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rho_min, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
#endif
  const Real vin = sums[1] > 0.0 ? sums[0]/sums[1] : 0.0;
  const Real beta_kin = maxima[0] > 0.0 ? vin/maxima[0] : 0.0;
  pdata->hdata[0] = vin;
  pdata->hdata[1] = maxima[0];
  pdata->hdata[2] = beta_kin;
  pdata->hdata[3] = maxima[1];
  pdata->hdata[4] = maxima[2];
  pdata->hdata[5] = maxima[3];
  pdata->hdata[6] = rho_min;
  pdata->hdata[7] = maxima[4];

#if MPI_PARALLEL_ENABLED
  // HistoryOutput applies a second sum reduction.  Leave the already-global
  // diagnostics only on rank zero so that this final reduction is idempotent.
  if (global_variable::my_rank != 0) {
    for (int n = 0; n < pdata->nhist; ++n) pdata->hdata[n] = 0.0;
  }
#endif
}

} // namespace
