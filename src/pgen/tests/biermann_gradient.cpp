//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file biermann_gradient.cpp
//! \brief Problem generator for Biermann battery verification with misaligned linear
//! density and pressure gradients.

#include <algorithm>
#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {

// State for optional balancing force in biermann_gradient test.
Real g_biermann_force_x1 = 0.0;
Real g_biermann_force_x2 = 0.0;

//----------------------------------------------------------------------------
// Add body-force source terms that exactly cancel uniform pressure gradients:
//   f = +grad(p), so that momentum equation has (-grad p + f) = 0 initially.
void BiermannGradientSourceTerms(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  bool is_ideal = pmbp->pmhd->peos->eos_data.is_ideal;
  Real fx1 = g_biermann_force_x1;
  Real fx2 = g_biermann_force_x2;

  par_for("biermann_balance_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u0(m,IM1,k,j,i) += bdt*fx1;
    u0(m,IM2,k,j,i) += bdt*fx2;
    if (is_ideal) {
      u0(m,IEN,k,j,i) += bdt*(fx1*w0(m,IVX,k,j,i) + fx2*w0(m,IVY,k,j,i));
    }
  });
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::BiermannGradient()
//! \brief Initializes a static state with rho(x1) and p(x2), zero velocity, and zero B.
//! This creates a non-zero Biermann source term proportional to grad(rho) x grad(p).

void ProblemGenerator::BiermannGradient(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "biermann_gradient requires <mhd>." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->one_d || pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "biermann_gradient requires a 2D mesh (nx2 > 1 and nx3 = 1)." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &eos = pmbp->pmhd->peos->eos_data;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "biermann_gradient requires ideal EOS." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real drho_dx1 = pin->GetOrAddReal("problem", "drho_dx1", 0.5);
  Real p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  Real dp_dx1 = pin->GetOrAddReal("problem", "dp_dx1", 0.0);
  Real dp_dx2 = pin->GetOrAddReal("problem", "dp_dx2", 0.25);

  // Optional angle mode for pressure-gradient direction:
  //   grad(rho) = (drho_mag, 0), grad(p) = dp_mag * (cos(theta), sin(theta))
  // theta is in degrees via theta_deg.
  bool use_theta_mode = pin->GetOrAddBoolean("problem", "use_theta_mode", false);
  if (use_theta_mode) {
    const Real pi = std::acos(-1.0);
    Real theta_deg = pin->GetOrAddReal("problem", "theta_deg", 90.0);
    Real theta_rad = theta_deg*(pi/180.0);

    Real drho_mag = pin->GetOrAddReal("problem", "drho_mag", std::abs(drho_dx1));
    Real dp_mag = pin->GetOrAddReal("problem", "dp_mag",
                                    std::sqrt(SQR(dp_dx1) + SQR(dp_dx2)));

    drho_dx1 = drho_mag;
    dp_dx1 = dp_mag*std::cos(theta_rad);
    dp_dx2 = dp_mag*std::sin(theta_rad);
  }

  // Optional body-force source that balances pressure-gradient forces.
  // f = +grad(p) so that the initial momentum source from pressure is canceled.
  bool balance_pressure_force =
      pin->GetOrAddBoolean("problem", "balance_pressure_force", false);
  if (balance_pressure_force) {
    user_srcs = true;
    user_srcs_func = BiermannGradientSourceTerms;
    g_biermann_force_x1 = dp_dx1;
    g_biermann_force_x2 = dp_dx2;

    if (global_variable::my_rank == 0) {
      Real xmin = pmy_mesh_->mesh_size.x1min;
      Real xmax = pmy_mesh_->mesh_size.x1max;
      Real rho_min = std::min(rho0 + drho_dx1*xmin, rho0 + drho_dx1*xmax);
      Real rho_max = std::max(rho0 + drho_dx1*xmin, rho0 + drho_dx1*xmax);
      Real g1_min = g_biermann_force_x1/rho_max;
      Real g1_max = g_biermann_force_x1/rho_min;
      Real g2_min = g_biermann_force_x2/rho_max;
      Real g2_max = g_biermann_force_x2/rho_min;
      std::cout << "biermann_gradient: balance_pressure_force enabled with "
                << "force_density=(" << g_biermann_force_x1 << ", "
                << g_biermann_force_x2 << "), accel_ranges x1=[" << g1_min
                << ", " << g1_max << "], x2=[" << g2_min << ", " << g2_max
                << "]" << std::endl;
    }
  }

  if (restart) return;

  Real xmin = pmy_mesh_->mesh_size.x1min;
  Real xmax = pmy_mesh_->mesh_size.x1max;
  Real ymin = pmy_mesh_->mesh_size.x2min;
  Real ymax = pmy_mesh_->mesh_size.x2max;

  Real rhomin = std::min(rho0 + drho_dx1*xmin, rho0 + drho_dx1*xmax);
  if (rhomin <= eos.dfloor) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "biermann_gradient density profile violates dfloor." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real pmin = std::min({p0 + dp_dx1*xmin + dp_dx2*ymin,
                        p0 + dp_dx1*xmin + dp_dx2*ymax,
                        p0 + dp_dx1*xmax + dp_dx2*ymin,
                        p0 + dp_dx1*xmax + dp_dx2*ymax});
  if (pmin <= eos.pfloor) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "biermann_gradient pressure profile violates pfloor." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  int &nscal = pmbp->pmhd->nscalars;
  Real gm1 = eos.gamma - 1.0;

  Kokkos::deep_copy(b0.x1f, 0.0);
  Kokkos::deep_copy(b0.x2f, 0.0);
  Kokkos::deep_copy(b0.x3f, 0.0);

  par_for("pgen_biermann_gradient", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);

    Real rho = rho0 + drho_dx1*x1v;
    Real pgas = p0 + dp_dx1*x1v + dp_dx2*x2v;

    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = pgas/gm1;
    for (int n = 0; n < nscal; ++n) {
      w0(m,IYF+n,k,j,i) = 0.0;
    }

    bcc0(m,IBX,k,j,i) = 0.0;
    bcc0(m,IBY,k,j,i) = 0.0;
    bcc0(m,IBZ,k,j,i) = 0.0;
  });

  pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
}
