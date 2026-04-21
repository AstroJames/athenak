//========================================================================================
// AthenaK astrophysical MHD code
//========================================================================================
//! \file coalescing_ft.cpp
//! \brief 2D coalescing flux-tube / plasmoid reconnection problem generator
//!
//! Follows the setup of Vicentin et al. (2025, ApJ), which builds on the coalescing
//! flux-tube configuration of Bhattacharjee et al. (2009) / Huang & Bhattacharjee (2010).
//!
//! Geometry (AthenaK coordinates):
//!   x1 : along the current sheet   (paper's x, reconnection outflow direction)
//!   x2 : across the current sheet  (paper's y, inflow / sheet-normal direction)
//!   x3 : guide-field direction      (paper's z)
//!
//! Initial magnetic field — Fadeev equilibrium:
//!   B = (ẑ × ∇ψ) + bg ẑ,   ψ = a0 * ln(cosh(x2/a0) + eps * cos(2π x1 / L1))
//!
//!   B_x1 = B0 * sinh(x2/a0) / (cosh(x2/a0) + eps * cos(2π x1/L1))
//!   B_x2 = B0 * a0*(2π/L1)*eps * sin(2π x1/L1) / (cosh(x2/a0) + eps*cos(2π x1/L1))
//!   B_x3 = bg   (uniform guide field)
//!
//!   eps = 0 recovers the pure Harris sheet.  Require eps < 1 to keep denom > 0.
//!
//! Density from isothermal total-pressure balance:
//!   p_tot = (1 + beta) * (B0² + bg²)/2
//!   rho   = (p_tot - B²/2) / cs²
//!
//! Velocity: zero mean + small-amplitude random noise (epsv)
//!
//! Recommended boundary conditions (input file):
//!   x1 : periodic   (current sheet repeats; plasmoids advect in x1)
//!   x2 : reflect     (perfectly-conducting, free-slip walls)
//!
//! Diffusion must be set in the <mhd> block of the input file:
//!   ohmic_resistivity = eta   (Lundquist number S = B0 * L / eta)
//!   viscosity         = nu    (Prandtl number Pm = nu / eta; paper uses Pm = 1)

// C++ headers
#include <cmath>
#include <iostream>
#include <sstream>

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  if (pmy_mesh_->pmb_pack->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in tearing_mode_2d: requires MHD physics" << std::endl;
    exit(EXIT_FAILURE);
  }

  // ---- problem parameters ----
  Real bb0  = pin->GetOrAddReal("problem", "b0",   1.0);    // reconnecting field amplitude
  Real bg   = pin->GetOrAddReal("problem", "bg",   0.5);    // guide field (x3)
  Real a0   = pin->GetOrAddReal("problem", "a0",   0.1);    // sheet half-thickness δ = S^{-1/2}
  Real beta = pin->GetOrAddReal("problem", "beta", 2.0);    // plasma β = 2ρ₀cs²/(B0²+bg²)
  Real eps  = pin->GetOrAddReal("problem", "eps",  0.2);    // Fadeev ε  (0 = pure Harris sheet)
  Real epsv = pin->GetOrAddReal("problem", "epsv", 1.0e-2); // velocity noise amplitude (in VA)
  int  seed = pin->GetOrAddInteger("problem", "rng_seed", 12345); // RNG seed

  if (eps >= 1.0) {
    std::cout << "### FATAL ERROR in tearing_mode_2d: eps must be < 1 (got "
              << eps << ")" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Domain length in x1 (needed for Fadeev periodicity in x1)
  Real L1 = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;

  // Isothermal sound speed from EOS input
  EOS_Data &eos = pmy_mesh_->pmb_pack->pmhd->peos->eos_data;
  Real cs2 = eos.iso_cs * eos.iso_cs;

  // Total pressure constant:  p_tot = (1 + β) * p_mag_max
  //   p_mag_max = (B0² + bg²)/2  (magnetic pressure far from current sheet)
  Real p_mag_max = 0.5 * (bb0*bb0 + bg*bg);
  Real p_tot     = (1.0 + beta) * p_mag_max;

  // ---- capture for kernels ----
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto &u0   = pmbp->pmhd->u0;
  auto &b0   = pmbp->pmhd->b0;

  uint64_t rng_seed = static_cast<uint64_t>(seed) + static_cast<uint64_t>(pmbp->gids);
  Kokkos::Random_XorShift64_Pool<> rand_pool(rng_seed);

  // ---- initialise MHD variables ----
  par_for("pgen_tearing2d", DevExeSpace(),
          0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Cell-centre coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v   = CellCenterX(i-is,   nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v   = CellCenterX(j-js,   nx2, x2min, x2max);

    // Face-edge coordinates
    Real x1f    = LeftEdgeX(i-is,   nx1, x1min, x1max);  // left  face in x1
    Real x1fp1  = LeftEdgeX(i-is+1, nx1, x1min, x1max);  // right face in x1
    Real x2f    = LeftEdgeX(j-js,   nx2, x2min, x2max);  // bottom face in x2
    Real x2fp1  = LeftEdgeX(j-js+1, nx2, x2min, x2max);  // top    face in x2

    // Precomputed constants
    Real twopi_L1 = 2.0*M_PI / L1;

    // Helper: Fadeev denominator at (x1_arg, x2_arg)
    // denom(x1,x2) = cosh(x2/a0) + eps * cos(2π x1/L1)
    // B_x1(x1,x2) = B0 * sinh(x2/a0) / denom
    // B_x2(x1,x2) = B0 * a0 * twopi_L1 * eps * sin(2π x1/L1) / denom

    // Face-centred B_x1 at left x1-face  (x1f, x2v)
    Real denom_1f = cosh(x2v/a0) + eps*cos(twopi_L1*x1f);
    Real Bx1_face = bb0 * sinh(x2v/a0) / denom_1f;

    // Face-centred B_x2 at bottom x2-face  (x1v, x2f)
    Real denom_2f = cosh(x2f/a0) + eps*cos(twopi_L1*x1v);
    Real Bx2_face = bb0 * a0 * twopi_L1 * eps * sin(twopi_L1*x1v) / denom_2f;

    // Cell-centred B for pressure balance
    Real denom_cc = cosh(x2v/a0) + eps*cos(twopi_L1*x1v);
    Real Bx1c = bb0 * sinh(x2v/a0) / denom_cc;
    Real Bx2c = bb0 * a0 * twopi_L1 * eps * sin(twopi_L1*x1v) / denom_cc;
    Real B2   = Bx1c*Bx1c + Bx2c*Bx2c + bg*bg;

    // Density from isothermal total-pressure balance; floor to prevent negatives
    Real rho = fmax((p_tot - 0.5*B2) / cs2, 1.0e-4);

    // Random velocity perturbation
    auto rgen = rand_pool.get_state();
    Real vx1 = epsv * (2.0*rgen.drand() - 1.0);
    Real vx2 = epsv * (2.0*rgen.drand() - 1.0);
    rand_pool.free_state(rgen);

    // Conserved variables  (isothermal: no IEN)
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho * vx1;
    u0(m,IM2,k,j,i) = rho * vx2;
    u0(m,IM3,k,j,i) = 0.0;

    // Face-centred magnetic field
    b0.x1f(m,k,j,i) = Bx1_face;
    b0.x2f(m,k,j,i) = Bx2_face;
    b0.x3f(m,k,j,i) = bg;

    // Upper-boundary faces (one extra per block edge)
    if (i == ie) {
      Real denom = cosh(x2v/a0) + eps*cos(twopi_L1*x1fp1);
      b0.x1f(m,k,j,i+1) = bb0 * sinh(x2v/a0) / denom;
    }
    if (j == je) {
      Real denom = cosh(x2fp1/a0) + eps*cos(twopi_L1*x1v);
      b0.x2f(m,k,j+1,i) = bb0 * a0 * twopi_L1 * eps * sin(twopi_L1*x1v) / denom;
    }
    if (k == ke) {
      b0.x3f(m,k+1,j,i) = bg;
    }
  });

  return;
}
