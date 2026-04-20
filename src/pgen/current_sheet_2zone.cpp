//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file current_sheet_2zone.cpp
//  \brief Problem generator for a single Harris current sheet.
//
// =============================================================================
//  Equilibrium (force balance, polytrope rho = p^(1/gamma))
// =============================================================================
//    By(x1) = b0 * tanh(x1/a0)                    (reconnecting field)
//    Bx     = 0,     Bz = bg                       (guide field)
//    p(x1)  = b0^2 / (2*cosh^2(x1/a0)) + p0        (thermal pressure)
//    rho(x1)= p(x1)^(1/gamma)
//
//  Upstream (|x1| >> a0):
//    By -> +/- b0,  p -> p0,  rho -> rho_up = p0^(1/gamma)
//
// -----------------------------------------------------------------------------
//  Plasma beta (upstream, asymptotic)
// -----------------------------------------------------------------------------
//    beta  ==  p_thermal / p_magnetic  =  p0 / (b0^2 / 2)  =  2*p0 / b0^2
//
//    For beta = 1 at a given p0:  b0 = sqrt(2*p0).
//    Cleanest normalization:  p0 = 1  ->  rho_up = 1,  b0 = sqrt(2)  for beta=1,
//                             then cs_up = sqrt(gamma),  v_A = sqrt(2).
//
// -----------------------------------------------------------------------------
//  Lundquist number (upstream)
// -----------------------------------------------------------------------------
//    S  ==  v_A * L / eta
//
//    with upstream Alfven speed      v_A = b0 / sqrt(rho_up) = b0 / p0^(1/(2*gamma))
//         characteristic length      L   = a0   (sheet half-width)
//         resistivity                eta = ohmic_resistivity
//
//    For target S:  eta = v_A * a0 / S.
//    Example: p0 = 1, b0 = sqrt(2), a0 = 0.05, S = 1e5
//             -> v_A = sqrt(2)  ->  eta = sqrt(2) * 0.05 / 1e5 ≈ 7.07e-7.
//
// -----------------------------------------------------------------------------
//  Perturbation (triggers pinch / reconnection)
// -----------------------------------------------------------------------------
//    delta_p(x1,x2) = epsp * [tanh(200*x1+2) + tanh(2-200*x1)]
//                          * [tanh(200*x2-10) + tanh(-10-200*x2)]
//    p_tot = p_eq * (1 + delta_p)
//
//  The envelope saturates to ~3.86 at the origin, so keep |epsp| < ~0.25 to
//  guarantee p_tot > 0 (avoids the EOS pressure floor clipping and the
//  resulting unphysical IC transient).
//
// -----------------------------------------------------------------------------
//  Boundary conditions (after t_switch)
// -----------------------------------------------------------------------------
//    x1: IC-matching (fill ghost zones with analytical equilibrium).
//    x2: hard-damping diode (damp tangential v and B to equilibrium,
//        diode on normal v, By from div(B) = 0).
//
// =============================================================================

// C/C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()
#include <limits>
// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "globals.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "pgen.hpp"

namespace {

struct CurrentSheet2ZoneBCData {
  Real b0, a0, bg, p0, t_switch;
};

CurrentSheet2ZoneBCData cs2_bc = {0.0, 1.0, 0.0, 1.0, -1.0};
bool cs2_bc_switch_logged = false;

void CurrentSheet2ZoneBoundary(Mesh *pm);

} // namespace


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for single Harris current sheet

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  Real bb0      = pin->GetOrAddReal("problem", "b0", 1.0);
  Real a0       = pin->GetOrAddReal("problem", "a0", 1.0);
  Real bg       = pin->GetOrAddReal("problem", "bg", 0.0);
  Real p0       = pin->GetOrAddReal("problem", "p0", 1.0);
  Real epsp     = pin->GetOrAddReal("problem", "epsp", 0.0);
  Real t_switch = pin->GetOrAddReal("problem", "t_switch", -1.0);

  cs2_bc = {bb0, a0, bg, p0, t_switch};
  user_bcs = true;
  user_bcs_func = CurrentSheet2ZoneBoundary;

  if (restart) return;

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // initialize MHD variables
  if (pmbp->pmhd != nullptr) {
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real inv_gamma = 1.0 / eos.gamma;
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;

    par_for("pgen_mhd", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i - is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j - js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;

      Real p_eq = 0.5*SQR(bb0)/SQR(cosh(x1v/a0)) + p0;
      Real by_eq = bb0*tanh(x1v/a0);

      Real delta_p = epsp * (tanh(200*x2v - 10) + tanh(-10 - 200*x2v))
                           * (tanh(200*x1v + 2)  + tanh(2  - 200*x1v));
      Real p_tot = p_eq * (1.0 + delta_p);

      u0(m,IDN,k,j,i) = pow(p_eq, inv_gamma);
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IPR,k,j,i) = p_tot;
      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) = p_tot/gm1 + 0.5*SQR(by_eq) + 0.5*SQR(bg);
      }

      // face-centered fields
      Real x1f   = LeftEdgeX(i   - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x1fp1 = LeftEdgeX(i+1 - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2f   = LeftEdgeX(j   - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x2fp1 = LeftEdgeX(j+1 - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3f   = LeftEdgeX(k   - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real x3fp1 = LeftEdgeX(k+1 - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);

      b0.x1f(m,k,j,i)   = 0.0;
      b0.x2f(m,k,j,i)   = bb0*tanh(x1v/a0);
      b0.x3f(m,k,j,i)   = bg;

      if (i == ie) { b0.x1f(m,k,j,i+1) = 0.0; }
      if (j == je) { b0.x2f(m,k,j+1,i) = bb0*tanh(x1v/a0); }
      if (k == ke) { b0.x3f(m,k+1,j,i) = bg; }
    });
  }

  return;
}

namespace {

void CurrentSheet2ZoneBoundary(Mesh *pm) {
  if (pm->pmb_pack->pmhd == nullptr) return;
  if (!cs2_bc_switch_logged && cs2_bc.t_switch >= 0.0 && pm->time >= cs2_bc.t_switch) {
    if (global_variable::my_rank == 0) {
      std::cout << "### INFO in current_sheet_2zone.cpp"
                << ": switching BCs (IC-x1/hard-damp-diode-x2) at t="
                << pm->time << " (t_switch=" << cs2_bc.t_switch << ")" << std::endl;
    }
    cs2_bc_switch_logged = true;
  }
  if (cs2_bc.t_switch < 0.0 || pm->time < cs2_bc.t_switch) return;

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie = indcs.ie;
  int &js = indcs.js;  int &je = indcs.je;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &u0 = pm->pmb_pack->pmhd->u0;
  auto &b0 = pm->pmb_pack->pmhd->b0;
  Real gamma = pm->pmb_pack->pmhd->peos->eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real inv_gamma = 1.0/gamma;
  int nmb = pm->pmb_pack->nmb_thispack;

  Real b0_amp = cs2_bc.b0;
  Real a0 = cs2_bc.a0;
  Real bg = cs2_bc.bg;
  Real p0 = cs2_bc.p0;

  // ── x1 boundaries: IC-matching ──────────────────────────────────────────────
  par_for("current_sheet_2zone_bc_x1", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    int nx1 = indcs.nx1;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;

    // inner x1 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) != BoundaryFlag::block) {
      for (int i=0; i<ng; ++i) {
        Real x1v = CellCenterX(-1-i, nx1, x1min, x1max);
        Real p_eq = 0.5*SQR(b0_amp)/SQR(cosh(x1v/a0)) + p0;
        Real d_eq = pow(p_eq, inv_gamma);
        Real by_eq = b0_amp*tanh(x1v/a0);

        u0(m,IDN,k,j,is-i-1) = d_eq;
        u0(m,IM1,k,j,is-i-1) = 0.0;
        u0(m,IM2,k,j,is-i-1) = 0.0;
        u0(m,IM3,k,j,is-i-1) = 0.0;
        u0(m,IEN,k,j,is-i-1) = p_eq/gm1 + 0.5*SQR(by_eq) + 0.5*SQR(bg);

        b0.x1f(m,k,j,is-i-1) = 0.0;
        b0.x2f(m,k,j,is-i-1) = by_eq;
        if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = by_eq;}
        b0.x3f(m,k,j,is-i-1) = bg;
        if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = bg;}
      }
    }

    // outer x1 boundary
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) != BoundaryFlag::block) {
      for (int i=0; i<ng; ++i) {
        Real x1v = CellCenterX(ie-is+1+i, nx1, x1min, x1max);
        Real p_eq = 0.5*SQR(b0_amp)/SQR(cosh(x1v/a0)) + p0;
        Real d_eq = pow(p_eq, inv_gamma);
        Real by_eq = b0_amp*tanh(x1v/a0);

        u0(m,IDN,k,j,ie+i+1) = d_eq;
        u0(m,IM1,k,j,ie+i+1) = 0.0;
        u0(m,IM2,k,j,ie+i+1) = 0.0;
        u0(m,IM3,k,j,ie+i+1) = 0.0;
        u0(m,IEN,k,j,ie+i+1) = p_eq/gm1 + 0.5*SQR(by_eq) + 0.5*SQR(bg);

        b0.x1f(m,k,j,ie+i+2) = 0.0;
        b0.x2f(m,k,j,ie+i+1) = by_eq;
        if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = by_eq;}
        b0.x3f(m,k,j,ie+i+1) = bg;
        if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = bg;}
      }
    }
  });

  // ── x2 boundaries: hard-damping diode ───────────────────────────────────────
  // damp tangential v and B to equilibrium, diode on normal v,
  // reconstruct energy from thermal pressure, By from div(B)=0
  par_for("current_sheet_2zone_bc_x2", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    // inner x2 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) != BoundaryFlag::block) {
      Real rho    = u0(m,IDN,k,js,i);
      Real im1    = u0(m,IM1,k,js,i);
      Real im2    = u0(m,IM2,k,js,i);
      Real im3    = u0(m,IM3,k,js,i);
      Real im2_bc = fmin(0.0, im2);

      Real bx_bcc = 0.5*(b0.x1f(m,k,js,i)   + b0.x1f(m,k,js,i+1));
      Real by_bcc = 0.5*(b0.x2f(m,k,js,i)   + b0.x2f(m,k,js+1,i));
      Real bz_bcc = 0.5*(b0.x3f(m,k,js,i)   + b0.x3f(m,k+1,js,i));
      Real p_th   = gm1*(u0(m,IEN,k,js,i)
                         - 0.5*(im1*im1 + im2*im2 + im3*im3)/rho
                         - 0.5*(bx_bcc*bx_bcc + by_bcc*by_bcc + bz_bcc*bz_bcc));

      Real by_face = b0.x2f(m,k,js,i);

      for (int j=0; j<ng; ++j) {
        u0(m,IDN,k,js-j-1,i) = rho;
        u0(m,IM1,k,js-j-1,i) = 0.0;
        u0(m,IM2,k,js-j-1,i) = im2_bc;
        u0(m,IM3,k,js-j-1,i) = 0.0;
        u0(m,IEN,k,js-j-1,i) = p_th/gm1 + 0.5*im2_bc*im2_bc/rho
                                + 0.5*(by_face*by_face + bg*bg);

        b0.x1f(m,k,js-j-1,i) = 0.0;
        if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = 0.0;}
        b0.x2f(m,k,js-j-1,i) = by_face;
        b0.x3f(m,k,js-j-1,i) = bg;
        if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = bg;}
      }
    }

    // outer x2 boundary
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) != BoundaryFlag::block) {
      Real rho    = u0(m,IDN,k,je,i);
      Real im1    = u0(m,IM1,k,je,i);
      Real im2    = u0(m,IM2,k,je,i);
      Real im3    = u0(m,IM3,k,je,i);
      Real im2_bc = fmax(0.0, im2);

      Real bx_bcc = 0.5*(b0.x1f(m,k,je,i)   + b0.x1f(m,k,je,i+1));
      Real by_bcc = 0.5*(b0.x2f(m,k,je,i)   + b0.x2f(m,k,je+1,i));
      Real bz_bcc = 0.5*(b0.x3f(m,k,je,i)   + b0.x3f(m,k+1,je,i));
      Real p_th   = gm1*(u0(m,IEN,k,je,i)
                         - 0.5*(im1*im1 + im2*im2 + im3*im3)/rho
                         - 0.5*(bx_bcc*bx_bcc + by_bcc*by_bcc + bz_bcc*bz_bcc));

      Real by_face = b0.x2f(m,k,je+1,i);

      for (int j=0; j<ng; ++j) {
        u0(m,IDN,k,je+j+1,i) = rho;
        u0(m,IM1,k,je+j+1,i) = 0.0;
        u0(m,IM2,k,je+j+1,i) = im2_bc;
        u0(m,IM3,k,je+j+1,i) = 0.0;
        u0(m,IEN,k,je+j+1,i) = p_th/gm1 + 0.5*im2_bc*im2_bc/rho
                                + 0.5*(by_face*by_face + bg*bg);

        b0.x1f(m,k,je+j+1,i) = 0.0;
        if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = 0.0;}
        b0.x2f(m,k,je+j+2,i) = by_face;
        b0.x3f(m,k,je+j+1,i) = bg;
        if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = bg;}
      }
    }
  });
}

} // namespace
