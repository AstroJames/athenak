//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file current_sheet_2zone.cpp
//  \ Problem generator for single Harris current sheet
//  Sets up different initial conditions selected by flag "iprob"
//    - iprob=1 : E_int = p_0 * rho / (gamma - 1) original single sheet
//    - iprob=2 : E_int = p/(gamma - 1)
//    - iprob=3 : rho = P^(1/gamma) force balance with density

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
  int iprob;
  Real b0, a0, bg, p0, t_switch;
};

CurrentSheet2ZoneBCData cs2_bc = {0, 0.0, 1.0, 0.0, 1.0, -1.0};
bool cs2_bc_switch_logged = false;

void CurrentSheet2ZoneBoundary(Mesh *pm);

} // namespace


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::CurrentSheet2Zone()
//  \Sets initial conditions for single Harris current sheet

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // read global parameters
  int iprob  = pin->GetInteger("problem", "iprob");
  Real d0 = pin->GetOrAddReal("problem", "d0", 1.0);
  Real ng = pin->GetOrAddReal("problem", "ng", 1.0);
  Real bb0 = pin->GetOrAddReal("problem", "b0", 1.0);
  Real a0 = pin->GetOrAddReal("problem", "a0", 1.0);
  Real bg = pin->GetOrAddReal("problem", "bg", 0.);
  Real kval = pin->GetOrAddReal("problem", "kval", 1.0);
  Real p0 = pin->GetOrAddReal("problem", "p0", 1.0); // defaulted to some value
  Real epsp = pin->GetOrAddReal("problem", "epsp", 0.0);
  Real t_switch = pin->GetOrAddReal("problem", "t_switch", -1.0);

  // For iprob==3, use a user BC callback that switches from periodic behavior to
  // equilibrium IC matching at t_switch.
  cs2_bc = {iprob, bb0, a0, bg, p0, t_switch};
  if (iprob == 3) {
    user_bcs_func = CurrentSheet2ZoneBoundary;
  }

  if (restart) return;

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;



// initialize MHD variables ------------------------------------------------------------
if (pmbp->pmhd != nullptr) {
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
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
    Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);

    // compute cell-centered conserved variables
    u0(m, IDN, k, j, i) = (d0 / (pow(cosh(x1v / a0), 2.0)) + ng);
    if (iprob == 3){
      u0(m, IDN, k, j, i) = pow((pow(bb0, 2.0) / 2.0 / pow(cosh(x1v / a0), 2.0) + p0), 1/eos.gamma) ; // force balance condition
    }
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;

    Real delta_p = 0.0;
    if (iprob == 1) {
      if (eos.is_ideal) {
        u0(m, IEN, k, j, i) = p0 * u0(m, IDN, k, j, i) / gm1;
      }
    } else if (iprob == 2 || iprob == 3) {
      delta_p = epsp * ( tanh(200*x2v - 10) + tanh(-10-200*x2v)) * (tanh(200*x1v + 2) + tanh(2-200*x1v));
      u0(m, IPR, k, j, i) = (pow(bb0, 2.0) / 2.0 / pow(cosh(x1v / a0), 2.0) + p0) * (1+delta_p);
      if (eos.is_ideal) {
        u0(m, IEN, k, j, i) = (u0(m, IPR, k, j, i) / gm1  + 1/2.0 * pow(bb0 * tanh(x1v / a0), 2.0));    // u0(m, IPR, k, j, i) / gm1; changed so that IEN is the total energy 
      }
    }


    // Compute face-centered fields
    Real x1f = LeftEdgeX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x1fp1 = LeftEdgeX(i + 1 - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2f = LeftEdgeX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x2fp1 = LeftEdgeX(j + 1 - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3f = LeftEdgeX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real x3fp1 = LeftEdgeX(k + 1 - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    // Initialize magnetic fields
    b0.x1f(m, k, j, i) = 0.0;

    if (iprob == 1) {
      b0.x2f(m, k, j, i) = bb0 * tanh(x1v / a0) - bb0;
    } else if (iprob == 2) {
      b0.x2f(m, k, j, i) = bb0 * tanh(x1v / a0);
    } else if (iprob == 3) {
      b0.x2f(m, k, j, i) = bb0 * tanh(x1v / a0);
    }

    b0.x3f(m, k, j, i) = bg;

    // Boundary faces
    if (i == ie) {
      b0.x1f(m, k, j, i + 1) = 0.0;
    }

    if (j == je) {
      if (iprob == 1) {
        b0.x2f(m, k, j + 1, i) = bb0 * tanh(x1v / a0) - bb0;
      } else if (iprob == 2) {
        b0.x2f(m, k, j + 1, i) = bb0 * tanh(x1v / a0);
      } else if (iprob == 3) {
        b0.x2f(m, k, j + 1, i) = bb0 * tanh(x1v / a0);
      }
    }

    if (k == ke) {
      b0.x3f(m, k + 1, j, i) = bg;
    }
  });  // <-- End of KOKKOS_LAMBDA
}  // End initialization MHD variables

return;
}

namespace {

void CurrentSheet2ZoneBoundary(Mesh *pm) {
  if (cs2_bc.iprob != 3) return;
  if (!cs2_bc_switch_logged && cs2_bc.t_switch >= 0.0 && pm->time >= cs2_bc.t_switch) {
    if (global_variable::my_rank == 0) {
      std::cout << "### INFO in " << __FILE__
                << ": switching current_sheet_2zone boundaries to IC-matching at t="
                << pm->time << " (t_switch=" << cs2_bc.t_switch << ")" << std::endl;
    }
    cs2_bc_switch_logged = true;
  }
  if (cs2_bc.t_switch < 0.0 || pm->time < cs2_bc.t_switch) return;
  if (pm->pmb_pack->pmhd == nullptr) return;

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
        u0(m,IEN,k,j,is-i-1) = p_eq/gm1 + 0.5*SQR(by_eq);

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
        u0(m,IEN,k,j,ie+i+1) = p_eq/gm1 + 0.5*SQR(by_eq);

        b0.x1f(m,k,j,ie+i+2) = 0.0;
        b0.x2f(m,k,j,ie+i+1) = by_eq;
        if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = by_eq;}
        b0.x3f(m,k,j,ie+i+1) = bg;
        if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = bg;}
      }
    }
  });

  // x2 (y) boundaries: enforce the same unperturbed equilibrium profile
  par_for("current_sheet_2zone_bc_x2", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    int nx1 = indcs.nx1;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real p_eq = 0.5*SQR(b0_amp)/SQR(cosh(x1v/a0)) + p0;
    Real d_eq = pow(p_eq, inv_gamma);
    Real by_eq = b0_amp*tanh(x1v/a0);

    // inner x2 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) != BoundaryFlag::block) {
      for (int j=0; j<ng; ++j) {
        u0(m,IDN,k,js-j-1,i) = d_eq;
        u0(m,IM1,k,js-j-1,i) = 0.0;
        u0(m,IM2,k,js-j-1,i) = 0.0;
        u0(m,IM3,k,js-j-1,i) = 0.0;
        u0(m,IEN,k,js-j-1,i) = p_eq/gm1 + 0.5*SQR(by_eq);

        b0.x1f(m,k,js-j-1,i) = 0.0;
        if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = 0.0;}
        b0.x2f(m,k,js-j-1,i) = by_eq;
        b0.x3f(m,k,js-j-1,i) = bg;
        if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = bg;}
      }
    }

    // outer x2 boundary
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) != BoundaryFlag::block) {
      for (int j=0; j<ng; ++j) {
        u0(m,IDN,k,je+j+1,i) = d_eq;
        u0(m,IM1,k,je+j+1,i) = 0.0;
        u0(m,IM2,k,je+j+1,i) = 0.0;
        u0(m,IM3,k,je+j+1,i) = 0.0;
        u0(m,IEN,k,je+j+1,i) = p_eq/gm1 + 0.5*SQR(by_eq);

        b0.x1f(m,k,je+j+1,i) = 0.0;
        if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = 0.0;}
        b0.x2f(m,k,je+j+2,i) = by_eq;
        b0.x3f(m,k,je+j+1,i) = bg;
        if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = bg;}
      }
    }
  });
}

} // namespace
