//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rt.cpp
//! \brief Problem generator for RT instabilty.
//!
//! Note the gravitational acceleration is hardwired to be 0.1. Density difference is
//! hardwired to be 2.0 in 2D, and is set by the input parameter `problem/rhoh` in 3D
//! (default value is 3.0). This reproduces 2D results of Liska & Wendroff, 3D results of
//! Dimonte et al.
//!
//! FOR 2D HYDRO:
//! Problem domain should be -1/6 < x < 1/6; -0.5 < y < 0.5 with gamma=1.4 to match Liska
//! & Wendroff. Interface is at y=0; perturbation added to Vy. Gravity acts in y-dirn.
//! Special reflecting boundary conditions added in x2 to improve hydrostatic eqm
//! (prevents launching of weak waves) Atwood number A=(d2-d1)/(d2+d1)=1/3. Options:
//!    - iprob = 1  -- Perturb V2 using single mode
//!    - iprob != 1 -- Perturb V2 using multiple mode
//!
//! FOR 3D:
//! Problem domain should be -.05 < x < .05; -.05 < y < .05, -.1 < z < .1, gamma=5/3 to
//! match Dimonte et al.  Interface is at z=0; perturbation added to Vz. Gravity acts in
//! z-dirn. Special reflecting boundary conditions added in x3.  A=1/2.  Options:
//!    - iprob = 1 -- Perturb V3 using single mode
//!    - iprob = 2 -- Perturb V3 using multiple mode
//!    - iprob = 3 -- B rotated by "angle" at interface, multimode perturbation
//!
//! REFERENCE: R. Liska & B. Wendroff, SIAM J. Sci. Comput., 25, 995 (2003)

// C++ headers
#include <cmath>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/grid_locations.hpp"
#include "utils/random.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Problem Generator for the Rayleigh-Taylor instability test

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  int64_t iseed = -1;

  Real kx = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);
  Real ky = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min);
  Real kz = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min);

  // Read perturbation amplitude, problem switch, density ratio
  Real amp = pin->GetReal("problem","amp");
  int iprob = pin->GetInteger("problem","iprob");
  Real drat = pin->GetOrAddReal("problem","drat",3.0);

  // capture variables for kernel
  int &nx1 = pmbp->mb_cells.nx1;
  int &nx2 = pmbp->mb_cells.nx2;
  int &nx3 = pmbp->mb_cells.nx3;
  int &is = pmbp->mb_cells.is, &ie = pmbp->mb_cells.ie;
  int &js = pmbp->mb_cells.js, &je = pmbp->mb_cells.je;
  int &ks = pmbp->mb_cells.ks, &ke = pmbp->mb_cells.ke;
  auto &size = pmbp->pmb->mbsize;

  // initialize Hydro variables ----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
   
    // 2D PROBLEM

    if (not pmbp->pmesh->nx3gt1) {
      Real grav_acc = pin->GetReal("gravity","const_acc2");

      auto u0 = pmbp->phydro->u0;
      par_for("rt2d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i)
        {
          Real den=1.0;
          Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
          Real x2v = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));
          if (x2v > 0.0) den *= drat;

          if (iprob == 1) {
            u0(m,IM2,k,j,i) = (1.0 + cos(kx*x1v))*(1.0 + cos(ky*x2v))/4.0;
          } else {
            u0(m,IM2,k,j,i) = (Ran2((int64_t*)iseed) - 0.5)*(1.0 + cos(ky*x2v));
          }

          u0(m,IDN,k,j,i) = den;
          u0(m,IM1,k,j,i) = 0.0;
          u0(m,IM2,k,j,i) *= (den*amp);
          u0(m,IM3,k,j,i) = 0.0;
          u0(m,IEN,k,j,i) = (p0 + grav_acc*den*x2v)/gm1 + 0.5*SQR(u0(m,IM2,k,j,i))/den;
        }
      );

    // 3D PROBLEM ----------------------------------------------------------------

    } else {
      Real grav_acc = pin->GetReal("gravity","const_acc3");

      auto u0 = pmbp->phydro->u0;
      par_for("rt2d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i)
        {
          Real den=1.0;
          Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
          Real x2v = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));
          Real x3v = CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m));

          if (x3v > 0.0) den *= drat;

          if (iprob == 1) {
            u0(m,IM3,k,j,i) = (1.0+cos(kx*x1v))*(1.0+cos(ky*x2v))*(1.0+cos(kz*x3v))/8.0;
          } else {
            u0(m,IM3,k,j,i) = amp*(Ran2((int64_t*)iseed) - 0.5)*(1.0 + cos(kz*x3v));
          }

          u0(m,IDN,k,j,i) = den;
          u0(m,IM1,k,j,i) = 0.0;
          u0(m,IM2,k,j,i) = 0.0;
          u0(m,IM3,k,j,i) *= (den*amp);
          u0(m,IEN,k,j,i) = (p0 + grav_acc*den*x3v)/gm1 + 0.5*SQR(u0(m,IM3,k,j,i))/den;
        }
      );
    }

  } // end of Hydro initialization

  return;
}