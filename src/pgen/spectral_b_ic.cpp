//========================================================================================
// AthenaK: astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectral_b_ic.cpp
//! \brief Problem generator that initialises the magnetic field with a turbulent power
//! spectrum specified in the <spectral_ic> block of the input file, while setting a
//! uniform background fluid state from the <problem> block.
//!
//! The vector potential A is generated via direct Fourier synthesis (SpectralICGenerator),
//! then curled to give face-centred B = curl(A), guaranteeing div(B) = 0 discretely.
//! After the curl, the global mean is subtracted and the field is normalised to rms_b.
//! An optional uniform background field (b_mean_{x,y,z}) is added last.
//!
//! Input parameters:
//!   <problem>
//!     pgen_name   = spectral_b_ic
//!     rho0        = 1.0       # background density
//!     pres0       = 0.6       # background pressure
//!     vx0,vy0,vz0 = 0.0      # background velocities
//!
//!   <spectral_ic>
//!     nlow         = 2        # minimum mode index (k_phys = n * 2π/L)
//!     nhigh        = 4        # maximum mode index
//!     spectrum     = power_law   # band | parabolic | power_law
//!     spectral_index = 1.67   # E_B(k) ∝ k^{-spectral_index} (power_law only)
//!     rms_b        = 1.0      # target RMS |B| after normalisation
//!     b_mean_x     = 0.0      # optional uniform background B_x
//!     b_mean_y     = 0.0      # optional uniform background B_y
//!     b_mean_z     = 0.0      # optional uniform background B_z
//!     iseed        = -1234    # RNG seed (negative → initialise)

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "pgen.hpp"
#include "utils/spectral_ic_gen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::SpectralBIC
//! \brief Spectral magnetic-field initial conditions.

void ProblemGenerator::SpectralBIC(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "spectral_b_ic requires MHD physics." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Background state
  Real rho0  = pin->GetOrAddReal("problem", "rho0",  1.0);
  Real pres0 = pin->GetOrAddReal("problem", "pres0", 0.6);
  Real vx0   = pin->GetOrAddReal("problem", "vx0",   0.0);
  Real vy0   = pin->GetOrAddReal("problem", "vy0",   0.0);
  Real vz0   = pin->GetOrAddReal("problem", "vz0",   0.0);

  // Spectral IC parameters
  Real rms_b   = pin->GetOrAddReal("spectral_ic", "rms_b",    1.0);
  Real bmean_x = pin->GetOrAddReal("spectral_ic", "b_mean_x", 0.0);
  Real bmean_y = pin->GetOrAddReal("spectral_ic", "b_mean_y", 0.0);
  Real bmean_z = pin->GetOrAddReal("spectral_ic", "b_mean_z", 0.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  bool is_relativistic = pmbp->pcoord->is_special_relativistic ||
                         pmbp->pcoord->is_general_relativistic ||
                         pmbp->pcoord->is_dynamical_relativistic;

  // -------------------------------------------------------------------------
  // Step 1: Generate spectral vector potential
  // -------------------------------------------------------------------------
  // Allocate temporary node-point arrays.  The size ncells{1,2,3} is sufficient
  // because ie+1 < ncells1, je+1 < ncells2, ke+1 < ncells3 (with ng >= 2).
  DvceArray4D<Real> ax("ax_spec", nmb, ncells3, ncells2, ncells1);
  DvceArray4D<Real> ay("ay_spec", nmb, ncells3, ncells2, ncells1);
  DvceArray4D<Real> az("az_spec", nmb, ncells3, ncells2, ncells1);

  // Zero-initialise (Kokkos does not guarantee this)
  par_for("spec_ic_zero_A", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2-1,
          0, ncells1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    ax(m,k,j,i) = 0.0;
    ay(m,k,j,i) = 0.0;
    az(m,k,j,i) = 0.0;
  });

  // Build generator and fill A via FFT-based path
  SpectralICGenerator gen(pmbp, pin);
  std::string backend = gen.GenerateVectorPotentialFFT(ax, ay, az);
  if (global_variable::my_rank == 0) {
    std::cout << "spectral_b_ic: vector potential generated using backend='"
              << backend << "'" << std::endl;
  }

  // -------------------------------------------------------------------------
  // Step 2: Curl A → face-centred b0
  // -------------------------------------------------------------------------
  auto &b0   = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;

  par_for("spec_ic_curl", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = (az(m,k,j+1,i) - az(m,k,j,i))/dx2
                     - (ay(m,k+1,j,i) - ay(m,k,j,i))/dx3;
    b0.x2f(m,k,j,i) = (ax(m,k+1,j,i) - ax(m,k,j,i))/dx3
                     - (az(m,k,j,i+1) - az(m,k,j,i))/dx1;
    b0.x3f(m,k,j,i) = (ay(m,k,j,i+1) - ay(m,k,j,i))/dx1
                     - (ax(m,k,j+1,i) - ax(m,k,j,i))/dx2;

    // Extra face at the upper boundary of each block
    if (i == ie) {
      b0.x1f(m,k,j,i+1) = (az(m,k,j+1,i+1) - az(m,k,j,i+1))/dx2
                          - (ay(m,k+1,j,i+1) - ay(m,k,j,i+1))/dx3;
    }
    if (j == je) {
      b0.x2f(m,k,j+1,i) = (ax(m,k+1,j+1,i) - ax(m,k,j+1,i))/dx3
                          - (az(m,k,j+1,i+1) - az(m,k,j+1,i))/dx1;
    }
    if (k == ke) {
      b0.x3f(m,k+1,j,i) = (ay(m,k+1,j,i+1) - ay(m,k+1,j,i))/dx1
                          - (ax(m,k+1,j+1,i) - ax(m,k+1,j,i))/dx2;
    }
  });

  // -------------------------------------------------------------------------
  // Step 3: Remove global mean and normalise to rms_b
  // -------------------------------------------------------------------------
  SubtractGlobalMeanB(pmbp, b0);
  NormalizeRmsB(pmbp, b0, rms_b);

  // -------------------------------------------------------------------------
  // Step 4: Add optional uniform background field
  // -------------------------------------------------------------------------
  if (bmean_x != 0.0 || bmean_y != 0.0 || bmean_z != 0.0) {
    par_for("spec_ic_bg_B", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) += bmean_x;
      b0.x2f(m,k,j,i) += bmean_y;
      b0.x3f(m,k,j,i) += bmean_z;
      if (i == ie) b0.x1f(m,k,j,i+1) += bmean_x;
      if (j == je) b0.x2f(m,k,j+1,i) += bmean_y;
      if (k == ke) b0.x3f(m,k+1,j,i) += bmean_z;
    });
  }

  // -------------------------------------------------------------------------
  // Step 5: Compute cell-centred bcc0 (needed for relativistic MHD)
  // -------------------------------------------------------------------------
  if (is_relativistic) {
    par_for("spec_ic_bcc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bcc0(m,IBX,k,j,i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      bcc0(m,IBY,k,j,i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      bcc0(m,IBZ,k,j,i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
  }

  // -------------------------------------------------------------------------
  // Step 6: Initialise conserved fluid variables
  // -------------------------------------------------------------------------
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto u0 = pmbp->pmhd->u0;

  par_for("spec_ic_fluid", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0 * vx0;
    u0(m,IM2,k,j,i) = rho0 * vy0;
    u0(m,IM3,k,j,i) = rho0 * vz0;
    if (eos.is_ideal) {
      Real ke_fluid = 0.5*rho0*(vx0*vx0 + vy0*vy0 + vz0*vz0);
      Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
      u0(m,IEN,k,j,i) = pres0/gm1 + ke_fluid + 0.5*(bx*bx + by*by + bz*bz);
    }
  });
}
