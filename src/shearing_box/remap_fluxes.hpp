#ifndef SHEARING_BOX_ORBITAL_ADVECTION_HPP_
#define SHEARING_BOX_ORBITAL_ADVECTION_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file orbital_advection.hpp
//! \brief Inline functions to compute "fluxes" for conservative remap in orbital
//! advection. Based on RemapFlux functions in athena4.2.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn DonorCellOrbAdvFlx()

KOKKOS_INLINE_FUNCTION
void DonorCellOrbAdvFlx(TeamMember_t const &member, const int jl, const int ju,
                        const Real eps, const ScrArray1D<Real> &u, ScrArray1D<Real> &q1,
                        ScrArray1D<Real> &ust) {
  if (eps > 0.0) {
    par_for_inner(member, jl, ju, [&](const int j) {
      ust(j) = u(j-1);
    });
  } else {
    par_for_inner(member, jl, ju, [&](const int j) {
      ust(j) = u(j);
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearOrbAdvFlx()

KOKKOS_INLINE_FUNCTION
void PcwsLinearOrbAdvFlx(TeamMember_t const &member, const int jl, const int ju,
                         const Real eps, const ScrArray1D<Real> &u, ScrArray1D<Real> &q1,
                         ScrArray1D<Real> &ust) {

  // compute limited slopes
  par_for_inner(member, jl-1, ju, [&](const int j) {
    Real dql = u(j  ) - u(j-1);
    Real dqr = u(j+1) - u(j  );
    // Apply limiter
    Real dq2 = dql*dqr;
    q1(j) = 0.0;
    if (dq2 > 0.0) q1(j) = dq2/(dql + dqr);
  });
  // compute upwind state (U_star)
  if (eps > 0.0) {
    par_for_inner(member, jl, ju, [&](const int j) {
      ust(j) = (u(j-1) + 0.5*(1.0 - eps)*q1(j-1));
    });
  } else {
    par_for_inner(member, jl, ju, [&](const int j) {
      ust(j) = (u(j) - 0.5*(1.0 + eps)*q1(j));
    });
  }
  return;
}

#endif // SHEARING_BOX_ORBITAL_ADVECTION_HPP_
