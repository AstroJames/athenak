#ifndef SRCTERMS_RELATIVISTIC_FORCING_HPP_
#define SRCTERMS_RELATIVISTIC_FORCING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file relativistic_forcing.hpp
//! \brief Device-callable algebra for mechanical forcing in special relativity.

#include "athena.hpp"

namespace srrmhd {

// The stochastic driver supplies the spatial components of proper four-acceleration in
// the preferred box frame.  The temporal component is fixed by u_mu A^mu = 0.
struct ProperFourAcceleration {
  Real a0 = 0.0;
  Real a1 = 0.0;
  Real a2 = 0.0;
  Real a3 = 0.0;
};

// Mechanical four-force density G^mu = s*w*A^mu.  The energy component is the
// box-frame power density, and the spatial components source conserved momentum.
struct MechanicalFourSource {
  Real g0 = 0.0;
  Real g1 = 0.0;
  Real g2 = 0.0;
  Real g3 = 0.0;
};

//----------------------------------------------------------------------------------------
//! \brief Complete spatial proper acceleration using AthenaK spatial four-velocity.

KOKKOS_INLINE_FUNCTION
ProperFourAcceleration CompleteProperAcceleration(const Real u1, const Real u2,
                                                   const Real u3, const Real a1,
                                                   const Real a2, const Real a3) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  ProperFourAcceleration acceleration;
  acceleration.a0 = (u1*a1 + u2*a2 + u3*a3)/lor;
  acceleration.a1 = a1;
  acceleration.a2 = a2;
  acceleration.a3 = a3;
  return acceleration;
}

//----------------------------------------------------------------------------------------
//! \brief Construct G^mu=s*w*A^mu for a box-frame mechanical acceleration.
//!
//! The caller supplies the positive matter enthalpy density w=rho+e+p and the scalar
//! acceleration amplitude s.  This helper performs no floors or normalization.

KOKKOS_INLINE_FUNCTION
MechanicalFourSource MechanicalForcingSource(const Real u1, const Real u2,
                                              const Real u3,
                                              const Real enthalpy_density,
                                              const Real amplitude, const Real a1,
                                              const Real a2, const Real a3) {
  const ProperFourAcceleration acceleration =
      CompleteProperAcceleration(u1, u2, u3, a1, a2, a3);
  const Real force_scale = amplitude*enthalpy_density;
  MechanicalFourSource source;
  source.g0 = force_scale*acceleration.a0;
  source.g1 = force_scale*acceleration.a1;
  source.g2 = force_scale*acceleration.a2;
  source.g3 = force_scale*acceleration.a3;
  return source;
}

}  // namespace srrmhd

#endif  // SRCTERMS_RELATIVISTIC_FORCING_HPP_
