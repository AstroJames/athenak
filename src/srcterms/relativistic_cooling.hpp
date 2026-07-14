#ifndef SRCTERMS_RELATIVISTIC_COOLING_HPP_
#define SRCTERMS_RELATIVISTIC_COOLING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file relativistic_cooling.hpp
//! \brief Local algebra for covariant entropy-relaxation cooling in flat spacetime.

#include <cmath>

#include "athena.hpp"

namespace relativistic_cooling {

//----------------------------------------------------------------------------------------
//! \brief Positive removal rates associated with G^mu_cool = -lambda u^mu.

struct CoolingRates {
  Real requested_lambda;
  Real lambda;
  Real limited_lambda;
  Real g0;
  Real g1;
  Real g2;
  Real g3;
};

//----------------------------------------------------------------------------------------
//! \brief Compute entropy-relaxation cooling and a proper-internal-energy limiter.
//!
//! The target adiabat is K0 = p/rho^gamma.  Cooling is one-sided, so material at or
//! below K0 is untouched.  The optional stage limiter uses d e/d tau = -lambda to keep
//! the proper internal energy above its EOS floor over the explicit stage.

KOKKOS_INLINE_FUNCTION
CoolingRates EntropyRelaxation(const Real rho, const Real eint,
                               const Real u1, const Real u2, const Real u3,
                               const Real gamma, const Real target_adiabat,
                               const Real cooling_time, const Real stage_dt,
                               const Real eint_floor) {
  CoolingRates rates = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (rho <= 0.0 || eint <= 0.0) return rates;

  const Real pressure = (gamma - 1.0)*eint;
  const Real adiabat = pressure/pow(rho, gamma);
  const Real entropy_excess = fmax(log(adiabat/target_adiabat), 0.0);
  rates.requested_lambda = eint*entropy_excess/cooling_time;

  const Real lorentz = sqrt(1.0 + u1*u1 + u2*u2 + u3*u3);
  rates.lambda = rates.requested_lambda;
  if (stage_dt > 0.0) {
    const Real available_eint = fmax(eint - eint_floor, 0.0);
    const Real lambda_max = available_eint*lorentz/stage_dt;
    rates.lambda = fmin(rates.lambda, lambda_max);
  }
  rates.limited_lambda = rates.requested_lambda - rates.lambda;
  rates.g0 = lorentz*rates.lambda;
  rates.g1 = u1*rates.lambda;
  rates.g2 = u2*rates.lambda;
  rates.g3 = u3*rates.lambda;
  return rates;
}

} // namespace relativistic_cooling

#endif  // SRCTERMS_RELATIVISTIC_COOLING_HPP_
