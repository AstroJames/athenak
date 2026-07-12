#ifndef MHD_RESISTIVITY_MODEL_HPP_
#define MHD_RESISTIVITY_MODEL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file resistivity_model.hpp
//! \brief Device-callable scalar resistivity models for resistive SRMHD.

#include "athena.hpp"

namespace srrmhd {

enum class ResistivityModel {uniform, charge_starvation};

struct ResistivityData {
  ResistivityModel model = ResistivityModel::uniform;
  Real eta_uniform = 0.0;
  Real eta_floor = 1.0e-8;
  Real eta_scale = 1.0;
  Real number_per_mass = 1.0;
};

//----------------------------------------------------------------------------------------
//! \brief Evaluate eta from known local primitives and fields.
//!
//! Primitive velocities are AthenaK's spatial four-velocity components.  For the
//! charge-starvation model,
//!   eta = max(eta_floor, eta_scale |Gamma(E + v x B - (E.v)v)| / n),
//! with n = number_per_mass*rho.  The caller freezes the returned value for an entire
//! IMEX stage; this evaluator is never called from a Newton or Picard residual.

KOKKOS_INLINE_FUNCTION
Real EvaluateResistivity(const ResistivityData &data, const Real rho, const Real u1,
                         const Real u2, const Real u3, const Real e1, const Real e2,
                         const Real e3, const Real b1, const Real b2, const Real b3) {
  if (data.model == ResistivityModel::uniform) return data.eta_uniform;

  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real v1 = u1/lor;
  const Real v2 = u2/lor;
  const Real v3 = u3/lor;
  const Real edotv = e1*v1 + e2*v2 + e3*v3;
  const Real es1 = lor*(e1 + v2*b3 - v3*b2 - edotv*v1);
  const Real es2 = lor*(e2 + v3*b1 - v1*b3 - edotv*v2);
  const Real es3 = lor*(e3 + v1*b2 - v2*b1 - edotv*v3);
  const Real number_density = data.number_per_mass*rho;
  const Real eta_eff = data.eta_scale
                     * sqrt(SQR(es1) + SQR(es2) + SQR(es3))/number_density;
  if (!(eta_eff >= 0.0) || !isfinite(eta_eff)) return data.eta_floor;
  return fmax(data.eta_floor, eta_eff);
}

} // namespace srrmhd

#endif // MHD_RESISTIVITY_MODEL_HPP_
