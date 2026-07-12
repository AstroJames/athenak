#ifndef EOS_RESISTIVE_SRMHD_HPP_
#define EOS_RESISTIVE_SRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_srmhd.hpp
//! \brief Single-state conversions for resistive special-relativistic MHD with known E.

#include "athena.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"

namespace srrmhd {

// Cell-centered electric-field indices local to resistive SRMHD.  The global passive
// scalar index IYF remains unchanged because Hydro and ideal MHD rely on it.
inline constexpr int IRE1 = 5;
inline constexpr int IRE2 = 6;
inline constexpr int IRE3 = 7;
inline constexpr int NSRRMHD = 8;

//----------------------------------------------------------------------------------------
//! \fn Real CellCenteredCharge()
//! \brief Discrete Gauss-law charge used by both evolution and diagnostics.
//!
//! Electric fields are cell centered.  Use the same centered second-order divergence
//! in the explicit advective-current source and in q output/tests so that the two paths
//! cannot acquire inconsistent signs or stencils.  Ghost zones supply boundary values.

template <typename Array5D>
KOKKOS_INLINE_FUNCTION
Real CellCenteredCharge(const Array5D &e, const int m, const int k, const int j,
                        const int i, const Real idx1, const Real idx2,
                        const Real idx3, const bool multi_d, const bool three_d) {
  Real q = 0.5*idx1*(e(m, IRE1, k, j, i+1) - e(m, IRE1, k, j, i-1));
  if (multi_d) {
    q += 0.5*idx2*(e(m, IRE2, k, j+1, i) - e(m, IRE2, k, j-1, i));
  }
  if (three_d) {
    q += 0.5*idx3*(e(m, IRE3, k+1, j, i) - e(m, IRE3, k-1, j, i));
  }
  return q;
}

struct SRRMHDPrim1D {
  Real d, vx, vy, vz, e;
  Real ex, ey, ez;
  Real bx, by, bz;
};

struct SRRMHDCons1D {
  Real d, mx, my, mz, e;
  Real ex, ey, ez;
  Real bx, by, bz;
};

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealSRRMHD()
//! \brief Convert one resistive-SRMHD primitive state to total conserved variables.
//!
//! The hydrodynamic conversion produces the fluid momentum and tau.  Electromagnetic
//! momentum E cross B and energy (E^2+B^2)/2 are then added to obtain the conserved
//! total stress-energy variables.  Electric and magnetic fields use Heaviside-Lorentz
//! units with c=1.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealSRRMHD(const SRRMHDPrim1D &w, const Real gamma,
                           SRRMHDCons1D &u) {
  HydPrim1D wh;
  wh.d = w.d;
  wh.vx = w.vx;
  wh.vy = w.vy;
  wh.vz = w.vz;
  wh.e = w.e;

  HydCons1D uh;
  SingleP2C_IdealSRHyd(wh, gamma, uh);

  u.d = uh.d;
  u.mx = uh.mx + w.ey*w.bz - w.ez*w.by;
  u.my = uh.my + w.ez*w.bx - w.ex*w.bz;
  u.mz = uh.mz + w.ex*w.by - w.ey*w.bx;
  u.e = uh.e + 0.5*(SQR(w.ex) + SQR(w.ey) + SQR(w.ez)
                    + SQR(w.bx) + SQR(w.by) + SQR(w.bz));
  u.ex = w.ex;
  u.ey = w.ey;
  u.ez = w.ez;
  u.bx = w.bx;
  u.by = w.by;
  u.bz = w.bz;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRRMHDKnownE()
//! \brief Recover one resistive-SRMHD primitive state for a known electric field.
//!
//! This is not the coupled implicit-stage solve.  It removes the electromagnetic
//! momentum and energy from a fixed total conserved state, then reuses the ideal SR
//! hydrodynamic inversion for the remaining fluid state.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRRMHDKnownE(const SRRMHDCons1D &u, const EOS_Data &eos,
                                 SRRMHDPrim1D &w, bool &dfloor_used,
                                 bool &efloor_used, bool &c2p_failure,
                                 int &iter_used) {
  HydCons1D uh;
  uh.d = u.d;
  uh.mx = u.mx - (u.ey*u.bz - u.ez*u.by);
  uh.my = u.my - (u.ez*u.bx - u.ex*u.bz);
  uh.mz = u.mz - (u.ex*u.by - u.ey*u.bx);
  uh.e = u.e - 0.5*(SQR(u.ex) + SQR(u.ey) + SQR(u.ez)
                    + SQR(u.bx) + SQR(u.by) + SQR(u.bz));

  Real s2 = SQR(uh.mx) + SQR(uh.my) + SQR(uh.mz);
  HydPrim1D wh;
  SingleC2P_IdealSRHyd(uh, eos, s2, wh, dfloor_used, efloor_used, c2p_failure,
                       iter_used);

  w.d = wh.d;
  w.vx = wh.vx;
  w.vy = wh.vy;
  w.vz = wh.vz;
  w.e = wh.e;
  w.ex = u.ex;
  w.ey = u.ey;
  w.ez = u.ez;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;
}

//----------------------------------------------------------------------------------------
//! \fn void ImplicitElectricField()
//! \brief Solve the local backward-Euler Ohmic update for a fixed four-velocity.
//!
//! For kappa = a_dt/eta, Ampere's stiff source gives
//!   [I + kappa*Gamma*(I-v v^T)] E = E_star - kappa*Gamma*(v cross B).
//! The rank-one matrix is inverted analytically.  This routine is also the exact E
//! elimination used by the coupled primitive recovery below.

KOKKOS_INLINE_FUNCTION
void ImplicitElectricField(const Real ux, const Real uy, const Real uz,
                           const Real ex_star, const Real ey_star,
                           const Real ez_star, const Real bx, const Real by,
                           const Real bz, const Real kappa, Real &ex, Real &ey,
                           Real &ez) {
  const Real lor = sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
  const Real vx = ux/lor;
  const Real vy = uy/lor;
  const Real vz = uz/lor;
  const Real a = kappa*lor;
  const Real inv_d = 1.0/(1.0 + a);
  const Real inv_parallel_d = 1.0/(1.0 + kappa/lor);

  const Real vxb_x = vy*bz - vz*by;
  const Real vxb_y = vz*bx - vx*bz;
  const Real vxb_z = vx*by - vy*bx;
  const Real v_dot_estar = vx*ex_star + vy*ey_star + vz*ez_star;
  const Real parallel_factor = a*inv_d*inv_parallel_d*v_dot_estar;

  ex = (ex_star - a*vxb_x)*inv_d + parallel_factor*vx;
  ey = (ey_star - a*vxb_y)*inv_d + parallel_factor*vy;
  ez = (ez_star - a*vxb_z)*inv_d + parallel_factor*vz;
}

//----------------------------------------------------------------------------------------
//! \fn bool ImplicitPrimitiveResidual()
//! \brief Evaluate the three-velocity residual after analytically eliminating E and p.

KOKKOS_INLINE_FUNCTION
bool ImplicitPrimitiveResidual(const SRRMHDCons1D &u, const EOS_Data &eos,
                               const Real ex_star, const Real ey_star,
                               const Real ez_star, const Real kappa, const Real ux,
                               const Real uy, const Real uz, Real &fx, Real &fy,
                               Real &fz, SRRMHDPrim1D &w) {
  if (!(u.d > 0.0) || !(eos.gamma > 1.0)) return false;

  const Real lor = sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
  ImplicitElectricField(ux, uy, uz, ex_star, ey_star, ez_star, u.bx, u.by, u.bz,
                        kappa, w.ex, w.ey, w.ez);

  const Real sx_fl = u.mx - (w.ey*u.bz - w.ez*u.by);
  const Real sy_fl = u.my - (w.ez*u.bx - w.ex*u.bz);
  const Real sz_fl = u.mz - (w.ex*u.by - w.ey*u.bx);
  const Real tau_fl = u.e - 0.5*(SQR(w.ex) + SQR(w.ey) + SQR(w.ez)
                                  + SQR(u.bx) + SQR(u.by) + SQR(u.bz));

  const Real gm1 = eos.gamma - 1.0;
  const Real pressure_den = eos.gamma*SQR(lor)/gm1 - 1.0;
  const Real pressure = (tau_fl + u.d - u.d*lor)/pressure_den;
  if (!(pressure > 0.0) || !isfinite(pressure)) return false;

  w.d = u.d/lor;
  w.vx = ux;
  w.vy = uy;
  w.vz = uz;
  w.e = pressure/gm1;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;

  const Real h = 1.0 + eos.gamma*w.e/w.d;
  const Real inv_dh = 1.0/(u.d*h);
  fx = ux - sx_fl*inv_dh;
  fy = uy - sy_fl*inv_dh;
  fz = uz - sz_fl*inv_dh;
  return isfinite(fx) && isfinite(fy) && isfinite(fz);
}

//----------------------------------------------------------------------------------------
//! \fn bool SolveLinear3x3()
//! \brief Pivoted Gaussian elimination for the Newton correction.

KOKKOS_INLINE_FUNCTION
bool SolveLinear3x3(Real a[3][3], Real b[3], Real x[3]) {
  for (int col = 0; col < 3; ++col) {
    int pivot = col;
    Real pivot_abs = fabs(a[col][col]);
    for (int row = col + 1; row < 3; ++row) {
      if (fabs(a[row][col]) > pivot_abs) {
        pivot = row;
        pivot_abs = fabs(a[row][col]);
      }
    }
    if (!(pivot_abs > 1.0e-14) || !isfinite(pivot_abs)) return false;
    if (pivot != col) {
      for (int j = col; j < 3; ++j) {
        const Real tmp = a[col][j];
        a[col][j] = a[pivot][j];
        a[pivot][j] = tmp;
      }
      const Real tmp = b[col];
      b[col] = b[pivot];
      b[pivot] = tmp;
    }
    for (int row = col + 1; row < 3; ++row) {
      const Real factor = a[row][col]/a[col][col];
      for (int j = col; j < 3; ++j) a[row][j] -= factor*a[col][j];
      b[row] -= factor*b[col];
    }
  }

  for (int row = 2; row >= 0; --row) {
    Real rhs = b[row];
    for (int j = row + 1; j < 3; ++j) rhs -= a[row][j]*x[j];
    if (!(fabs(a[row][row]) > 1.0e-14)) return false;
    x[row] = rhs/a[row][row];
  }
  return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]);
}

//----------------------------------------------------------------------------------------
//! \fn bool SingleC2P_IdealSRRMHDImplicit()
//! \brief Coupled local implicit-stage recovery of u, E, rho, and internal energy.
//!
//! The first implementation uses a centered finite-difference Jacobian and a damped
//! three-variable Newton iteration.  All storage and loop bounds are fixed for device
//! execution.  No floors are applied inside the nonlinear solve: an inadmissible state
//! is reported to the caller as a failure.

KOKKOS_INLINE_FUNCTION
bool SingleC2P_IdealSRRMHDImplicit(const SRRMHDCons1D &u, const EOS_Data &eos,
                                  const Real ex_star, const Real ey_star,
                                  const Real ez_star, const Real kappa,
                                  const SRRMHDPrim1D &guess, SRRMHDPrim1D &w,
                                  int &iter_used) {
  constexpr int max_iterations = 30;
  constexpr int max_backtracks = 10;
  const Real tolerance = (sizeof(Real) == sizeof(float)) ? 2.0e-6 : 2.0e-12;
  const Real fd_scale = (sizeof(Real) == sizeof(float)) ? 3.0e-3 : 1.0e-6;

  Real velocity[3] = {guess.vx, guess.vy, guess.vz};
  Real residual[3];
  iter_used = 0;
  if (!ImplicitPrimitiveResidual(u, eos, ex_star, ey_star, ez_star, kappa,
                                 velocity[0], velocity[1], velocity[2], residual[0],
                                 residual[1], residual[2], w)) {
    return false;
  }

  for (iter_used = 0; iter_used < max_iterations; ++iter_used) {
    const Real residual_norm = fmax(fabs(residual[0]),
                                    fmax(fabs(residual[1]), fabs(residual[2])));
    if (residual_norm < tolerance) return true;

    Real jacobian[3][3];
    for (int col = 0; col < 3; ++col) {
      const Real step = fd_scale*(1.0 + fabs(velocity[col]));
      Real plus[3] = {velocity[0], velocity[1], velocity[2]};
      Real minus[3] = {velocity[0], velocity[1], velocity[2]};
      plus[col] += step;
      minus[col] -= step;
      Real fplus[3], fminus[3];
      SRRMHDPrim1D scratch;
      const bool plus_ok = ImplicitPrimitiveResidual(
          u, eos, ex_star, ey_star, ez_star, kappa, plus[0], plus[1], plus[2],
          fplus[0], fplus[1], fplus[2], scratch);
      const bool minus_ok = ImplicitPrimitiveResidual(
          u, eos, ex_star, ey_star, ez_star, kappa, minus[0], minus[1], minus[2],
          fminus[0], fminus[1], fminus[2], scratch);
      if (!plus_ok || !minus_ok) return false;
      for (int row = 0; row < 3; ++row) {
        jacobian[row][col] = (fplus[row] - fminus[row])/(2.0*step);
      }
    }

    Real rhs[3] = {-residual[0], -residual[1], -residual[2]};
    Real correction[3] = {0.0, 0.0, 0.0};
    if (!SolveLinear3x3(jacobian, rhs, correction)) return false;

    bool accepted = false;
    Real damping = 1.0;
    for (int backtrack = 0; backtrack < max_backtracks; ++backtrack) {
      Real trial[3] = {velocity[0] + damping*correction[0],
                       velocity[1] + damping*correction[1],
                       velocity[2] + damping*correction[2]};
      Real trial_residual[3];
      SRRMHDPrim1D trial_w;
      const bool trial_ok = ImplicitPrimitiveResidual(
          u, eos, ex_star, ey_star, ez_star, kappa, trial[0], trial[1], trial[2],
          trial_residual[0], trial_residual[1], trial_residual[2], trial_w);
      Real trial_norm = 2.0*residual_norm;
      if (trial_ok) {
        trial_norm = fmax(fabs(trial_residual[0]),
                          fmax(fabs(trial_residual[1]), fabs(trial_residual[2])));
      }
      if (trial_norm < residual_norm) {
        for (int n = 0; n < 3; ++n) {
          velocity[n] = trial[n];
          residual[n] = trial_residual[n];
        }
        w = trial_w;
        accepted = true;
        break;
      }
      damping *= 0.5;
    }
    if (!accepted) return false;
  }
  return false;
}

} // namespace srrmhd

#endif // EOS_RESISTIVE_SRMHD_HPP_
