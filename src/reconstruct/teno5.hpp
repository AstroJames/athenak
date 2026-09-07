#ifndef RECONSTRUCT_TENO5_HPP_
#define RECONSTRUCT_TENO5_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file teno5.hpp
//! \brief Five-point targeted ENO reconstruction for uniform Cartesian-like grids.
//!
//! Standard TENO5 has fifth-order formal accuracy.  TENO5-opt uses the paper's
//! spectrally optimized background operator, which deliberately reduces the formal
//! order to four in exchange for lower intermediate- and high-wavenumber dissipation.
//!
//! REFERENCES:
//! Fu L., Hu X.Y., Adams N.A., "A family of high-order targeted ENO schemes for
//! compressible-fluid simulations", JCP, 305, 333-359 (2016)

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn TENO5StencilSelection()
//! \brief Select smooth TENO5 candidate stencils.
//!
//! The ratios r_k = (beta_k + eps)/(beta_k + eps + tau_5) are the reciprocals of
//! the scale-separation factors in Fu et al.  Normalizing them by their minimum before
//! applying q=6 is algebraically equivalent to normalizing gamma_k, but avoids overflow
//! when beta_k is very small.  The standard TENO5 cutoff is C_T=1.e-5.

KOKKOS_INLINE_FUNCTION
void TENO5StencilSelection(const Real beta0, const Real beta1, const Real beta2,
                          const Real cutoff,
                          Real &delta0, Real &delta1, Real &delta2) noexcept {
#if SINGLE_PRECISION_ENABLED
  const Real eps = 1.0e-20;
#else
  const Real eps = 1.0e-40;
#endif
  const Real tau5 = fabs(beta0 - beta2);
  const Real ratio0 = (beta0 + eps)/(beta0 + eps + tau5);
  const Real ratio1 = (beta1 + eps)/(beta1 + eps + tau5);
  const Real ratio2 = (beta2 + eps)/(beta2 + eps + tau5);
  const Real ratio_min = fmin(ratio0, fmin(ratio1, ratio2));

  const Real scaled0 = ratio_min/ratio0;
  const Real scaled1 = ratio_min/ratio1;
  const Real scaled2 = ratio_min/ratio2;
  const Real scaled0_sq = SQR(scaled0);
  const Real scaled1_sq = SQR(scaled1);
  const Real scaled2_sq = SQR(scaled2);
  const Real gamma0 = scaled0_sq*scaled0_sq*scaled0_sq;
  const Real gamma1 = scaled1_sq*scaled1_sq*scaled1_sq;
  const Real gamma2 = scaled2_sq*scaled2_sq*scaled2_sq;
  const Real gamma_sum = gamma0 + gamma1 + gamma2;

  const Real cutoff_sum = cutoff*gamma_sum;
  delta0 = (gamma0 < cutoff_sum) ? 0.0 : 1.0;
  delta1 = (gamma1 < cutoff_sum) ? 0.0 : 1.0;
  delta2 = (gamma2 < cutoff_sum) ? 0.0 : 1.0;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn TENO5Weights()
//! \brief Normalize the optimal weights of the selected TENO5 candidate stencils.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO5Weights(const Real delta0, const Real delta1, const Real delta2,
                  Real &weight0, Real &weight1, Real &weight2) noexcept {
  Real d0, d1, d2;
  if constexpr (optimized) {
    // TENO5-opt weights from Fu et al. Table 5, mapped from the paper's
    // incremental-width ordering to the classical left/center/right candidates.
    // These weights give the spectrally optimized fourth-order background operator.
    d0 = 0.05;
    d1 = 0.55;
    d2 = 0.40;
  } else {
    d0 = 0.1;
    d1 = 0.6;
    d2 = 0.3;
  }

  // Keep the all-smooth state first since it is the common high-order path.  The
  // rejected-stencil path preserves the arithmetic of the original normalized weights.
  if (delta0 == 1.0 && delta1 == 1.0 && delta2 == 1.0) {
    weight0 = d0;
    weight1 = d1;
    weight2 = d2;
    return;
  }

  const Real weight_sum = d0*delta0 + d1*delta1 + d2*delta2;
  weight0 = d0*delta0/weight_sum;
  weight1 = d1*delta1/weight_sum;
  weight2 = d2*delta2/weight_sum;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn TENO5()
//! \brief Reconstruct a fifth-order polynomial to compute ql(i+1) and qr(i).
//! Works in any dimension by passing the appropriate q_im2,...,q_ip2.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO5(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
           const Real &q_ip2, const Real cutoff, Real &ql_ip1, Real &qr_i) noexcept {
  const Real beta_coeff[2]{13.0/12.0, 0.25};

  Real beta[3];
  beta[0] = beta_coeff[0]*SQR(q_im2 + q_i - 2.0*q_im1)
          + beta_coeff[1]*SQR(q_im2 + 3.0*q_i - 4.0*q_im1);
  beta[1] = beta_coeff[0]*SQR(q_im1 + q_ip1 - 2.0*q_i)
          + beta_coeff[1]*SQR(q_im1 - q_ip1);
  beta[2] = beta_coeff[0]*SQR(q_ip2 + q_i - 2.0*q_ip1)
          + beta_coeff[1]*SQR(q_ip2 + 3.0*q_i - 4.0*q_ip1);

  Real delta[3];
  TENO5StencilSelection(beta[0], beta[1], beta[2], cutoff,
                        delta[0], delta[1], delta[2]);

  Real weight[3];
  TENO5Weights<optimized>(delta[0], delta[1], delta[2],
                          weight[0], weight[1], weight[2]);
  Real f0 = 2.0*q_im2 - 7.0*q_im1 + 11.0*q_i;
  Real f1 = -q_im1 + 5.0*q_i + 2.0*q_ip1;
  Real f2 = 2.0*q_i + 5.0*q_ip1 - q_ip2;
  ql_ip1 = (weight[0]*f0 + weight[1]*f1 + weight[2]*f2)/6.0;

  TENO5Weights<optimized>(delta[2], delta[1], delta[0],
                          weight[0], weight[1], weight[2]);
  f0 = 2.0*q_ip2 - 7.0*q_ip1 + 11.0*q_i;
  f1 = -q_ip1 + 5.0*q_i + 2.0*q_im1;
  f2 = 2.0*q_i + 5.0*q_im1 - q_im2;
  qr_i = (weight[0]*f0 + weight[1]*f1 + weight[2]*f2)/6.0;
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Wrapper for TENO5 reconstruction in the x1 direction.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO5X1(TeamMember_t const &member, const EOS_Data &eos, const Real cutoff,
     const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      Real &qim2 = q(m,n,k,j,i-2);
      Real &qim1 = q(m,n,k,j,i-1);
      Real &qi   = q(m,n,k,j,i  );
      Real &qip1 = q(m,n,k,j,i+1);
      Real &qip2 = q(m,n,k,j,i+2);
      TENO5<optimized>(qim2, qim1, qi, qip1, qip2, cutoff, ql(n,i+1), qr(n,i));
      if (apply_floors) {
        if (n == IDN) {
          ql(IDN,i+1) = fmax(ql(IDN,i+1), dfloor_);
          qr(IDN,i  ) = fmax(qr(IDN,i  ), dfloor_);
        }
        if (n == IEN) {
          ql(IEN,i+1) = fmax(ql(IEN,i+1), efloor_);
          qr(IEN,i  ) = fmax(qr(IEN,i  ), efloor_);
        }
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Wrapper for TENO5 reconstruction in the x2 direction.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO5X2(TeamMember_t const &member, const EOS_Data &eos, const Real cutoff,
     const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      Real &qjm2 = q(m,n,k,j-2,i);
      Real &qjm1 = q(m,n,k,j-1,i);
      Real &qj   = q(m,n,k,j  ,i);
      Real &qjp1 = q(m,n,k,j+1,i);
      Real &qjp2 = q(m,n,k,j+2,i);
      TENO5<optimized>(qjm2, qjm1, qj, qjp1, qjp2, cutoff, ql_jp1(n,i), qr_j(n,i));
      if (apply_floors) {
        if (n == IDN) {
          ql_jp1(IDN,i) = fmax(ql_jp1(IDN,i), dfloor_);
          qr_j  (IDN,i) = fmax(qr_j  (IDN,i), dfloor_);
        }
        if (n == IEN) {
          ql_jp1(IEN,i) = fmax(ql_jp1(IEN,i), efloor_);
          qr_j  (IEN,i) = fmax(qr_j  (IEN,i), efloor_);
        }
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Wrapper for TENO5 reconstruction in the x3 direction.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO5X3(TeamMember_t const &member, const EOS_Data &eos, const Real cutoff,
     const bool apply_floors,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k) {
  int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      Real &qkm2 = q(m,n,k-2,j,i);
      Real &qkm1 = q(m,n,k-1,j,i);
      Real &qk   = q(m,n,k  ,j,i);
      Real &qkp1 = q(m,n,k+1,j,i);
      Real &qkp2 = q(m,n,k+2,j,i);
      TENO5<optimized>(qkm2, qkm1, qk, qkp1, qkp2, cutoff, ql_kp1(n,i), qr_k(n,i));
      if (apply_floors) {
        if (n == IDN) {
          ql_kp1(IDN,i) = fmax(ql_kp1(IDN,i), dfloor_);
          qr_k  (IDN,i) = fmax(qr_k  (IDN,i), dfloor_);
        }
        if (n == IEN) {
          ql_kp1(IEN,i) = fmax(ql_kp1(IEN,i), efloor_);
          qr_k  (IEN,i) = fmax(qr_k  (IEN,i), efloor_);
        }
      }
    });
  }
  return;
}

#endif // RECONSTRUCT_TENO5_HPP_
