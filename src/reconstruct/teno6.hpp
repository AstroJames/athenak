#ifndef RECONSTRUCT_TENO6_HPP_
#define RECONSTRUCT_TENO6_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file teno6.hpp
//! \brief Six-point targeted ENO reconstruction for uniform Cartesian-like grids.
//!
//! Standard TENO6 has sixth-order formal accuracy and a non-dissipative central
//! background operator.  TENO6-opt uses the paper's fifth-order, slightly upwind-biased
//! background operator to add controlled high-wavenumber dissipation.
//!
//! REFERENCES:
//! Fu L., Hu X.Y., Adams N.A., "A family of high-order targeted ENO schemes for
//! compressible-fluid simulations", JCP, 305, 333-359 (2016)

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn TENO6Beta4()
//! \brief Jiang-Shu smoothness indicator for the four-point candidate {i,...,i+3}.
//!
//! The positive sum-of-squares form is algebraically equivalent to Eq. (23) of Fu et al.
//! It avoids a small negative indicator from cancellation in the expanded quadratic.

KOKKOS_INLINE_FUNCTION
Real TENO6Beta4(const Real q_i, const Real q_ip1, const Real q_ip2,
                const Real q_ip3) noexcept {
  // Coefficients of the cubic whose cell averages are q_i,...,q_ip3, using cell i as
  // the origin.  Its constant coefficient does not contribute to smoothness.
  const Real a1 = (-43.0*q_i + 69.0*q_ip1 - 33.0*q_ip2 + 7.0*q_ip3)/24.0;
  const Real a2 = (2.0*q_i - 5.0*q_ip1 + 4.0*q_ip2 - q_ip3)/2.0;
  const Real a3 = (-q_i + 3.0*q_ip1 - 3.0*q_ip2 + q_ip3)/6.0;
  const Real z1 = a1 + 0.25*a3;
  return SQR(z1) + (13.0/3.0)*SQR(a2) + (781.0/20.0)*SQR(a3);
}

//----------------------------------------------------------------------------------------
//! \fn TENO6Beta6()
//! \brief Full six-point WENO-CU6 smoothness indicator used to construct tau_6.
//!
//! This is Eq. (18) of Fu et al. in a positive sum-of-squares factorization.  The
//! polynomial coefficients are reconstructed from cell averages on {i-2,...,i+3}.

KOKKOS_INLINE_FUNCTION
Real TENO6Beta6(const Real q_im2, const Real q_im1, const Real q_i,
                const Real q_ip1, const Real q_ip2, const Real q_ip3) noexcept {
  const Real a1 = (341.0*q_im2 - 2785.0*q_im1 - 2590.0*q_i + 6670.0*q_ip1
                  - 1895.0*q_ip2 + 259.0*q_ip3)/5760.0;
  const Real a2 = (-q_im2 + 12.0*q_im1 - 22.0*q_i + 12.0*q_ip1 - q_ip2)/16.0;
  const Real a3 = (-5.0*q_im2 - 11.0*q_im1 + 70.0*q_i - 94.0*q_ip1
                  + 47.0*q_ip2 - 7.0*q_ip3)/144.0;
  const Real a4 = (q_im2 - 4.0*q_im1 + 6.0*q_i - 4.0*q_ip1 + q_ip2)/24.0;
  const Real a5 = (-q_im2 + 5.0*q_im1 - 10.0*q_i + 10.0*q_ip1
                  - 5.0*q_ip2 + q_ip3)/120.0;

  const Real z1 = a1 + 0.25*a3 + 0.0625*a5;
  const Real z2 = a2 + (63.0/130.0)*a4;
  const Real z3 = a3 + (8825.0/10934.0)*a5;
  return SQR(z1) + (13.0/3.0)*SQR(z2) + (781.0/20.0)*SQR(z3)
       + (1421461.0/2275.0)*SQR(a4) + (21520059541.0/1377684.0)*SQR(a5);
}

//----------------------------------------------------------------------------------------
//! \fn TENO6StencilSelection()
//! \brief Select smooth TENO6 candidates in the paper's {S0,S1,S2,S3} ordering.

KOKKOS_INLINE_FUNCTION
void TENO6StencilSelection(const Real q_im2, const Real q_im1, const Real q_i,
                           const Real q_ip1, const Real q_ip2, const Real q_ip3,
                           const Real cutoff, Real &delta0, Real &delta1,
                           Real &delta2, Real &delta3) noexcept {
  const Real beta_coeff[2]{13.0/12.0, 0.25};
  // Fu et al. order the three classical candidates as center, downwind, upwind.
  const Real beta0 = beta_coeff[0]*SQR(q_im1 - 2.0*q_i + q_ip1)
                   + beta_coeff[1]*SQR(q_im1 - q_ip1);
  const Real beta1 = beta_coeff[0]*SQR(q_i - 2.0*q_ip1 + q_ip2)
                   + beta_coeff[1]*SQR(3.0*q_i - 4.0*q_ip1 + q_ip2);
  const Real beta2 = beta_coeff[0]*SQR(q_im2 - 2.0*q_im1 + q_i)
                   + beta_coeff[1]*SQR(q_im2 - 4.0*q_im1 + 3.0*q_i);
  const Real beta3 = TENO6Beta4(q_i, q_ip1, q_ip2, q_ip3);
  const Real beta6 = TENO6Beta6(q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3);
  const Real tau6 = fabs(beta6 - (beta0 + 4.0*beta1 + beta2)/6.0);

#if SINGLE_PRECISION_ENABLED
  const Real eps = 1.0e-20;
#else
  const Real eps = 1.0e-40;
#endif
  // These are reciprocal scale-separation factors.  Scaling by their minimum before
  // taking q=6 is algebraically equivalent to normalizing gamma_k and avoids overflow.
  const Real ratio0 = (beta0 + eps)/(beta0 + eps + tau6);
  const Real ratio1 = (beta1 + eps)/(beta1 + eps + tau6);
  const Real ratio2 = (beta2 + eps)/(beta2 + eps + tau6);
  const Real ratio3 = (beta3 + eps)/(beta3 + eps + tau6);
  const Real ratio_min = fmin(fmin(ratio0, ratio1), fmin(ratio2, ratio3));

  const Real scaled0_sq = SQR(ratio_min/ratio0);
  const Real scaled1_sq = SQR(ratio_min/ratio1);
  const Real scaled2_sq = SQR(ratio_min/ratio2);
  const Real scaled3_sq = SQR(ratio_min/ratio3);
  const Real gamma0 = scaled0_sq*scaled0_sq*scaled0_sq;
  const Real gamma1 = scaled1_sq*scaled1_sq*scaled1_sq;
  const Real gamma2 = scaled2_sq*scaled2_sq*scaled2_sq;
  const Real gamma3 = scaled3_sq*scaled3_sq*scaled3_sq;
  const Real gamma_sum = gamma0 + gamma1 + gamma2 + gamma3;

  const Real cutoff_sum = cutoff*gamma_sum;
  delta0 = (gamma0 < cutoff_sum) ? 0.0 : 1.0;
  delta1 = (gamma1 < cutoff_sum) ? 0.0 : 1.0;
  delta2 = (gamma2 < cutoff_sum) ? 0.0 : 1.0;
  delta3 = (gamma3 < cutoff_sum) ? 0.0 : 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn TENO6ReconstructSide()
//! \brief Assemble one face state from selected TENO6 candidate polynomials.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
Real TENO6ReconstructSide(const Real q_im2, const Real q_im1, const Real q_i,
                          const Real q_ip1, const Real q_ip2, const Real q_ip3,
                          const Real delta0, const Real delta1, const Real delta2,
                          const Real delta3) noexcept {
  Real d0, d1, d2, d3;
  if constexpr (optimized) {
    d0 = 0.462;
    d1 = 0.300;
    d2 = 0.054;
    d3 = 0.184;
  } else {
    d0 = 0.450;
    d1 = 0.300;
    d2 = 0.050;
    d3 = 0.200;
  }
  const Real weight_sum = d0*delta0 + d1*delta1 + d2*delta2 + d3*delta3;
  const Real weight0 = d0*delta0/weight_sum;
  const Real weight1 = d1*delta1/weight_sum;
  const Real weight2 = d2*delta2/weight_sum;
  const Real weight3 = d3*delta3/weight_sum;

  const Real candidate0 = (-q_im1 + 5.0*q_i + 2.0*q_ip1)/6.0;
  const Real candidate1 = (2.0*q_i + 5.0*q_ip1 - q_ip2)/6.0;
  const Real candidate2 = (2.0*q_im2 - 7.0*q_im1 + 11.0*q_i)/6.0;
  const Real candidate3 = (3.0*q_i + 13.0*q_ip1 - 5.0*q_ip2 + q_ip3)/12.0;
  return weight0*candidate0 + weight1*candidate1
       + weight2*candidate2 + weight3*candidate3;
}

//----------------------------------------------------------------------------------------
//! \fn TENO6()
//! \brief Reconstruct both states at one face from the same six cell averages.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO6(const Real q_im2, const Real q_im1, const Real q_i, const Real q_ip1,
           const Real q_ip2, const Real q_ip3, const Real cutoff,
           Real &ql, Real &qr) noexcept {
  Real dl0, dl1, dl2, dl3;
  TENO6StencilSelection(q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3, cutoff,
                        dl0, dl1, dl2, dl3);
  Real dr0, dr1, dr2, dr3;
  TENO6StencilSelection(q_ip3, q_ip2, q_ip1, q_i, q_im1, q_im2, cutoff,
                        dr0, dr1, dr2, dr3);

  const bool left_smooth = (dl0 == 1.0 && dl1 == 1.0 && dl2 == 1.0 && dl3 == 1.0);
  const bool right_smooth = (dr0 == 1.0 && dr1 == 1.0 && dr2 == 1.0 && dr3 == 1.0);
  if (left_smooth && right_smooth) {
    if constexpr (optimized) {
      // Table 5 weights: fifth-order upwind-biased operator with eta=0.54.
      ql = (9.0/500.0)*q_im2 - (7.0/50.0)*q_im1 + (63.0/100.0)*q_i
          + (181.0/300.0)*q_ip1 - (19.0/150.0)*q_ip2 + (23.0/1500.0)*q_ip3;
      qr = (9.0/500.0)*q_ip3 - (7.0/50.0)*q_ip2 + (63.0/100.0)*q_ip1
          + (181.0/300.0)*q_i - (19.0/150.0)*q_im1 + (23.0/1500.0)*q_im2;
    } else {
      // The standard all-smooth operator is sixth-order central, so both states are
      // deliberately assigned from the same arithmetic result. Pair coefficients
      // exchanged by reflection so the central sum is bitwise symmetric as well.
      const Real outer_pair = q_im2 + q_ip3;
      const Real near_pair = q_im1 + q_ip2;
      const Real center_pair = q_i + q_ip1;
      const Real near_term = 8.0*near_pair;
      const Real center_term = 37.0*center_pair;
      const Real central = ((outer_pair - near_term) + center_term)/60.0;
      ql = central;
      qr = central;
    }
    return;
  }

  ql = TENO6ReconstructSide<optimized>(q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3,
                                        dl0, dl1, dl2, dl3);
  qr = TENO6ReconstructSide<optimized>(q_ip3, q_ip2, q_ip1, q_i, q_im1, q_im2,
                                        dr0, dr1, dr2, dr3);
}

//----------------------------------------------------------------------------------------
//! \brief Face-oriented TENO6 wrapper in the x1 direction.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO6X1(TeamMember_t const &member, const EOS_Data &eos, const Real cutoff,
     const bool apply_floors, const int m, const int k, const int j,
     const int fl, const int fu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  const int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  const Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, fl, fu, [&](const int f) {
      TENO6<optimized>(q(m,n,k,j,f-3), q(m,n,k,j,f-2), q(m,n,k,j,f-1),
                       q(m,n,k,j,f), q(m,n,k,j,f+1), q(m,n,k,j,f+2), cutoff,
                       ql(n,f), qr(n,f));
      if (apply_floors) {
        if (n == IDN) {
          ql(IDN,f) = fmax(ql(IDN,f), dfloor_);
          qr(IDN,f) = fmax(qr(IDN,f), dfloor_);
        }
        if (n == IEN) {
          ql(IEN,f) = fmax(ql(IEN,f), efloor_);
          qr(IEN,f) = fmax(qr(IEN,f), efloor_);
        }
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \brief Face-oriented TENO6 wrapper in the x2 direction.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO6X2(TeamMember_t const &member, const EOS_Data &eos, const Real cutoff,
     const bool apply_floors, const int m, const int k, const int f,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  const int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  const Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      TENO6<optimized>(q(m,n,k,f-3,i), q(m,n,k,f-2,i), q(m,n,k,f-1,i),
                       q(m,n,k,f,i), q(m,n,k,f+1,i), q(m,n,k,f+2,i), cutoff,
                       ql(n,i), qr(n,i));
      if (apply_floors) {
        if (n == IDN) {
          ql(IDN,i) = fmax(ql(IDN,i), dfloor_);
          qr(IDN,i) = fmax(qr(IDN,i), dfloor_);
        }
        if (n == IEN) {
          ql(IEN,i) = fmax(ql(IEN,i), efloor_);
          qr(IEN,i) = fmax(qr(IEN,i), efloor_);
        }
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \brief Face-oriented TENO6 wrapper in the x3 direction.

template <bool optimized>
KOKKOS_INLINE_FUNCTION
void TENO6X3(TeamMember_t const &member, const EOS_Data &eos, const Real cutoff,
     const bool apply_floors, const int m, const int f, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  const int nvar = q.extent_int(1);
  const Real &dfloor_ = eos.dfloor;
  // TODO(jmstone): ideal gas only for now
  const Real efloor_ = eos.pfloor/(eos.gamma - 1.0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      TENO6<optimized>(q(m,n,f-3,j,i), q(m,n,f-2,j,i), q(m,n,f-1,j,i),
                       q(m,n,f,j,i), q(m,n,f+1,j,i), q(m,n,f+2,j,i), cutoff,
                       ql(n,i), qr(n,i));
      if (apply_floors) {
        if (n == IDN) {
          ql(IDN,i) = fmax(ql(IDN,i), dfloor_);
          qr(IDN,i) = fmax(qr(IDN,i), dfloor_);
        }
        if (n == IEN) {
          ql(IEN,i) = fmax(ql(IEN,i), efloor_);
          qr(IEN,i) = fmax(qr(IEN,i), efloor_);
        }
      }
    });
  }
}

#endif // RECONSTRUCT_TENO6_HPP_
