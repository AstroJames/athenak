#ifndef MHD_DUAL_CT_HPP_
#define MHD_DUAL_CT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dual_ct.hpp
//! \brief Compatible face-divergence and edge-curl operators for dual CT.

#include "athena.hpp"

namespace srrmhd {

//----------------------------------------------------------------------------------------
//! \brief Relativistic scalar-Ohm current in the Eulerian frame.

KOKKOS_INLINE_FUNCTION
void OhmicCurrent(const Real e1, const Real e2, const Real e3, const Real b1,
                  const Real b2, const Real b3, const Real u1, const Real u2,
                  const Real u3, const Real q, const Real eta, Real &j1,
                  Real &j2, Real &j3) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real v1 = u1/lor;
  const Real v2 = u2/lor;
  const Real v3 = u3/lor;
  const Real edotv = e1*v1 + e2*v2 + e3*v3;
  const Real sigma_lor = lor/eta;
  j1 = q*v1 + sigma_lor*(e1 + v2*b3 - v3*b2 - edotv*v1);
  j2 = q*v2 + sigma_lor*(e2 + v3*b1 - v1*b3 - edotv*v2);
  j3 = q*v3 + sigma_lor*(e3 + v1*b2 - v2*b1 - edotv*v3);
}

//----------------------------------------------------------------------------------------
//! \brief Discrete face-to-cell divergence used to define charge q = div(E).

KOKKOS_INLINE_FUNCTION
Real FaceDivergence(const DvceFaceFld4D<Real> &f, const int m, const int k,
                    const int j, const int i, const Real idx1, const Real idx2,
                    const Real idx3, const bool multi_d, const bool three_d) {
  Real divergence = (f.x1f(m,k,j,i+1) - f.x1f(m,k,j,i))*idx1;
  if (multi_d) {
    divergence += (f.x2f(m,k,j+1,i) - f.x2f(m,k,j,i))*idx2;
  }
  if (three_d) {
    divergence += (f.x3f(m,k+1,j,i) - f.x3f(m,k,j,i))*idx3;
  }
  return divergence;
}

//----------------------------------------------------------------------------------------
//! \brief First component of curl(B_edge), collocated with an x1 face.

KOKKOS_INLINE_FUNCTION
Real EdgeCurl1(const DvceEdgeFld4D<Real> &b, const int m, const int k,
               const int j, const int i, const Real idx2, const Real idx3,
               const bool multi_d, const bool three_d) {
  Real curl = 0.0;
  if (multi_d) {
    curl += (b.x3e(m,k,j+1,i) - b.x3e(m,k,j,i))*idx2;
  }
  if (three_d) {
    curl -= (b.x2e(m,k+1,j,i) - b.x2e(m,k,j,i))*idx3;
  }
  return curl;
}

//----------------------------------------------------------------------------------------
//! \brief Second component of curl(B_edge), collocated with an x2 face.

KOKKOS_INLINE_FUNCTION
Real EdgeCurl2(const DvceEdgeFld4D<Real> &b, const int m, const int k,
               const int j, const int i, const Real idx1, const Real idx3,
               const bool three_d) {
  Real curl = -(b.x3e(m,k,j,i+1) - b.x3e(m,k,j,i))*idx1;
  if (three_d) {
    curl += (b.x1e(m,k+1,j,i) - b.x1e(m,k,j,i))*idx3;
  }
  return curl;
}

//----------------------------------------------------------------------------------------
//! \brief Third component of curl(B_edge), collocated with an x3 face.

KOKKOS_INLINE_FUNCTION
Real EdgeCurl3(const DvceEdgeFld4D<Real> &b, const int m, const int k,
               const int j, const int i, const Real idx1, const Real idx2,
               const bool multi_d) {
  Real curl = (b.x2e(m,k,j,i+1) - b.x2e(m,k,j,i))*idx1;
  if (multi_d) {
    curl -= (b.x1e(m,k,j+1,i) - b.x1e(m,k,j,i))*idx2;
  }
  return curl;
}

//----------------------------------------------------------------------------------------
//! \brief Average primary face-centered E to the cell center for fluid coupling.

KOKKOS_INLINE_FUNCTION
void ElectricFaceToCell(const DvceFaceFld4D<Real> &e, const int m, const int k,
                        const int j, const int i, Real &e1, Real &e2, Real &e3) {
  e1 = 0.5*(e.x1f(m,k,j,i) + e.x1f(m,k,j,i+1));
  e2 = 0.5*(e.x2f(m,k,j,i) + e.x2f(m,k,j+1,i));
  e3 = 0.5*(e.x3f(m,k,j,i) + e.x3f(m,k+1,j,i));
}

} // namespace srrmhd

#endif // MHD_DUAL_CT_HPP_
