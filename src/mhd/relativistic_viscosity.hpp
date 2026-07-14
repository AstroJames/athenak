#ifndef MHD_RELATIVISTIC_VISCOSITY_HPP_
#define MHD_RELATIVISTIC_VISCOSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file relativistic_viscosity.hpp
//! \brief Device-callable algebra for isotropic Israel--Stewart shear stress.

#include "athena.hpp"
#include "eos/resistive_srmhd.hpp"

namespace srrmhd {

constexpr int NVISC = 6;
enum ViscousStressIndex {IVP11=0, IVP22=1, IVP33=2, IVP12=3, IVP13=4, IVP23=5};

// The six spatial components are stored in the global Cartesian basis.  Projection
// removes one redundant trace component, leaving the five physical shear degrees of
// freedom.  Temporal components are reconstructed from the fluid velocity.
struct ShearStress {
  Real p11 = 0.0;
  Real p22 = 0.0;
  Real p33 = 0.0;
  Real p12 = 0.0;
  Real p13 = 0.0;
  Real p23 = 0.0;
};

struct ShearTemporal {
  Real p00 = 0.0;
  Real p01 = 0.0;
  Real p02 = 0.0;
  Real p03 = 0.0;
};

struct FourVector {
  Real v[4] = {0.0, 0.0, 0.0, 0.0};
};

// du[mu][nu] = partial_mu u^nu.  The time component in every derivative row is
// fixed by differentiating u_mu u^mu=-1.
struct FourVelocityGradient {
  Real du[4][4] = {};
};

struct SymmetricTensor4 {
  Real t[4][4] = {};
};

struct RelativisticViscosityData {
  bool enabled = false;
  bool linearized_target_1d = false;
  Real nu = 0.0;       // kinematic shear viscosity, eta_sh = w*nu
  Real tau = 0.0;      // proper Israel--Stewart relaxation time
  Real chi_max = 2.0;  // maximum sqrt(pi_mn pi^mn)/(energy density + pressure)
};

struct ShearGradient1D {
  Real pressure = 0.0;
  Real u1 = 0.0;
  Real u2 = 0.0;
  Real u3 = 0.0;
  ShearStress pi;
};

// Spatial derivative index first, Cartesian component index second:
// du[d][i] = partial_(x^d) u^(i+1), dpi[d] = partial_(x^d) pi^(ij).
struct ShearGradient3D {
  Real pressure[3] = {};
  Real du[3][3] = {};
  ShearStress dpi[3];
};

//----------------------------------------------------------------------------------------
//! \brief Convert the compact symmetric spatial representation to a 3x3 matrix.

KOKKOS_INLINE_FUNCTION
void ShearStressMatrix(const ShearStress &pi, Real matrix[3][3]) {
  matrix[0][0] = pi.p11;
  matrix[0][1] = matrix[1][0] = pi.p12;
  matrix[0][2] = matrix[2][0] = pi.p13;
  matrix[1][1] = pi.p22;
  matrix[1][2] = matrix[2][1] = pi.p23;
  matrix[2][2] = pi.p33;
}

//----------------------------------------------------------------------------------------
//! \brief Convert a symmetric 3x3 matrix to the compact spatial representation.

KOKKOS_INLINE_FUNCTION
ShearStress MatrixShearStress(const Real matrix[3][3]) {
  ShearStress pi;
  pi.p11 = matrix[0][0];
  pi.p22 = matrix[1][1];
  pi.p33 = matrix[2][2];
  pi.p12 = matrix[0][1];
  pi.p13 = matrix[0][2];
  pi.p23 = matrix[1][2];
  return pi;
}

//----------------------------------------------------------------------------------------
//! \brief Convert between primitive pi^(ij) and conservative P^(ij)=D pi^(ij).

KOKKOS_INLINE_FUNCTION
ShearStress ConservativeShearStress(const Real d, const ShearStress &pi) {
  ShearStress p = pi;
  p.p11 *= d;
  p.p22 *= d;
  p.p33 *= d;
  p.p12 *= d;
  p.p13 *= d;
  p.p23 *= d;
  return p;
}

KOKKOS_INLINE_FUNCTION
ShearStress PrimitiveShearStress(const Real d, const ShearStress &p) {
  const Real inv_d = 1.0/d;
  ShearStress pi = p;
  pi.p11 *= inv_d;
  pi.p22 *= inv_d;
  pi.p33 *= inv_d;
  pi.p12 *= inv_d;
  pi.p13 *= inv_d;
  pi.p23 *= inv_d;
  return pi;
}

//----------------------------------------------------------------------------------------
//! \brief Reconstruct pi^(0 mu) from spatial pi^(ij) and AthenaK spatial four-velocity.

KOKKOS_INLINE_FUNCTION
ShearTemporal ShearTemporalComponents(const Real u1, const Real u2, const Real u3,
                                      const ShearStress &pi) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real v1 = u1/lor;
  const Real v2 = u2/lor;
  const Real v3 = u3/lor;

  ShearTemporal t;
  t.p01 = v1*pi.p11 + v2*pi.p12 + v3*pi.p13;
  t.p02 = v1*pi.p12 + v2*pi.p22 + v3*pi.p23;
  t.p03 = v1*pi.p13 + v2*pi.p23 + v3*pi.p33;
  t.p00 = v1*t.p01 + v2*t.p02 + v3*t.p03;
  return t;
}

//----------------------------------------------------------------------------------------
//! \brief Assemble the full orthogonal spacetime tensor represented by spatial pi^ij.

KOKKOS_INLINE_FUNCTION
SymmetricTensor4 AssembleSpacetimeShear(const Real u1, const Real u2, const Real u3,
                                        const ShearStress &pi) {
  const ShearTemporal temporal = ShearTemporalComponents(u1, u2, u3, pi);
  SymmetricTensor4 tensor;
  tensor.t[0][0] = temporal.p00;
  tensor.t[0][1] = tensor.t[1][0] = temporal.p01;
  tensor.t[0][2] = tensor.t[2][0] = temporal.p02;
  tensor.t[0][3] = tensor.t[3][0] = temporal.p03;
  tensor.t[1][1] = pi.p11;
  tensor.t[2][2] = pi.p22;
  tensor.t[3][3] = pi.p33;
  tensor.t[1][2] = tensor.t[2][1] = pi.p12;
  tensor.t[1][3] = tensor.t[3][1] = pi.p13;
  tensor.t[2][3] = tensor.t[3][2] = pi.p23;
  return tensor;
}

//----------------------------------------------------------------------------------------
//! \brief Lorentz-invariant magnitude sqrt(pi^(mu nu) pi_(mu nu)).

KOKKOS_INLINE_FUNCTION
Real ShearInvariantNorm(const Real u1, const Real u2, const Real u3,
                        const ShearStress &pi) {
  const ShearTemporal t = ShearTemporalComponents(u1, u2, u3, pi);
  const Real norm_squared = SQR(t.p00)
      - 2.0*(SQR(t.p01) + SQR(t.p02) + SQR(t.p03))
      + SQR(pi.p11) + SQR(pi.p22) + SQR(pi.p33)
      + 2.0*(SQR(pi.p12) + SQR(pi.p13) + SQR(pi.p23));
  return sqrt(fmax(0.0, norm_squared));
}

//----------------------------------------------------------------------------------------
//! \brief Uniformly limit shear to chi=sqrt(pi^2)/(energy density + pressure).
//!
//! Uniform rescaling preserves symmetry, orthogonality, and tracelessness.  The caller
//! supplies the positive rest-frame enthalpy density (energy density plus pressure).

KOKKOS_INLINE_FUNCTION
ShearStress LimitShearInverseReynolds(const Real u1, const Real u2, const Real u3,
                                      const Real enthalpy_density,
                                      const Real chi_max, const ShearStress &pi) {
  const Real norm = ShearInvariantNorm(u1, u2, u3, pi);
  Real scale = 1.0;
  if (!(enthalpy_density > 0.0) || !(chi_max > 0.0) || !isfinite(norm)) {
    scale = 0.0;
  } else if (norm > chi_max*enthalpy_density) {
    scale = chi_max*enthalpy_density/norm;
  }
  ShearStress limited = pi;
  limited.p11 *= scale;
  limited.p22 *= scale;
  limited.p33 *= scale;
  limited.p12 *= scale;
  limited.p13 *= scale;
  limited.p23 *= scale;
  return limited;
}

//----------------------------------------------------------------------------------------
//! \brief Extract the six stored spatial components of a symmetric spacetime tensor.

KOKKOS_INLINE_FUNCTION
ShearStress SpatialShearComponents(const SymmetricTensor4 &tensor) {
  ShearStress pi;
  pi.p11 = tensor.t[1][1];
  pi.p22 = tensor.t[2][2];
  pi.p33 = tensor.t[3][3];
  pi.p12 = tensor.t[1][2];
  pi.p13 = tensor.t[1][3];
  pi.p23 = tensor.t[2][3];
  return pi;
}

//----------------------------------------------------------------------------------------
//! \brief Complete partial_mu u^0 so every gradient row preserves u_mu u^mu=-1.

KOKKOS_INLINE_FUNCTION
void CompleteFourVelocityGradient(const Real u1, const Real u2, const Real u3,
                                  FourVelocityGradient &gradient) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  for (int mu = 0; mu < 4; ++mu) {
    gradient.du[mu][0] = (u1*gradient.du[mu][1] + u2*gradient.du[mu][2]
                           + u3*gradient.du[mu][3])/lor;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Compute a^nu = u^mu partial_mu u^nu from a normalized four-gradient.

KOKKOS_INLINE_FUNCTION
FourVector FluidFourAcceleration(const Real u1, const Real u2, const Real u3,
                                 const FourVelocityGradient &gradient) {
  const Real u[4] = {sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3)), u1, u2, u3};
  FourVector acceleration;
  for (int nu = 0; nu < 4; ++nu) {
    for (int mu = 0; mu < 4; ++mu) {
      acceleration.v[nu] += u[mu]*gradient.du[mu][nu];
    }
  }
  return acceleration;
}

//----------------------------------------------------------------------------------------
//! \brief Cartesian reduction of (w Delta^alpha_lambda+pi^alpha_lambda) a^lambda.
//!
//! Acceleration orthogonality gives a^0=v_l a^l.  For a spatial equation alpha=i,
//! Delta^i_lambda a^lambda=a^i and
//!   pi^i_lambda a^lambda=pi^ij (delta_jl-v_j v_l) a^l.
//! The resulting local 3x3 matrix is exact; no ideal-fluid or lagged acceleration is
//! used in this algebraic reduction.

KOKKOS_INLINE_FUNCTION
void SpatialShearAccelerationMatrix(const Real u1, const Real u2, const Real u3,
                                    const Real enthalpy_density,
                                    const ShearStress &pi, Real matrix[3][3]) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real v[3] = {u1/lor, u2/lor, u3/lor};
  const Real spatial_pi[3][3] = {
      {pi.p11, pi.p12, pi.p13},
      {pi.p12, pi.p22, pi.p23},
      {pi.p13, pi.p23, pi.p33}};
  for (int i = 0; i < 3; ++i) {
    for (int l = 0; l < 3; ++l) {
      matrix[i][l] = (i == l) ? enthalpy_density : 0.0;
      for (int j = 0; j < 3; ++j) {
        const Real rest_space_covector = ((j == l) ? 1.0 : 0.0) - v[j]*v[l];
        matrix[i][l] += spatial_pi[i][j]*rest_space_covector;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Solve the exact spatial acceleration mass matrix and reconstruct a^0.

KOKKOS_INLINE_FUNCTION
bool SolveShearAccelerationMassMatrix(const Real u1, const Real u2, const Real u3,
                                      const Real enthalpy_density,
                                      const ShearStress &pi, const Real rhs[3],
                                      FourVector &acceleration) {
  if (!(enthalpy_density > 0.0) || !isfinite(enthalpy_density)) return false;
  Real matrix[3][3];
  SpatialShearAccelerationMatrix(u1, u2, u3, enthalpy_density, pi, matrix);
  Real local_rhs[3] = {rhs[0], rhs[1], rhs[2]};
  Real spatial_acceleration[3] = {0.0, 0.0, 0.0};
  if (!SolveLinear3x3(matrix, local_rhs, spatial_acceleration)) return false;
  for (int i = 0; i < 3; ++i) {
    if (!isfinite(spatial_acceleration[i])) return false;
    acceleration.v[i+1] = spatial_acceleration[i];
  }
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  acceleration.v[0] = (u1*acceleration.v[1] + u2*acceleration.v[2]
                         + u3*acceleration.v[3])/lor;
  return isfinite(acceleration.v[0]);
}

//----------------------------------------------------------------------------------------
//! \brief Lorentz-invariant contraction of two orthogonal symmetric tensors.

KOKKOS_INLINE_FUNCTION
Real ShearTensorInnerProduct(const Real u1, const Real u2, const Real u3,
                             const ShearStress &a, const ShearStress &b) {
  const ShearTemporal at = ShearTemporalComponents(u1, u2, u3, a);
  const ShearTemporal bt = ShearTemporalComponents(u1, u2, u3, b);
  return at.p00*bt.p00 - 2.0*(at.p01*bt.p01 + at.p02*bt.p02 + at.p03*bt.p03)
       + a.p11*b.p11 + a.p22*b.p22 + a.p33*b.p33
       + 2.0*(a.p12*b.p12 + a.p13*b.p13 + a.p23*b.p23);
}

//----------------------------------------------------------------------------------------
//! \brief Spatial sigma^(ij) and theta from 3D gradients and four-acceleration.

KOKKOS_INLINE_FUNCTION
ShearStress SpatialVelocityShear(const Real u1, const Real u2, const Real u3,
                                 const ShearGradient3D &gradient,
                                 const FourVector &acceleration, Real &expansion) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real u[3] = {u1, u2, u3};
  Real dlor[3] = {};
  Real dv[3][3] = {};
  for (int d = 0; d < 3; ++d) {
    for (int i = 0; i < 3; ++i) dlor[d] += u[i]*gradient.du[d][i]/lor;
    for (int i = 0; i < 3; ++i) {
      dv[d][i] = gradient.du[d][i]/lor - u[i]*dlor[d]/SQR(lor);
    }
  }
  expansion = acceleration.v[0]/lor;
  for (int d = 0; d < 3; ++d) expansion += lor*dv[d][d];

  Real sigma_matrix[3][3] = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const Real projector = ((i == j) ? 1.0 : 0.0) + u[i]*u[j];
      sigma_matrix[i][j] = 0.5*(gradient.du[i][j] + gradient.du[j][i]
          + u[i]*acceleration.v[j+1] + u[j]*acceleration.v[i+1])
          - projector*expansion/3.0;
    }
  }
  return MatrixShearStress(sigma_matrix);
}

//----------------------------------------------------------------------------------------
//! \brief Embed a one-dimensional gradient in the dimension-independent representation.

KOKKOS_INLINE_FUNCTION
ShearGradient3D EmbedShearGradient1D(const ShearGradient1D &gradient) {
  ShearGradient3D embedded;
  embedded.pressure[0] = gradient.pressure;
  embedded.du[0][0] = gradient.u1;
  embedded.du[0][1] = gradient.u2;
  embedded.du[0][2] = gradient.u3;
  embedded.dpi[0] = gradient.pi;
  return embedded;
}

KOKKOS_INLINE_FUNCTION
ShearStress SpatialVelocityShear1D(const Real u1, const Real u2, const Real u3,
                                   const ShearGradient1D &gradient,
                                   const FourVector &acceleration, Real &expansion) {
  return SpatialVelocityShear(
      u1, u2, u3, EmbedShearGradient1D(gradient), acceleration, expansion);
}

//----------------------------------------------------------------------------------------
//! \brief A^mu=pi^(mu lambda) a_lambda reconstructed from spatial quantities.

KOKKOS_INLINE_FUNCTION
FourVector ShearAccelerationContraction(const Real u1, const Real u2, const Real u3,
                                        const ShearStress &pi,
                                        const FourVector &acceleration) {
  const SymmetricTensor4 pi4 = AssembleSpacetimeShear(u1, u2, u3, pi);
  const Real a_lower[4] = {-acceleration.v[0], acceleration.v[1],
                            acceleration.v[2], acceleration.v[3]};
  FourVector contraction;
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      contraction.v[mu] += pi4.t[mu][nu]*a_lower[nu];
    }
  }
  return contraction;
}

//----------------------------------------------------------------------------------------
//! \brief Exact 3D gamma-law momentum residual after eliminating time derivatives.
//!
//! Spatial gradients are Eulerian centered/reconstructed gradients at one stage.
//! For a trial acceleration, the gamma-law energy equation supplies dot(p), the
//! Israel--Stewart equation supplies dot(pi^ij), and orthogonality supplies temporal
//! stress derivatives.  The returned residual is affine in the three spatial
//! acceleration components.

KOKKOS_INLINE_FUNCTION
bool ShearAccelerationResidual(
    const Real u1, const Real u2, const Real u3, const Real pressure,
    const Real enthalpy_density, const Real gamma, const Real dynamic_viscosity,
    const Real tau, const ShearStress &pi, const ShearGradient3D &gradient,
    const FourVector &four_force, const FourVector &acceleration,
    Real residual[3], ShearStress &sigma) {
  if (!(pressure > 0.0) || !(enthalpy_density > 0.0) || !(gamma > 1.0)
      || !(tau > 0.0)) return false;
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real u[4] = {lor, u1, u2, u3};
  const Real v[3] = {u1/lor, u2/lor, u3/lor};
  Real dlor[3] = {};
  Real dv[3][3] = {};
  for (int d = 0; d < 3; ++d) {
    for (int i = 0; i < 3; ++i) dlor[d] += u[i+1]*gradient.du[d][i]/lor;
    for (int i = 0; i < 3; ++i) {
      dv[d][i] = gradient.du[d][i]/lor - u[i+1]*dlor[d]/SQR(lor);
    }
  }

  Real expansion = 0.0;
  sigma = SpatialVelocityShear(u1, u2, u3, gradient, acceleration, expansion);
  const FourVector contraction =
      ShearAccelerationContraction(u1, u2, u3, pi, acceleration);
  ShearStress projected_rhs;
  projected_rhs.p11 = (-pi.p11 - 2.0*dynamic_viscosity*sigma.p11)/tau;
  projected_rhs.p22 = (-pi.p22 - 2.0*dynamic_viscosity*sigma.p22)/tau;
  projected_rhs.p33 = (-pi.p33 - 2.0*dynamic_viscosity*sigma.p33)/tau;
  projected_rhs.p12 = (-pi.p12 - 2.0*dynamic_viscosity*sigma.p12)/tau;
  projected_rhs.p13 = (-pi.p13 - 2.0*dynamic_viscosity*sigma.p13)/tau;
  projected_rhs.p23 = (-pi.p23 - 2.0*dynamic_viscosity*sigma.p23)/tau;
  ShearStress dot_pi;
  dot_pi.p11 = projected_rhs.p11 + 2.0*u1*contraction.v[1];
  dot_pi.p22 = projected_rhs.p22 + 2.0*u2*contraction.v[2];
  dot_pi.p33 = projected_rhs.p33 + 2.0*u3*contraction.v[3];
  dot_pi.p12 = projected_rhs.p12 + u1*contraction.v[2] + u2*contraction.v[1];
  dot_pi.p13 = projected_rhs.p13 + u1*contraction.v[3] + u3*contraction.v[1];
  dot_pi.p23 = projected_rhs.p23 + u2*contraction.v[3] + u3*contraction.v[2];

  const ShearTemporal temporal = ShearTemporalComponents(u1, u2, u3, pi);
  Real spatial_pi[3][3];
  Real spatial_dot_pi[3][3];
  Real spatial_dpi[3][3][3];
  ShearStressMatrix(pi, spatial_pi);
  ShearStressMatrix(dot_pi, spatial_dot_pi);
  for (int d = 0; d < 3; ++d) ShearStressMatrix(gradient.dpi[d], spatial_dpi[d]);

  Real dt_v[3];
  for (int i = 0; i < 3; ++i) {
    dt_v[i] = (acceleration.v[i+1] - v[i]*acceleration.v[0])/SQR(lor);
    for (int d = 0; d < 3; ++d) dt_v[i] -= v[d]*dv[d][i];
  }
  Real dt_pi[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      dt_pi[i][j] = spatial_dot_pi[i][j]/lor;
      for (int d = 0; d < 3; ++d) dt_pi[i][j] -= v[d]*spatial_dpi[d][i][j];
    }
  }
  Real dt_p0i[3] = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      dt_p0i[i] += dt_v[j]*spatial_pi[j][i] + v[j]*dt_pi[j][i];
    }
  }
  Real dt_p00 = 0.0;
  for (int i = 0; i < 3; ++i) {
    const Real p0i = (i == 0) ? temporal.p01 : ((i == 1) ? temporal.p02 : temporal.p03);
    dt_p00 += dt_v[i]*p0i + v[i]*dt_p0i[i];
  }
  Real div_pi[4] = {dt_p00, dt_p0i[0], dt_p0i[1], dt_p0i[2]};
  for (int d = 0; d < 3; ++d) {
    for (int j = 0; j < 3; ++j) {
      div_pi[0] += dv[d][j]*spatial_pi[d][j] + v[j]*spatial_dpi[d][d][j];
    }
    for (int i = 0; i < 3; ++i) div_pi[i+1] += spatial_dpi[d][d][i];
  }

  const Real pi_sigma = ShearTensorInnerProduct(u1, u2, u3, pi, sigma);
  const Real u_dot_force = -lor*four_force.v[0] + u1*four_force.v[1]
                         + u2*four_force.v[2] + u3*four_force.v[3];
  const Real dot_pressure = -gamma*pressure*expansion
                          - (gamma - 1.0)*(pi_sigma + u_dot_force);
  Real dt_pressure = dot_pressure/lor;
  for (int d = 0; d < 3; ++d) dt_pressure -= v[d]*gradient.pressure[d];
  const Real pressure_gradient[4] = {
      -dt_pressure, gradient.pressure[0], gradient.pressure[1], gradient.pressure[2]};

  Real force_balance[4];
  for (int mu = 0; mu < 4; ++mu) {
    force_balance[mu] = pressure_gradient[mu] + div_pi[mu] - four_force.v[mu];
  }
  const Real u_dot_balance = -lor*force_balance[0] + u1*force_balance[1]
                           + u2*force_balance[2] + u3*force_balance[3];
  for (int i = 0; i < 3; ++i) {
    residual[i] = enthalpy_density*acceleration.v[i+1]
                + force_balance[i+1] + u[i+1]*u_dot_balance;
  }
  return isfinite(residual[0]) && isfinite(residual[1]) && isfinite(residual[2]);
}

//----------------------------------------------------------------------------------------
//! \brief Solve the full affine 3D acceleration operator for a gamma-law fluid.

KOKKOS_INLINE_FUNCTION
bool SolveShearAcceleration(
    const Real u1, const Real u2, const Real u3, const Real pressure,
    const Real enthalpy_density, const Real gamma, const Real dynamic_viscosity,
    const Real tau, const ShearStress &pi, const ShearGradient3D &gradient,
    const FourVector &four_force, FourVector &acceleration, ShearStress &sigma) {
  FourVector zero_acceleration;
  Real constant[3];
  ShearStress scratch_sigma;
  if (!ShearAccelerationResidual(
          u1, u2, u3, pressure, enthalpy_density, gamma, dynamic_viscosity, tau,
          pi, gradient, four_force, zero_acceleration, constant, scratch_sigma)) {
    return false;
  }
  Real matrix[3][3];
  for (int col = 0; col < 3; ++col) {
    FourVector basis;
    basis.v[col+1] = 1.0;
    const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
    basis.v[0] = (u1*basis.v[1] + u2*basis.v[2] + u3*basis.v[3])/lor;
    Real value[3];
    if (!ShearAccelerationResidual(
            u1, u2, u3, pressure, enthalpy_density, gamma, dynamic_viscosity,
            tau, pi, gradient, four_force, basis, value, scratch_sigma)) return false;
    for (int row = 0; row < 3; ++row) matrix[row][col] = value[row] - constant[row];
  }
  Real rhs[3] = {-constant[0], -constant[1], -constant[2]};
  Real spatial_acceleration[3];
  if (!SolveLinear3x3(matrix, rhs, spatial_acceleration)) return false;
  acceleration.v[1] = spatial_acceleration[0];
  acceleration.v[2] = spatial_acceleration[1];
  acceleration.v[3] = spatial_acceleration[2];
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  acceleration.v[0] = (u1*acceleration.v[1] + u2*acceleration.v[2]
                         + u3*acceleration.v[3])/lor;
  Real residual[3];
  return ShearAccelerationResidual(
      u1, u2, u3, pressure, enthalpy_density, gamma, dynamic_viscosity, tau,
      pi, gradient, four_force, acceleration, residual, sigma);
}

//----------------------------------------------------------------------------------------
//! \brief Compatibility wrappers for one-dimensional reference tests.

KOKKOS_INLINE_FUNCTION
bool ShearAccelerationResidual1D(
    const Real u1, const Real u2, const Real u3, const Real pressure,
    const Real enthalpy_density, const Real gamma, const Real dynamic_viscosity,
    const Real tau, const ShearStress &pi, const ShearGradient1D &gradient,
    const FourVector &four_force, const FourVector &acceleration,
    Real residual[3], ShearStress &sigma) {
  return ShearAccelerationResidual(
      u1, u2, u3, pressure, enthalpy_density, gamma, dynamic_viscosity, tau,
      pi, EmbedShearGradient1D(gradient), four_force, acceleration, residual, sigma);
}

KOKKOS_INLINE_FUNCTION
bool SolveShearAcceleration1D(
    const Real u1, const Real u2, const Real u3, const Real pressure,
    const Real enthalpy_density, const Real gamma, const Real dynamic_viscosity,
    const Real tau, const ShearStress &pi, const ShearGradient1D &gradient,
    const FourVector &four_force, FourVector &acceleration, ShearStress &sigma) {
  return SolveShearAcceleration(
      u1, u2, u3, pressure, enthalpy_density, gamma, dynamic_viscosity, tau,
      pi, EmbedShearGradient1D(gradient), four_force, acceleration, sigma);
}

//----------------------------------------------------------------------------------------
//! \brief Source per D from the spatial-only constraint-propagating acceleration term.

KOKKOS_INLINE_FUNCTION
ShearStress KinematicShearSourcePerD(const Real u1, const Real u2, const Real u3,
                                     const ShearStress &pi,
                                     const FourVector &acceleration) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real v1 = u1/lor;
  const Real v2 = u2/lor;
  const Real v3 = u3/lor;
  const FourVector a_pi =
      ShearAccelerationContraction(u1, u2, u3, pi, acceleration);
  ShearStress source;
  source.p11 = 2.0*v1*a_pi.v[1];
  source.p22 = 2.0*v2*a_pi.v[2];
  source.p33 = 2.0*v3*a_pi.v[3];
  source.p12 = v1*a_pi.v[2] + v2*a_pi.v[1];
  source.p13 = v1*a_pi.v[3] + v3*a_pi.v[1];
  source.p23 = v2*a_pi.v[3] + v3*a_pi.v[2];
  return source;
}

//----------------------------------------------------------------------------------------
//! \brief Matter four-force from charge plus the scalar relativistic Ohm current.

KOKKOS_INLINE_FUNCTION
bool ResistiveMatterFourForce(const Real charge, const Real resistivity,
                              const Real u1, const Real u2, const Real u3,
                              const Real e1, const Real e2, const Real e3,
                              const Real b1, const Real b2, const Real b3,
                              FourVector &force) {
  if (!(resistivity > 0.0) || !isfinite(resistivity)) return false;
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real v1 = u1/lor;
  const Real v2 = u2/lor;
  const Real v3 = u3/lor;
  const Real edotv = e1*v1 + e2*v2 + e3*v3;
  const Real scale = lor/resistivity;
  const Real j1 = charge*v1
      + scale*(e1 + v2*b3 - v3*b2 - edotv*v1);
  const Real j2 = charge*v2
      + scale*(e2 + v3*b1 - v1*b3 - edotv*v2);
  const Real j3 = charge*v3
      + scale*(e3 + v1*b2 - v2*b1 - edotv*v3);
  force.v[0] = j1*e1 + j2*e2 + j3*e3;
  force.v[1] = charge*e1 + j2*b3 - j3*b2;
  force.v[2] = charge*e2 + j3*b1 - j1*b3;
  force.v[3] = charge*e3 + j1*b2 - j2*b1;
  return isfinite(force.v[0]) && isfinite(force.v[1])
      && isfinite(force.v[2]) && isfinite(force.v[3]);
}

//----------------------------------------------------------------------------------------
//! \brief Compute sigma^(mu nu)=Delta^(mu nu)_(a b) partial^a u^b.

KOKKOS_INLINE_FUNCTION
SymmetricTensor4 SpacetimeVelocityShear(const Real u1, const Real u2, const Real u3,
                                        const FourVelocityGradient &gradient) {
  const Real u[4] = {sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3)), u1, u2, u3};
  Real projector[4][4];
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      Real metric = 0.0;
      if (mu == nu) metric = (mu == 0) ? -1.0 : 1.0;
      projector[mu][nu] = metric + u[mu]*u[nu];
    }
  }

  Real projected_gradient[4][4] = {};
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      for (int alpha = 0; alpha < 4; ++alpha) {
        projected_gradient[mu][nu] +=
            projector[mu][alpha]*gradient.du[alpha][nu];
      }
    }
  }

  Real expansion = 0.0;
  for (int mu = 0; mu < 4; ++mu) expansion += gradient.du[mu][mu];
  SymmetricTensor4 shear;
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      shear.t[mu][nu] = 0.5*(projected_gradient[mu][nu]
                                      + projected_gradient[nu][mu])
                         - projector[mu][nu]*expansion/3.0;
    }
  }
  return shear;
}

//----------------------------------------------------------------------------------------
//! \brief Undo the projected comoving derivative while propagating shear constraints.
//!
//! If R^(mu nu)=Delta^(mu nu)_(a b) D pi^(a b), differentiating u_mu pi^(mu nu)=0
//! gives
//!   D pi^(mu nu) = R^(mu nu) + u^mu q^nu + u^nu q^mu,
//!   q^nu = pi^(nu alpha) a_alpha.

KOKKOS_INLINE_FUNCTION
SymmetricTensor4 UnprojectedComovingShearDerivative(
    const Real u1, const Real u2, const Real u3, const ShearStress &pi,
    const ShearStress &projected_rhs, const FourVector &acceleration) {
  const Real u[4] = {sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3)), u1, u2, u3};
  const Real a_lower[4] = {-acceleration.v[0], acceleration.v[1],
                            acceleration.v[2], acceleration.v[3]};
  const SymmetricTensor4 pi4 = AssembleSpacetimeShear(u1, u2, u3, pi);
  const SymmetricTensor4 rhs4 =
      AssembleSpacetimeShear(u1, u2, u3, projected_rhs);
  Real q[4] = {0.0, 0.0, 0.0, 0.0};
  for (int nu = 0; nu < 4; ++nu) {
    for (int alpha = 0; alpha < 4; ++alpha) {
      q[nu] += pi4.t[nu][alpha]*a_lower[alpha];
    }
  }

  SymmetricTensor4 derivative;
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      derivative.t[mu][nu] = rhs4.t[mu][nu] + u[mu]*q[nu] + u[nu]*q[mu];
    }
  }
  return derivative;
}

//----------------------------------------------------------------------------------------
//! \brief Build pi_NS^(ij)=-2 eta_sh sigma^(ij), with eta_sh=w*nu.

KOKKOS_INLINE_FUNCTION
ShearStress NavierStokesShearTarget(const Real enthalpy_density, const Real nu,
                                    const SymmetricTensor4 &shear) {
  const Real coefficient = -2.0*enthalpy_density*nu;
  ShearStress target = SpatialShearComponents(shear);
  target.p11 *= coefficient;
  target.p22 *= coefficient;
  target.p33 *= coefficient;
  target.p12 *= coefficient;
  target.p13 *= coefficient;
  target.p23 *= coefficient;
  return target;
}

//----------------------------------------------------------------------------------------
//! \brief Project an arbitrary symmetric spatial tensor into the rest-space STF sector.
//!
//! A spatial tensor in this storage representation already defines an orthogonal
//! spacetime tensor through ShearTemporalComponents.  Its covariant trace is
//!   C = (delta_ij - v_i v_j) A^ij.
//! Removing C Delta^(mu nu)/3 gives the spatial projection
//!   pi^ij = A^ij - (delta^ij + u^i u^j) C/3.
//! This map is idempotent and its reconstructed spacetime tensor satisfies both
//! u_mu pi^(mu nu)=0 and pi^mu_mu=0.

KOKKOS_INLINE_FUNCTION
ShearStress ProjectShearStress(const Real u1, const Real u2, const Real u3,
                               const ShearStress &a) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const Real v1 = u1/lor;
  const Real v2 = u2/lor;
  const Real v3 = u3/lor;
  const Real vav = SQR(v1)*a.p11 + SQR(v2)*a.p22 + SQR(v3)*a.p33
                 + 2.0*(v1*v2*a.p12 + v1*v3*a.p13 + v2*v3*a.p23);
  const Real one_third_trace = (a.p11 + a.p22 + a.p33 - vav)/3.0;

  ShearStress pi;
  pi.p11 = a.p11 - (1.0 + SQR(u1))*one_third_trace;
  pi.p22 = a.p22 - (1.0 + SQR(u2))*one_third_trace;
  pi.p33 = a.p33 - (1.0 + SQR(u3))*one_third_trace;
  pi.p12 = a.p12 - u1*u2*one_third_trace;
  pi.p13 = a.p13 - u1*u3*one_third_trace;
  pi.p23 = a.p23 - u2*u3*one_third_trace;
  return pi;
}

//----------------------------------------------------------------------------------------
//! \brief Return maximum orthogonality and absolute trace residuals.

KOKKOS_INLINE_FUNCTION
void ShearConstraintErrors(const Real u1, const Real u2, const Real u3,
                           const ShearStress &pi, Real &orthogonality_error,
                           Real &trace_error) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const ShearTemporal t = ShearTemporalComponents(u1, u2, u3, pi);
  const Real r0 = -lor*t.p00 + u1*t.p01 + u2*t.p02 + u3*t.p03;
  const Real r1 = -lor*t.p01 + u1*pi.p11 + u2*pi.p12 + u3*pi.p13;
  const Real r2 = -lor*t.p02 + u1*pi.p12 + u2*pi.p22 + u3*pi.p23;
  const Real r3 = -lor*t.p03 + u1*pi.p13 + u2*pi.p23 + u3*pi.p33;
  orthogonality_error = fmax(fabs(r0),
      fmax(fabs(r1), fmax(fabs(r2), fabs(r3))));
  trace_error = fabs(-t.p00 + pi.p11 + pi.p22 + pi.p33);
}

//----------------------------------------------------------------------------------------
//! \brief Add or remove the viscous share of total momentum and tau.

KOKKOS_INLINE_FUNCTION
void AddShearToConserved(const Real u1, const Real u2, const Real u3,
                         const ShearStress &pi, SRRMHDCons1D &u) {
  const ShearTemporal t = ShearTemporalComponents(u1, u2, u3, pi);
  u.mx += t.p01;
  u.my += t.p02;
  u.mz += t.p03;
  u.e += t.p00;
}

KOKKOS_INLINE_FUNCTION
void RemoveShearFromConserved(const Real u1, const Real u2, const Real u3,
                              const ShearStress &pi, SRRMHDCons1D &u) {
  const ShearTemporal t = ShearTemporalComponents(u1, u2, u3, pi);
  u.mx -= t.p01;
  u.my -= t.p02;
  u.mz -= t.p03;
  u.e -= t.p00;
}

//----------------------------------------------------------------------------------------
//! \brief Unit-speed LLF fluxes of conservative shear and its energy-momentum share.

KOKKOS_INLINE_FUNCTION
void ViscousLLFContributions(const int direction,
                             const Real rho_l, const Real u1_l,
                             const Real u2_l, const Real u3_l,
                             const ShearStress &pi_l,
                             const Real rho_r, const Real u1_r,
                             const Real u2_r, const Real u3_r,
                             const ShearStress &pi_r,
                             Real stress_flux[NVISC],
                             Real momentum_flux[3], Real &energy_flux) {
  const Real lor_l = sqrt(1.0 + SQR(u1_l) + SQR(u2_l) + SQR(u3_l));
  const Real lor_r = sqrt(1.0 + SQR(u1_r) + SQR(u2_r) + SQR(u3_r));
  const Real velocity_l[3] = {u1_l/lor_l, u2_l/lor_l, u3_l/lor_l};
  const Real velocity_r[3] = {u1_r/lor_r, u2_r/lor_r, u3_r/lor_r};
  const ShearStress p_l = ConservativeShearStress(rho_l*lor_l, pi_l);
  const ShearStress p_r = ConservativeShearStress(rho_r*lor_r, pi_r);
  const Real p_left[NVISC] =
      {p_l.p11, p_l.p22, p_l.p33, p_l.p12, p_l.p13, p_l.p23};
  const Real p_right[NVISC] =
      {p_r.p11, p_r.p22, p_r.p33, p_r.p12, p_r.p13, p_r.p23};
  for (int n = 0; n < NVISC; ++n) {
    stress_flux[n] = 0.5*(velocity_l[direction]*p_left[n]
        + velocity_r[direction]*p_right[n] - (p_right[n] - p_left[n]));
  }

  Real spatial_l[3][3];
  Real spatial_r[3][3];
  ShearStressMatrix(pi_l, spatial_l);
  ShearStressMatrix(pi_r, spatial_r);
  const ShearTemporal temporal_l =
      ShearTemporalComponents(u1_l, u2_l, u3_l, pi_l);
  const ShearTemporal temporal_r =
      ShearTemporalComponents(u1_r, u2_r, u3_r, pi_r);
  const Real p0_l[3] = {temporal_l.p01, temporal_l.p02, temporal_l.p03};
  const Real p0_r[3] = {temporal_r.p01, temporal_r.p02, temporal_r.p03};
  for (int component = 0; component < 3; ++component) {
    momentum_flux[component] = 0.5*(spatial_l[direction][component]
        + spatial_r[direction][component] - (p0_r[component] - p0_l[component]));
  }
  energy_flux = 0.5*(p0_l[direction] + p0_r[direction]
                               - (temporal_r.p00 - temporal_l.p00));
}

//----------------------------------------------------------------------------------------
//! \brief Fixed-velocity backward-Euler relaxation followed by constraint projection.

KOKKOS_INLINE_FUNCTION
ShearStress ImplicitShearRelaxation(const Real u1, const Real u2, const Real u3,
                                    const ShearStress &pi_star,
                                    const ShearStress &pi_ns, const Real kappa) {
  const Real inv = 1.0/(1.0 + kappa);
  ShearStress pi;
  pi.p11 = (pi_star.p11 + kappa*pi_ns.p11)*inv;
  pi.p22 = (pi_star.p22 + kappa*pi_ns.p22)*inv;
  pi.p33 = (pi_star.p33 + kappa*pi_ns.p33)*inv;
  pi.p12 = (pi_star.p12 + kappa*pi_ns.p12)*inv;
  pi.p13 = (pi_star.p13 + kappa*pi_ns.p13)*inv;
  pi.p23 = (pi_star.p23 + kappa*pi_ns.p23)*inv;
  return ProjectShearStress(u1, u2, u3, pi);
}

//----------------------------------------------------------------------------------------
//! \brief Local proper-time relaxation of conservative P^(ij)=D pi^(ij).

KOKKOS_INLINE_FUNCTION
ShearStress ImplicitConservativeShearRelaxation(
    const Real d, const Real u1, const Real u2, const Real u3,
    const ShearStress &p_star, const ShearStress &pi_ns,
    const Real dt_over_tau, const Real enthalpy_density, const Real chi_max) {
  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  const ShearStress pi_star = PrimitiveShearStress(d, p_star);
  const ShearStress relaxed = ImplicitShearRelaxation(
      u1, u2, u3, pi_star, pi_ns, dt_over_tau/lor);
  const ShearStress pi = LimitShearInverseReynolds(
      u1, u2, u3, enthalpy_density, chi_max, relaxed);
  return ConservativeShearStress(d, pi);
}

//----------------------------------------------------------------------------------------
//! \brief Velocity residual after analytically eliminating E, pi, and gas pressure.

KOKKOS_INLINE_FUNCTION
bool ImplicitViscousPrimitiveResidual(
    const SRRMHDCons1D &u, const EOS_Data &eos, const Real ex_star,
    const Real ey_star, const Real ez_star, const Real electric_kappa,
    const ShearStress &pi_star, const ShearStress &pi_ns,
    const Real shear_dt_over_tau, const Real shear_chi_max,
    const bool fixed_spatial_shear, const Real u1, const Real u2, const Real u3,
    Real &f1, Real &f2, Real &f3, SRRMHDPrim1D &w, ShearStress &pi) {
  if (!(u.d > 0.0) || !(eos.gamma > 1.0)) return false;

  const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
  ImplicitElectricField(u1, u2, u3, ex_star, ey_star, ez_star, u.bx, u.by, u.bz,
                        electric_kappa, w.ex, w.ey, w.ez);
  if (fixed_spatial_shear) {
    pi = pi_star;
  } else {
    const Real shear_kappa = shear_dt_over_tau/lor;
    pi = ImplicitShearRelaxation(u1, u2, u3, pi_star, pi_ns, shear_kappa);
  }
  const Real gm1 = eos.gamma - 1.0;
  const Real pressure_den = eos.gamma*SQR(lor)/gm1 - 1.0;
  const Real em_energy = 0.5*(SQR(w.ex) + SQR(w.ey) + SQR(w.ez)
                               + SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  ShearTemporal temporal = ShearTemporalComponents(u1, u2, u3, pi);
  Real pressure = (u.e - em_energy - temporal.p00 + u.d - u.d*lor)/pressure_den;
  if (!(pressure > 0.0) || !isfinite(pressure)) return false;

  const Real shear_norm = ShearInvariantNorm(u1, u2, u3, pi);
  const Real enthalpy_density = u.d/lor + eos.gamma*pressure/gm1;
  if (shear_norm > shear_chi_max*enthalpy_density) {
    const Real pressure0 = (u.e - em_energy + u.d - u.d*lor)/pressure_den;
    const Real enthalpy0 = u.d/lor + eos.gamma*pressure0/gm1;
    const Real enthalpy_slope = eos.gamma*temporal.p00/(gm1*pressure_den);
    const Real scale_den = shear_norm + shear_chi_max*enthalpy_slope;
    if (!(enthalpy0 > 0.0) || !(scale_den > 0.0)) return false;
    const Real scale = fmin(1.0, fmax(0.0, shear_chi_max*enthalpy0/scale_den));
    pi.p11 *= scale;
    pi.p22 *= scale;
    pi.p33 *= scale;
    pi.p12 *= scale;
    pi.p13 *= scale;
    pi.p23 *= scale;
    temporal = ShearTemporalComponents(u1, u2, u3, pi);
    pressure = (u.e - em_energy - temporal.p00 + u.d - u.d*lor)/pressure_den;
    if (!(pressure > 0.0) || !isfinite(pressure)) return false;
  }

  const Real s1_fl = u.mx - (w.ey*u.bz - w.ez*u.by) - temporal.p01;
  const Real s2_fl = u.my - (w.ez*u.bx - w.ex*u.bz) - temporal.p02;
  const Real s3_fl = u.mz - (w.ex*u.by - w.ey*u.bx) - temporal.p03;

  w.d = u.d/lor;
  w.vx = u1;
  w.vy = u2;
  w.vz = u3;
  w.e = pressure/gm1;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;

  const Real h = 1.0 + eos.gamma*w.e/w.d;
  const Real inv_dh = 1.0/(u.d*h);
  f1 = u1 - s1_fl*inv_dh;
  f2 = u2 - s2_fl*inv_dh;
  f3 = u3 - s3_fl*inv_dh;
  return isfinite(f1) && isfinite(f2) && isfinite(f3);
}

//----------------------------------------------------------------------------------------
//! \brief Coupled local recovery with block-eliminated electric and shear relaxation.
//!
//! For every trial spatial four-velocity, E is obtained from the existing analytic
//! Ohmic inverse and pi from the fixed-velocity backward-Euler relaxation.  Their
//! temporal energy-momentum shares are then removed from the fixed total conserved
//! state.  Only the three velocity residuals enter the damped Newton solve.

KOKKOS_INLINE_FUNCTION
bool SingleC2P_IdealSRRMHDImplicitViscous(
    const SRRMHDCons1D &u, const EOS_Data &eos, const Real ex_star,
    const Real ey_star, const Real ez_star, const Real electric_kappa,
    const ShearStress &pi_star, const ShearStress &pi_ns,
    const Real shear_dt_over_tau, const Real shear_chi_max,
    const bool fixed_spatial_shear, const SRRMHDPrim1D &guess, SRRMHDPrim1D &w,
    ShearStress &pi, int &iter_used) {
  constexpr int max_iterations = 30;
  constexpr int max_backtracks = 10;
  const Real tolerance = (sizeof(Real) == sizeof(float)) ? 2.0e-6 : 2.0e-12;
  const Real fd_scale = (sizeof(Real) == sizeof(float)) ? 3.0e-3 : 1.0e-6;

  Real velocity[3] = {guess.vx, guess.vy, guess.vz};
  Real residual[3];
  iter_used = 0;
  if (!ImplicitViscousPrimitiveResidual(
          u, eos, ex_star, ey_star, ez_star, electric_kappa, pi_star, pi_ns,
          shear_dt_over_tau, shear_chi_max, fixed_spatial_shear, velocity[0],
          velocity[1], velocity[2], residual[0], residual[1], residual[2], w, pi)) {
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
      SRRMHDPrim1D scratch_w;
      ShearStress scratch_pi;
      const bool plus_ok = ImplicitViscousPrimitiveResidual(
          u, eos, ex_star, ey_star, ez_star, electric_kappa, pi_star, pi_ns,
          shear_dt_over_tau, shear_chi_max, fixed_spatial_shear, plus[0], plus[1],
          plus[2], fplus[0], fplus[1], fplus[2], scratch_w, scratch_pi);
      const bool minus_ok = ImplicitViscousPrimitiveResidual(
          u, eos, ex_star, ey_star, ez_star, electric_kappa, pi_star, pi_ns,
          shear_dt_over_tau, shear_chi_max, fixed_spatial_shear, minus[0], minus[1],
          minus[2], fminus[0], fminus[1], fminus[2], scratch_w, scratch_pi);
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
      ShearStress trial_pi;
      const bool trial_ok = ImplicitViscousPrimitiveResidual(
          u, eos, ex_star, ey_star, ez_star, electric_kappa, pi_star, pi_ns,
          shear_dt_over_tau, shear_chi_max, fixed_spatial_shear, trial[0], trial[1],
          trial[2], trial_residual[0], trial_residual[1], trial_residual[2],
          trial_w, trial_pi);
      Real trial_norm = 2.0*residual_norm;
      if (trial_ok) {
        trial_norm = fmax(fabs(trial_residual[0]),
                          fmax(fabs(trial_residual[1]), fabs(trial_residual[2])));
      }
      if (trial_norm < residual_norm) {
        for (int component = 0; component < 3; ++component) {
          velocity[component] = trial[component];
          residual[component] = trial_residual[component];
        }
        w = trial_w;
        pi = trial_pi;
        accepted = true;
        break;
      }
      damping *= 0.5;
    }
    if (!accepted) return false;
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \brief Conservative gamma-law linear causality margin.
//!
//! Using c_s,max^2=gamma-1 and c_L^2=c_s^2+4 nu/(3 tau), nonnegative margin
//! guarantees the linear longitudinal viscous signal does not exceed light speed for
//! any thermodynamic state of the gamma-law gas.  This is not a nonlinear proof.

KOKKOS_INLINE_FUNCTION
Real LinearViscosityCausalityMargin(const Real gamma,
                                    const RelativisticViscosityData &data) {
  if (!(data.tau > 0.0)) return -1.0;
  return 1.0 - (gamma - 1.0) - 4.0*data.nu/(3.0*data.tau);
}

KOKKOS_INLINE_FUNCTION
Real TransverseShearSpeedSquared(const RelativisticViscosityData &data) {
  if (!(data.tau > 0.0)) return -1.0;
  return data.nu/data.tau;
}

KOKKOS_INLINE_FUNCTION
Real LongitudinalViscousSpeedSquared(const Real sound_speed_squared,
                                     const RelativisticViscosityData &data) {
  if (!(data.tau > 0.0)) return -1.0;
  return sound_speed_squared + 4.0*data.nu/(3.0*data.tau);
}

} // namespace srrmhd

#endif // MHD_RELATIVISTIC_VISCOSITY_HPP_
