//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_viscosity.cpp
//! \brief Manufactured covariant-kinematics tests for relativistic shear viscosity.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "mhd/relativistic_viscosity.hpp"

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDViscosityKinematicsErrors()
//! \brief Verify shear projection, acceleration terms, and linear characteristic speeds.

void SRRMHDViscosityKinematicsErrors(Mesh *pm) {
  constexpr int ncases = 5;
  constexpr int nerrors = 26;
  DvceArray2D<Real> result("rsrmhd_viscosity_kinematics_test", ncases, nerrors);

  Kokkos::parallel_for("rsrmhd_viscosity_kinematics_test",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncases), KOKKOS_LAMBDA(const int n) {
    const Real u1_values[ncases] = {0.0, 0.2, -0.7, 1.4, -2.1};
    const Real u2_values[ncases] = {0.0, -0.4, 0.5, -1.1, 0.8};
    const Real u3_values[ncases] = {0.0, 0.3, 1.2, 0.6, -1.7};
    const Real u1 = u1_values[n];
    const Real u2 = u2_values[n];
    const Real u3 = u3_values[n];
    const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
    const Real u[4] = {lor, u1, u2, u3};
    const Real u_lower[4] = {-lor, u1, u2, u3};

    srrmhd::FourVelocityGradient gradient;
    for (int mu = 0; mu < 4; ++mu) {
      for (int j = 1; j < 4; ++j) {
        gradient.du[mu][j] = 0.013*(1.0 + n + 2.0*mu + 3.0*j);
        if ((mu + j + n) % 2 != 0) gradient.du[mu][j] *= -1.0;
      }
    }
    srrmhd::CompleteFourVelocityGradient(u1, u2, u3, gradient);

    Real normalization_error = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
      Real residual = 0.0;
      for (int nu = 0; nu < 4; ++nu) {
        residual += u_lower[nu]*gradient.du[mu][nu];
      }
      normalization_error = fmax(normalization_error, fabs(residual));
    }
    result(n, 5) = normalization_error;

    const srrmhd::SymmetricTensor4 shear =
        srrmhd::SpacetimeVelocityShear(u1, u2, u3, gradient);
    Real symmetry_error = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        symmetry_error = fmax(symmetry_error,
                              fabs(shear.t[mu][nu] - shear.t[nu][mu]));
      }
    }
    result(n, 0) = symmetry_error;

    const srrmhd::ShearStress spatial_shear =
        srrmhd::SpatialShearComponents(shear);
    Real shear_orthogonality, shear_trace;
    srrmhd::ShearConstraintErrors(u1, u2, u3, spatial_shear,
                                  shear_orthogonality, shear_trace);
    result(n, 1) = shear_orthogonality;
    result(n, 2) = shear_trace;

    const srrmhd::SymmetricTensor4 reconstructed_shear =
        srrmhd::AssembleSpacetimeShear(u1, u2, u3, spatial_shear);
    Real reconstruction_error = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        reconstruction_error = fmax(reconstruction_error,
            fabs(reconstructed_shear.t[mu][nu] - shear.t[mu][nu]));
      }
    }
    result(n, 3) = reconstruction_error;

    const srrmhd::FourVector acceleration =
        srrmhd::FluidFourAcceleration(u1, u2, u3, gradient);
    Real acceleration_orthogonality = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
      acceleration_orthogonality += u_lower[mu]*acceleration.v[mu];
    }
    result(n, 4) = fabs(acceleration_orthogonality);

    srrmhd::ShearStress raw_pi;
    raw_pi.p11 = 0.31 + 0.07*n;
    raw_pi.p22 = -0.18 + 0.04*n;
    raw_pi.p33 = 0.27 - 0.03*n;
    raw_pi.p12 = -0.11 + 0.02*n;
    raw_pi.p13 = 0.09 - 0.015*n;
    raw_pi.p23 = 0.14 + 0.01*n;
    const srrmhd::ShearStress pi =
        srrmhd::ProjectShearStress(u1, u2, u3, raw_pi);
    const srrmhd::SymmetricTensor4 pi4 =
        srrmhd::AssembleSpacetimeShear(u1, u2, u3, pi);
    const srrmhd::SymmetricTensor4 derivative =
        srrmhd::UnprojectedComovingShearDerivative(
            u1, u2, u3, pi, spatial_shear, acceleration);
    const Real a_lower[4] = {-acceleration.v[0], acceleration.v[1],
                              acceleration.v[2], acceleration.v[3]};
    Real derivative_constraint_error = 0.0;
    for (int nu = 0; nu < 4; ++nu) {
      Real u_dot_derivative = 0.0;
      Real a_dot_pi = 0.0;
      for (int mu = 0; mu < 4; ++mu) {
        u_dot_derivative += u_lower[mu]*derivative.t[mu][nu];
        a_dot_pi += a_lower[mu]*pi4.t[mu][nu];
      }
      derivative_constraint_error = fmax(
          derivative_constraint_error, fabs(u_dot_derivative + a_dot_pi));
    }
    result(n, 6) = derivative_constraint_error;
    result(n, 7) = fabs(-derivative.t[0][0] + derivative.t[1][1]
                         + derivative.t[2][2] + derivative.t[3][3]);

    Real rest_frame_error = 0.0;
    Real target_sign_error = 0.0;
    if (n == 0) {
      const Real expansion = gradient.du[1][1] + gradient.du[2][2]
                           + gradient.du[3][3];
      rest_frame_error = fmax(rest_frame_error,
          fabs(shear.t[1][1] - (gradient.du[1][1] - expansion/3.0)));
      rest_frame_error = fmax(rest_frame_error,
          fabs(shear.t[2][2] - (gradient.du[2][2] - expansion/3.0)));
      rest_frame_error = fmax(rest_frame_error,
          fabs(shear.t[3][3] - (gradient.du[3][3] - expansion/3.0)));
      rest_frame_error = fmax(rest_frame_error,
          fabs(shear.t[1][2]
               - 0.5*(gradient.du[1][2] + gradient.du[2][1])));

      srrmhd::FourVelocityGradient simple_shear_gradient;
      simple_shear_gradient.du[1][2] = 0.4;
      srrmhd::CompleteFourVelocityGradient(0.0, 0.0, 0.0,
                                           simple_shear_gradient);
      const srrmhd::SymmetricTensor4 simple_shear =
          srrmhd::SpacetimeVelocityShear(0.0, 0.0, 0.0,
                                         simple_shear_gradient);
      const srrmhd::ShearStress target =
          srrmhd::NavierStokesShearTarget(2.0, 0.1, simple_shear);
      target_sign_error = fabs(target.p12 + 0.08);
      target_sign_error = fmax(target_sign_error, fabs(target.p11));
      target_sign_error = fmax(target_sign_error, fabs(target.p22));
      target_sign_error = fmax(target_sign_error, fabs(target.p33));
    }
    result(n, 8) = rest_frame_error;
    result(n, 9) = target_sign_error;

    Real direct_norm_squared = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
      const Real sign_mu = (mu == 0) ? -1.0 : 1.0;
      for (int nu = 0; nu < 4; ++nu) {
        const Real sign_nu = (nu == 0) ? -1.0 : 1.0;
        direct_norm_squared += sign_mu*sign_nu*SQR(pi4.t[mu][nu]);
      }
    }
    const Real invariant_norm = srrmhd::ShearInvariantNorm(u1, u2, u3, pi);
    result(n, 10) = fabs(SQR(invariant_norm) - direct_norm_squared);

    constexpr Real limiter_enthalpy = 0.05;
    constexpr Real limiter_chi = 0.4;
    const srrmhd::ShearStress limited = srrmhd::LimitShearInverseReynolds(
        u1, u2, u3, limiter_enthalpy, limiter_chi, pi);
    const Real limited_norm = srrmhd::ShearInvariantNorm(u1, u2, u3, limited);
    result(n, 11) = fabs(limited_norm
        - fmin(invariant_norm, limiter_chi*limiter_enthalpy));
    Real limited_orthogonality, limited_trace;
    srrmhd::ShearConstraintErrors(u1, u2, u3, limited,
                                  limited_orthogonality, limited_trace);
    result(n, 12) = fmax(limited_orthogonality, limited_trace);

    const Real enthalpy_density = 2.5 + 0.1*n;
    srrmhd::FourVector manufactured_acceleration;
    manufactured_acceleration.v[1] = 0.07 - 0.01*n;
    manufactured_acceleration.v[2] = -0.04 + 0.006*n;
    manufactured_acceleration.v[3] = 0.03 + 0.004*n;
    manufactured_acceleration.v[0] =
        (u1*manufactured_acceleration.v[1]
           + u2*manufactured_acceleration.v[2]
           + u3*manufactured_acceleration.v[3])/lor;
    const Real manufactured_a_lower[4] = {
        -manufactured_acceleration.v[0], manufactured_acceleration.v[1],
        manufactured_acceleration.v[2], manufactured_acceleration.v[3]};
    Real acceleration_rhs[3] = {};
    for (int i = 1; i < 4; ++i) {
      acceleration_rhs[i-1] = enthalpy_density*manufactured_acceleration.v[i];
      for (int mu = 0; mu < 4; ++mu) {
        acceleration_rhs[i-1] += pi4.t[i][mu]*manufactured_a_lower[mu];
      }
    }
    srrmhd::FourVector recovered_acceleration;
    const bool acceleration_success = srrmhd::SolveShearAccelerationMassMatrix(
        u1, u2, u3, enthalpy_density, pi, acceleration_rhs,
        recovered_acceleration);
    Real acceleration_error = acceleration_success ? 0.0 : 1.0;
    Real acceleration_residual = acceleration_success ? 0.0 : 1.0;
    Real solved_orthogonality = acceleration_success ? 0.0 : 1.0;
    if (acceleration_success) {
      for (int mu = 0; mu < 4; ++mu) {
        acceleration_error = fmax(acceleration_error,
            fabs(recovered_acceleration.v[mu] - manufactured_acceleration.v[mu]));
      }
      const Real recovered_a_lower[4] = {
          -recovered_acceleration.v[0], recovered_acceleration.v[1],
          recovered_acceleration.v[2], recovered_acceleration.v[3]};
      for (int i = 1; i < 4; ++i) {
        Real residual = enthalpy_density*recovered_acceleration.v[i]
                      - acceleration_rhs[i-1];
        for (int mu = 0; mu < 4; ++mu) {
          residual += pi4.t[i][mu]*recovered_a_lower[mu];
        }
        acceleration_residual = fmax(acceleration_residual, fabs(residual));
      }
      for (int mu = 0; mu < 4; ++mu) {
        solved_orthogonality += u_lower[mu]*recovered_acceleration.v[mu];
      }
      solved_orthogonality = fabs(solved_orthogonality);
    }
    result(n, 13) = acceleration_error;
    result(n, 14) = acceleration_residual;
    result(n, 15) = solved_orthogonality;

    constexpr Real gamma = 5.0/3.0;
    const Real density = 1.0 + 0.1*n;
    const Real internal_energy = 0.5 + 0.04*n;
    const Real pressure = (gamma - 1.0)*internal_energy;
    const Real nonlinear_enthalpy = density + gamma*internal_energy;
    srrmhd::ShearGradient1D nonlinear_gradient;
    nonlinear_gradient.pressure = 0.017*(n + 1.0);
    nonlinear_gradient.u1 = gradient.du[1][1];
    nonlinear_gradient.u2 = gradient.du[1][2];
    nonlinear_gradient.u3 = gradient.du[1][3];
    nonlinear_gradient.pi.p11 = 0.011 + 0.002*n;
    nonlinear_gradient.pi.p22 = -0.008 + 0.001*n;
    nonlinear_gradient.pi.p33 = 0.006 - 0.0005*n;
    nonlinear_gradient.pi.p12 = -0.004 + 0.0007*n;
    nonlinear_gradient.pi.p13 = 0.003 + 0.0004*n;
    nonlinear_gradient.pi.p23 = -0.002 + 0.0003*n;
    srrmhd::FourVector four_force;
    four_force.v[0] = 0.009 + 0.001*n;
    four_force.v[1] = -0.013 + 0.002*n;
    four_force.v[2] = 0.007 - 0.0005*n;
    four_force.v[3] = -0.005 + 0.0008*n;
    srrmhd::FourVector nonlinear_acceleration;
    srrmhd::ShearStress nonlinear_sigma;
    const bool nonlinear_success = srrmhd::SolveShearAcceleration1D(
        u1, u2, u3, pressure, nonlinear_enthalpy, gamma,
        0.01*nonlinear_enthalpy, 0.2, pi, nonlinear_gradient, four_force,
        nonlinear_acceleration, nonlinear_sigma);
    Real nonlinear_residual_error = nonlinear_success ? 0.0 : 1.0;
    Real nonlinear_constraint_error = nonlinear_success ? 0.0 : 1.0;
    Real contraction_orthogonality = nonlinear_success ? 0.0 : 1.0;
    Real source_reduction_error = nonlinear_success ? 0.0 : 1.0;
    if (nonlinear_success) {
      Real nonlinear_residual[3];
      srrmhd::ShearStress residual_sigma;
      const bool residual_success = srrmhd::ShearAccelerationResidual1D(
          u1, u2, u3, pressure, nonlinear_enthalpy, gamma,
          0.01*nonlinear_enthalpy, 0.2, pi, nonlinear_gradient, four_force,
          nonlinear_acceleration, nonlinear_residual, residual_sigma);
      if (!residual_success) {
        nonlinear_residual_error = 1.0;
      } else {
        nonlinear_residual_error = fmax(fabs(nonlinear_residual[0]),
            fmax(fabs(nonlinear_residual[1]), fabs(nonlinear_residual[2])));
      }
      Real sigma_orthogonality, sigma_trace;
      srrmhd::ShearConstraintErrors(u1, u2, u3, nonlinear_sigma,
                                    sigma_orthogonality, sigma_trace);
      nonlinear_constraint_error = fmax(sigma_orthogonality, sigma_trace);
      const srrmhd::FourVector a_pi = srrmhd::ShearAccelerationContraction(
          u1, u2, u3, pi, nonlinear_acceleration);
      contraction_orthogonality = fabs(-lor*a_pi.v[0] + u1*a_pi.v[1]
                                         + u2*a_pi.v[2] + u3*a_pi.v[3]);
      const srrmhd::ShearStress source = srrmhd::KinematicShearSourcePerD(
          u1, u2, u3, pi, nonlinear_acceleration);
      source_reduction_error = fmax(fabs(source.p11 - 2.0*u1*a_pi.v[1]/lor),
          fmax(fabs(source.p22 - 2.0*u2*a_pi.v[2]/lor),
          fmax(fabs(source.p33 - 2.0*u3*a_pi.v[3]/lor),
          fmax(fabs(source.p12 - (u1*a_pi.v[2] + u2*a_pi.v[1])/lor),
          fmax(fabs(source.p13 - (u1*a_pi.v[3] + u3*a_pi.v[1])/lor),
               fabs(source.p23 - (u2*a_pi.v[3] + u3*a_pi.v[2])/lor))))));
    }
    result(n, 16) = nonlinear_residual_error;
    result(n, 17) = nonlinear_constraint_error;
    result(n, 18) = contraction_orthogonality;
    result(n, 19) = source_reduction_error;

    Real compression_acceleration_error = 0.0;
    Real compression_target_error = 0.0;
    if (n == 0) {
      srrmhd::ShearStress zero_pi;
      srrmhd::ShearGradient1D compression_gradient;
      compression_gradient.u1 = 0.3;
      srrmhd::FourVector zero_force;
      srrmhd::FourVector compression_acceleration;
      srrmhd::ShearStress compression_sigma;
      const bool compression_success = srrmhd::SolveShearAcceleration1D(
          0.0, 0.0, 0.0, pressure, nonlinear_enthalpy, gamma,
          0.01*nonlinear_enthalpy, 0.2, zero_pi, compression_gradient,
          zero_force, compression_acceleration, compression_sigma);
      if (!compression_success) {
        compression_acceleration_error = 1.0;
        compression_target_error = 1.0;
      } else {
        for (int mu = 0; mu < 4; ++mu) {
          compression_acceleration_error = fmax(
              compression_acceleration_error, fabs(compression_acceleration.v[mu]));
        }
        compression_target_error = fmax(fabs(compression_sigma.p11 - 0.2),
            fmax(fabs(compression_sigma.p22 + 0.1),
            fmax(fabs(compression_sigma.p33 + 0.1),
            fmax(fabs(compression_sigma.p12),
            fmax(fabs(compression_sigma.p13), fabs(compression_sigma.p23))))));
      }
    }
    result(n, 20) = compression_acceleration_error;
    result(n, 21) = compression_target_error;

    // Exercise the dimension-independent operator with all derivative directions,
    // then cyclically relabel the Cartesian axes.  A Cartesian constitutive operator
    // must return the correspondingly relabelled acceleration and shear tensor.
    srrmhd::ShearGradient3D full_gradient;
    for (int d = 0; d < 3; ++d) {
      full_gradient.pressure[d] = 0.006*(n + 1.0)*(d + 1.0);
      if ((d + n) % 2 != 0) full_gradient.pressure[d] *= -1.0;
      for (int i = 0; i < 3; ++i) {
        full_gradient.du[d][i] = 0.004*(1.0 + n + 2.0*d + 3.0*i);
        if ((d + 2*i + n) % 2 != 0) full_gradient.du[d][i] *= -1.0;
      }
      Real dpi_matrix[3][3] = {};
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          Real value = 0.001*(1.0 + n + 2.0*d + 3.0*i + 5.0*j);
          if ((d + i + j + n) % 2 != 0) value *= -1.0;
          dpi_matrix[i][j] = dpi_matrix[j][i] = value;
        }
      }
      full_gradient.dpi[d] = srrmhd::MatrixShearStress(dpi_matrix);
    }
    srrmhd::FourVector full_acceleration;
    srrmhd::ShearStress full_sigma;
    const bool full_success = srrmhd::SolveShearAcceleration(
        u1, u2, u3, pressure, nonlinear_enthalpy, gamma,
        0.01*nonlinear_enthalpy, 0.2, pi, full_gradient, four_force,
        full_acceleration, full_sigma);
    Real full_residual_error = full_success ? 0.0 : 1.0;
    Real full_constraint_error = full_success ? 0.0 : 1.0;
    Real rotation_acceleration_error = full_success ? 0.0 : 1.0;
    Real rotation_sigma_error = full_success ? 0.0 : 1.0;
    if (full_success) {
      Real full_residual[3];
      srrmhd::ShearStress residual_sigma;
      if (!srrmhd::ShearAccelerationResidual(
              u1, u2, u3, pressure, nonlinear_enthalpy, gamma,
              0.01*nonlinear_enthalpy, 0.2, pi, full_gradient, four_force,
              full_acceleration, full_residual, residual_sigma)) {
        full_residual_error = 1.0;
      } else {
        for (int i = 0; i < 3; ++i) {
          full_residual_error = fmax(full_residual_error, fabs(full_residual[i]));
        }
      }
      Real sigma_orthogonality, sigma_trace;
      srrmhd::ShearConstraintErrors(u1, u2, u3, full_sigma,
                                    sigma_orthogonality, sigma_trace);
      full_constraint_error = fmax(sigma_orthogonality, sigma_trace);

      constexpr int permutation[3] = {1, 2, 0};
      const Real spatial_u[3] = {u1, u2, u3};
      const Real permuted_u[3] = {spatial_u[permutation[0]],
                                  spatial_u[permutation[1]],
                                  spatial_u[permutation[2]]};
      srrmhd::FourVector permuted_force;
      permuted_force.v[0] = four_force.v[0];
      for (int i = 0; i < 3; ++i) {
        permuted_force.v[i+1] = four_force.v[permutation[i]+1];
      }
      Real pi_matrix[3][3];
      Real permuted_pi_matrix[3][3];
      srrmhd::ShearStressMatrix(pi, pi_matrix);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          permuted_pi_matrix[i][j] = pi_matrix[permutation[i]][permutation[j]];
        }
      }
      const srrmhd::ShearStress permuted_pi =
          srrmhd::MatrixShearStress(permuted_pi_matrix);
      srrmhd::ShearGradient3D permuted_gradient;
      for (int d = 0; d < 3; ++d) {
        permuted_gradient.pressure[d] = full_gradient.pressure[permutation[d]];
        for (int i = 0; i < 3; ++i) {
          permuted_gradient.du[d][i] =
              full_gradient.du[permutation[d]][permutation[i]];
        }
        Real dpi_matrix[3][3];
        Real permuted_dpi_matrix[3][3];
        srrmhd::ShearStressMatrix(full_gradient.dpi[permutation[d]], dpi_matrix);
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            permuted_dpi_matrix[i][j] =
                dpi_matrix[permutation[i]][permutation[j]];
          }
        }
        permuted_gradient.dpi[d] = srrmhd::MatrixShearStress(permuted_dpi_matrix);
      }
      srrmhd::FourVector permuted_acceleration;
      srrmhd::ShearStress permuted_sigma;
      const bool permuted_success = srrmhd::SolveShearAcceleration(
          permuted_u[0], permuted_u[1], permuted_u[2], pressure,
          nonlinear_enthalpy, gamma, 0.01*nonlinear_enthalpy, 0.2,
          permuted_pi, permuted_gradient, permuted_force, permuted_acceleration,
          permuted_sigma);
      if (!permuted_success) {
        rotation_acceleration_error = 1.0;
        rotation_sigma_error = 1.0;
      } else {
        rotation_acceleration_error =
            fabs(permuted_acceleration.v[0] - full_acceleration.v[0]);
        for (int i = 0; i < 3; ++i) {
          rotation_acceleration_error = fmax(rotation_acceleration_error,
              fabs(permuted_acceleration.v[i+1]
                   - full_acceleration.v[permutation[i]+1]));
        }
        Real sigma_matrix[3][3];
        Real permuted_sigma_matrix[3][3];
        srrmhd::ShearStressMatrix(full_sigma, sigma_matrix);
        srrmhd::ShearStressMatrix(permuted_sigma, permuted_sigma_matrix);
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            rotation_sigma_error = fmax(rotation_sigma_error,
                fabs(permuted_sigma_matrix[i][j]
                     - sigma_matrix[permutation[i]][permutation[j]]));
          }
        }
      }
    }
    result(n, 22) = full_residual_error;
    result(n, 23) = full_constraint_error;
    result(n, 24) = rotation_acceleration_error;
    result(n, 25) = rotation_sigma_error;
  });

  auto r = Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
  Real errors[nerrors] = {};
  for (int n = 0; n < ncases; ++n) {
    for (int e = 0; e < nerrors; ++e) {
      errors[e] = std::max(errors[e], r(n, e));
    }
  }

  const auto data = pm->pmb_pack->pmhd->relativistic_viscosity_data;
  const Real transverse_speed = srrmhd::TransverseShearSpeedSquared(data);
  const Real longitudinal_speed =
      srrmhd::LongitudinalViscousSpeedSquared(0.2, data);
  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_viscosity_kinematics-errs.dat");
    file << std::setprecision(17);
    for (int e = 0; e < nerrors; ++e) file << errors[e] << " ";
    file << transverse_speed << " " << longitudinal_speed << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDViscousImplicitErrors()
//! \brief Verify block-eliminated electric-plus-shear primitive recovery.

void SRRMHDViscousImplicitErrors(Mesh *pm) {
  auto eos = pm->pmb_pack->pmhd->peos->eos_data;
  constexpr int ncases = 4;
  DvceArray2D<Real> result("rsrmhd_viscous_implicit_test", ncases, 10);

  Kokkos::parallel_for("rsrmhd_viscous_implicit_test",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncases), KOKKOS_LAMBDA(const int n) {
    const Real electric_kappas[ncases] = {1.0e-4, 0.1, 10.0, 1.0e4};
    const Real target_shear_kappas[ncases] = {1.0e4, 10.0, 0.1, 1.0e-4};
    const Real electric_kappa = electric_kappas[n];
    const Real target_shear_kappa = target_shear_kappas[n];

    srrmhd::SRRMHDPrim1D target;
    target.d = 0.9 + 0.15*n;
    target.vx = 0.22 + 0.07*n;
    target.vy = -0.31 + 0.035*n;
    target.vz = 0.16 - 0.025*n;
    target.e = 0.55 + 0.12*n;
    target.ex = 0.11 - 0.02*n;
    target.ey = -0.24 + 0.025*n;
    target.ez = 0.29 - 0.015*n;
    target.bx = 0.65;
    target.by = -0.35;
    target.bz = 0.85;

    srrmhd::ShearStress raw_target_pi;
    raw_target_pi.p11 = 0.012 + 0.001*n;
    raw_target_pi.p22 = -0.007 + 0.0005*n;
    raw_target_pi.p33 = 0.004 - 0.0003*n;
    raw_target_pi.p12 = -0.006 + 0.0004*n;
    raw_target_pi.p13 = 0.005 - 0.0002*n;
    raw_target_pi.p23 = 0.003 + 0.0001*n;
    const srrmhd::ShearStress target_pi = srrmhd::ProjectShearStress(
        target.vx, target.vy, target.vz, raw_target_pi);

    srrmhd::ShearStress raw_pi_ns;
    raw_pi_ns.p11 = -0.005;
    raw_pi_ns.p22 = 0.008;
    raw_pi_ns.p33 = -0.002;
    raw_pi_ns.p12 = 0.004;
    raw_pi_ns.p13 = -0.003;
    raw_pi_ns.p23 = 0.002;
    const srrmhd::ShearStress pi_ns = srrmhd::ProjectShearStress(
        target.vx, target.vy, target.vz, raw_pi_ns);

    srrmhd::ShearStress pi_star;
    pi_star.p11 = (1.0 + target_shear_kappa)*target_pi.p11
                - target_shear_kappa*pi_ns.p11;
    pi_star.p22 = (1.0 + target_shear_kappa)*target_pi.p22
                - target_shear_kappa*pi_ns.p22;
    pi_star.p33 = (1.0 + target_shear_kappa)*target_pi.p33
                - target_shear_kappa*pi_ns.p33;
    pi_star.p12 = (1.0 + target_shear_kappa)*target_pi.p12
                - target_shear_kappa*pi_ns.p12;
    pi_star.p13 = (1.0 + target_shear_kappa)*target_pi.p13
                - target_shear_kappa*pi_ns.p13;
    pi_star.p23 = (1.0 + target_shear_kappa)*target_pi.p23
                - target_shear_kappa*pi_ns.p23;

    srrmhd::SRRMHDCons1D conserved;
    srrmhd::SingleP2C_IdealSRRMHD(target, eos.gamma, conserved);
    srrmhd::AddShearToConserved(target.vx, target.vy, target.vz, target_pi,
                                conserved);

    const Real lor = sqrt(1.0 + SQR(target.vx) + SQR(target.vy)
                           + SQR(target.vz));
    const Real shear_dt_over_tau = target_shear_kappa*lor;
    const Real v1 = target.vx/lor;
    const Real v2 = target.vy/lor;
    const Real v3 = target.vz/lor;
    const Real vxb1 = v2*target.bz - v3*target.by;
    const Real vxb2 = v3*target.bx - v1*target.bz;
    const Real vxb3 = v1*target.by - v2*target.bx;
    const Real edotv = v1*target.ex + v2*target.ey + v3*target.ez;
    const Real electric_scale = electric_kappa*lor;
    const Real ex_star = target.ex
        + electric_scale*(target.ex + vxb1 - edotv*v1);
    const Real ey_star = target.ey
        + electric_scale*(target.ey + vxb2 - edotv*v2);
    const Real ez_star = target.ez
        + electric_scale*(target.ez + vxb3 - edotv*v3);

    srrmhd::SRRMHDPrim1D guess = target;
    guess.vx += 0.10;
    guess.vy -= 0.08;
    guess.vz += 0.06;
    srrmhd::SRRMHDPrim1D recovered = guess;
    srrmhd::ShearStress recovered_pi;
    int iterations = 0;
    const bool success = srrmhd::SingleC2P_IdealSRRMHDImplicitViscous(
        conserved, eos, ex_star, ey_star, ez_star, electric_kappa, pi_star,
        pi_ns, shear_dt_over_tau, 2.0, false, guess, recovered, recovered_pi,
        iterations);

    Real primitive_error = success ? 0.0 : 1.0;
    Real electric_error = success ? 0.0 : 1.0;
    Real shear_error = success ? 0.0 : 1.0;
    if (success) {
      primitive_error = fmax(primitive_error, fabs(recovered.d - target.d));
      primitive_error = fmax(primitive_error, fabs(recovered.vx - target.vx));
      primitive_error = fmax(primitive_error, fabs(recovered.vy - target.vy));
      primitive_error = fmax(primitive_error, fabs(recovered.vz - target.vz));
      primitive_error = fmax(primitive_error, fabs(recovered.e - target.e));
      electric_error = fmax(fabs(recovered.ex - target.ex),
          fmax(fabs(recovered.ey - target.ey), fabs(recovered.ez - target.ez)));
      shear_error = fmax(fabs(recovered_pi.p11 - target_pi.p11),
          fmax(fabs(recovered_pi.p22 - target_pi.p22),
          fmax(fabs(recovered_pi.p33 - target_pi.p33),
          fmax(fabs(recovered_pi.p12 - target_pi.p12),
          fmax(fabs(recovered_pi.p13 - target_pi.p13),
               fabs(recovered_pi.p23 - target_pi.p23))))));
    }

    Real f1, f2, f3;
    srrmhd::SRRMHDPrim1D residual_state;
    srrmhd::ShearStress residual_pi;
    const bool residual_ok = srrmhd::ImplicitViscousPrimitiveResidual(
        conserved, eos, ex_star, ey_star, ez_star, electric_kappa, pi_star,
        pi_ns, shear_dt_over_tau, 2.0, false, recovered.vx, recovered.vy,
        recovered.vz, f1, f2, f3, residual_state, residual_pi);
    Real residual_error = 1.0;
    if (success && residual_ok) {
      residual_error = fmax(fabs(f1), fmax(fabs(f2), fabs(f3)));
    }

    Real constraint_error = 1.0;
    if (success) {
      Real orthogonality_error, trace_error;
      srrmhd::ShearConstraintErrors(recovered.vx, recovered.vy, recovered.vz,
                                    recovered_pi, orthogonality_error, trace_error);
      constraint_error = fmax(orthogonality_error, trace_error);
    }

    result(n, 0) = success ? 0.0 : 1.0;
    result(n, 1) = primitive_error;
    result(n, 2) = electric_error;
    result(n, 3) = shear_error;
    result(n, 4) = residual_error;
    result(n, 5) = constraint_error;
    result(n, 6) = iterations;

    const srrmhd::ShearStress conservative_pi =
        srrmhd::ConservativeShearStress(conserved.d, target_pi);
    const srrmhd::ShearStress primitive_pi =
        srrmhd::PrimitiveShearStress(conserved.d, conservative_pi);
    result(n, 7) = fmax(fabs(primitive_pi.p11 - target_pi.p11),
        fmax(fabs(primitive_pi.p22 - target_pi.p22),
        fmax(fabs(primitive_pi.p33 - target_pi.p33),
        fmax(fabs(primitive_pi.p12 - target_pi.p12),
        fmax(fabs(primitive_pi.p13 - target_pi.p13),
             fabs(primitive_pi.p23 - target_pi.p23))))));

    srrmhd::ShearStress zero_target;
    const Real homogeneous_kappa = 0.7;
    const srrmhd::ShearStress relaxed_p =
        srrmhd::ImplicitConservativeShearRelaxation(
            conserved.d, target.vx, target.vy, target.vz, conservative_pi,
            zero_target, homogeneous_kappa*lor,
            target.d + eos.gamma*target.e, 2.0);
    const srrmhd::ShearStress relaxed_pi =
        srrmhd::PrimitiveShearStress(conserved.d, relaxed_p);
    result(n, 8) = fabs(
        relaxed_pi.p12 - target_pi.p12/(1.0 + homogeneous_kappa));

    constexpr Real test_chi_max = 1.0e-3;
    Real limited_f1, limited_f2, limited_f3;
    srrmhd::SRRMHDPrim1D limited_state;
    srrmhd::ShearStress limited_pi;
    const bool limited_ok = srrmhd::ImplicitViscousPrimitiveResidual(
        conserved, eos, ex_star, ey_star, ez_star, electric_kappa, pi_star,
        pi_ns, shear_dt_over_tau, test_chi_max, false, target.vx, target.vy,
        target.vz, limited_f1, limited_f2, limited_f3, limited_state, limited_pi);
    Real limiter_error = 1.0;
    if (limited_ok) {
      const Real limited_enthalpy = limited_state.d + eos.gamma*limited_state.e;
      const Real limited_chi = srrmhd::ShearInvariantNorm(
          target.vx, target.vy, target.vz, limited_pi)/limited_enthalpy;
      limiter_error = fabs(limited_chi - test_chi_max);
    }
    result(n, 9) = limiter_error;
  });

  auto r = Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
  Real failures = 0.0;
  Real errors[8] = {};
  Real max_iterations = 0.0;
  for (int n = 0; n < ncases; ++n) {
    failures += r(n, 0);
    for (int e = 0; e < 5; ++e) errors[e] = std::max(errors[e], r(n, e + 1));
    errors[5] = std::max(errors[5], r(n, 7));
    errors[6] = std::max(errors[6], r(n, 8));
    errors[7] = std::max(errors[7], r(n, 9));
    max_iterations = std::max(max_iterations, r(n, 6));
  }

  auto *pmhd = pm->pmb_pack->pmhd;
  auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
  auto visc_w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_w0);
  auto mhd_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u0);
  auto &indcs = pm->mb_indcs;
  Real persistent_state_error = 0.0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      for (int j = indcs.js; j <= indcs.je; ++j) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          for (int n = 0; n < srrmhd::NVISC; ++n) {
            persistent_state_error = std::max(persistent_state_error,
                fabs(visc_u(m, n, k, j, i)
                     - mhd_u(m, IDN, k, j, i)*visc_w(m, n, k, j, i)));
          }
        }
      }
    }
  }

  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_viscous_implicit-errs.dat");
    file << std::setprecision(17) << failures << " ";
    for (int e = 0; e < 8; ++e) file << errors[e] << " ";
    file << max_iterations << " " << persistent_state_error << std::endl;
  }
}
