//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file antenna_driver.cpp
//! \brief Electromagnetic oscillating-Langevin antenna for resistive SRMHD.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "globals.hpp"
#include "mhd/dual_ct.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"
#include "antenna_driver.hpp"

namespace {

[[noreturn]] void AntennaFatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

bool NearlyEqual(Real a, Real b) {
  const Real scale = std::max({1.0, std::abs(a), std::abs(b)});
  return std::abs(a - b) <= 100.0*std::numeric_limits<Real>::epsilon()*scale;
}

Real AntennaGaussian(RNG_State *state) {
  if (state->idum < 0) state->iset = 0;
  if (state->iset != 0) {
    state->iset = 0;
    return state->gset;
  }
  Real radius_squared, value1, value2;
  do {
    value1 = 2.0*RanSt(state) - 1.0;
    value2 = 2.0*RanSt(state) - 1.0;
    radius_squared = value1*value1 + value2*value2;
  } while (radius_squared >= 1.0 || radius_squared == 0.0);
  const Real factor = std::sqrt(-2.0*std::log(radius_squared)/radius_squared);
  state->gset = value1*factor;
  state->iset = 1;
  return value2*factor;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \brief Construct the literature-baseline Zhdankin eight-mode antenna.

AntennaDriver::AntennaDriver(MeshBlockPack *pp, ParameterInput *pin) :
    current("antenna_current", 1, 1, 1, 1, 1),
    current_face("antenna_current_face", 1, 1, 1, 1),
    mode_state("antenna_mode_state", num_families, num_modes, num_quadratures),
    mode_wavevector("antenna_mode_wavevector", num_modes, 3),
    stage_task_id(0),
    pmy_pack(pp) {
  if (pmy_pack->pmhd == nullptr || !pmy_pack->pmhd->is_resistive_rel
      || !pmy_pack->pcoord->is_special_relativistic) {
    AntennaFatal("<antenna_driving> requires special-relativistic resistive MHD with "
                 "an evolved electric field");
  }
  if (!pmy_pack->pmesh->three_d) {
    AntennaFatal("The zhdankin8 antenna currently requires a three-dimensional mesh");
  }
  if (pmy_pack->pmesh->multilevel) {
    AntennaFatal("The first antenna implementation supports uniform meshes only");
  }
  auto &mesh_indcs = pmy_pack->pmesh->mesh_indcs;
  if (mesh_indcs.nx1 < 4 || mesh_indcs.nx1 != mesh_indcs.nx2
      || mesh_indcs.nx1 != mesh_indcs.nx3) {
    AntennaFatal("mode_set=zhdankin8 requires at least 4^3 cells and equal resolution");
  }

  const std::string mode_set = pin->GetOrAddString(
      "antenna_driving", "mode_set", "zhdankin8");
  if (mode_set != "zhdankin8") {
    AntennaFatal("Unknown <antenna_driving>/mode_set='" + mode_set
                 + "'; the first implementation supports zhdankin8");
  }
  const std::string guide_axis = pin->GetOrAddString(
      "antenna_driving", "guide_axis", "z");
  if (guide_axis != "z" && guide_axis != "x3") {
    AntennaFatal("The zhdankin8 antenna currently requires guide_axis=z");
  }
  const std::string current_geometry = pin->GetOrAddString(
      "antenna_driving", "current_geometry", "apar_double_curl");
  if (current_geometry != "apar_double_curl") {
    AntennaFatal("The first antenna implementation requires "
                 "current_geometry=apar_double_curl");
  }

  const char *boundary_names[6] = {
      "ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"};
  for (const char *name : boundary_names) {
    if (pin->GetString("mesh", name) != "periodic") {
      AntennaFatal("The Fourier antenna requires periodic boundaries in all directions");
    }
  }

  const Real lx = pmy_pack->pmesh->mesh_size.x1max
                  - pmy_pack->pmesh->mesh_size.x1min;
  const Real ly = pmy_pack->pmesh->mesh_size.x2max
                  - pmy_pack->pmesh->mesh_size.x2min;
  const Real lz = pmy_pack->pmesh->mesh_size.x3max
                  - pmy_pack->pmesh->mesh_size.x3min;
  if (lx <= 0.0 || !NearlyEqual(lx, ly) || !NearlyEqual(lx, lz)) {
    AntennaFatal("mode_set=zhdankin8 requires a periodic cubic domain");
  }

  apply_source = pin->GetOrAddBoolean("antenna_driving", "apply_source", true);
  frequency_factor = pin->GetOrAddReal(
      "antenna_driving", "frequency_factor", 0.6);
  decorrelation_factor = pin->GetOrAddReal(
      "antenna_driving", "decorrelation_factor", 0.5);
  const std::string amplitude_normalization = pin->GetOrAddString(
      "antenna_driving", "amplitude_normalization", "apar_rms");
  if (amplitude_normalization == "apar_rms") {
    apar_rms[0] = pin->GetOrAddReal("antenna_driving", "apar_rms_plus", 1.0e-3);
    apar_rms[1] = pin->GetOrAddReal("antenna_driving", "apar_rms_minus", 1.0e-3);
  } else if (amplitude_normalization == "zhdankin") {
    zhdankin_amplitude = true;
    amplitude_fraction[0] = pin->GetOrAddReal(
        "antenna_driving", "amplitude_fraction_plus", 1.0);
    amplitude_fraction[1] = pin->GetOrAddReal(
        "antenna_driving", "amplitude_fraction_minus", 1.0);
  } else {
    AntennaFatal("<antenna_driving>/amplitude_normalization must be apar_rms or "
                 "zhdankin");
  }
  if (frequency_factor < 0.0 || decorrelation_factor < 0.0
      || apar_rms[0] < 0.0 || apar_rms[1] < 0.0
      || amplitude_fraction[0] < 0.0 || amplitude_fraction[1] < 0.0) {
    AntennaFatal("Antenna frequency, decorrelation, and RMS amplitudes must be "
                 "nonnegative");
  }

  const std::string initial_state = pin->GetOrAddString(
      "antenna_driving", "initial_state", "stationary");
  if (initial_state == "stationary") {
    stationary_initial_state = true;
  } else if (initial_state == "zero") {
    stationary_initial_state = false;
  } else {
    AntennaFatal("<antenna_driving>/initial_state must be stationary or zero");
  }

  va_reference = pin->GetOrAddString(
      "antenna_driving", "va_reference", "initial_mean");
  if (va_reference == "fixed") {
    fixed_alfven_speed = true;
    alfven_speed_reference = pin->GetReal("antenna_driving", "alfven_speed");
    if (alfven_speed_reference <= 0.0 || alfven_speed_reference >= 1.0) {
      AntennaFatal("A fixed antenna Alfven speed must lie strictly between zero and one");
    }
  } else if (va_reference != "initial_mean") {
    AntennaFatal("<antenna_driving>/va_reference must be initial_mean or fixed");
  }

  int seed = pin->GetOrAddInteger("antenna_driving", "seed", 210989);
  if (seed == 0) seed = 1;
  rstate = {};
  rstate.idum = -std::abs(static_cast<int64_t>(seed));

  // Four independent wavevectors; their complex conjugates supply the other four
  // members of the eight-mode set used by Zhdankin et al. (2018).
  const int mode_integer[num_modes][3] = {
      {1, 0, 1}, {0, 1, 1}, {-1, 0, 1}, {0, -1, 1}};
  const Real dk = 2.0*M_PI/lx;
  for (int n = 0; n < num_modes; ++n) {
    for (int d = 0; d < 3; ++d) {
      mode_wavevector.h_view(n, d) = dk*mode_integer[n][d];
    }
  }
  mode_wavevector.template modify<HostMemSpace>();
  mode_wavevector.template sync<DevExeSpace>();

  for (int family = 0; family < num_families; ++family) {
    for (int n = 0; n < num_modes; ++n) {
      for (int q = 0; q < num_quadratures; ++q) {
        mode_state.h_view(family, n, q) = 0.0;
      }
    }
  }
  mode_state.template modify<HostMemSpace>();
  mode_state.template sync<DevExeSpace>();

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = indcs.nx2 + 2*indcs.ng;
  const int n3 = indcs.nx3 + 2*indcs.ng;
  Kokkos::realloc(current, nmb, 3, n3, n2, n1);
  Kokkos::realloc(current_face.x1f, nmb, n3, n2, n1 + 1);
  Kokkos::realloc(current_face.x2f, nmb, n3, n2 + 1, n1);
  Kokkos::realloc(current_face.x3f, nmb, n3 + 1, n2, n1);
  Kokkos::deep_copy(current, 0.0);
  Kokkos::deep_copy(current_face.x1f, 0.0);
  Kokkos::deep_copy(current_face.x2f, 0.0);
  Kokkos::deep_copy(current_face.x3f, 0.0);
}

//----------------------------------------------------------------------------------------
//! \brief Add the coefficient update/current synthesis before the time integrator.

void AntennaDriver::IncludeUpdateTask(std::shared_ptr<TaskList> tl, TaskID start) {
  auto id = tl->AddTask(&AntennaDriver::UpdateAntenna, this, start);
  (void)id;
}

//----------------------------------------------------------------------------------------
//! \brief Insert the electromagnetic source after the explicit E update.

void AntennaDriver::IncludeApplyTask(std::shared_ptr<TaskList> tl, TaskID start) {
  TaskID none(0);
  TaskID dependency = (start == none) ? pmy_pack->pmhd->id.ect : start;
  stage_task_id = tl->InsertTask(&AntennaDriver::ApplyAntenna, this, dependency,
                                pmy_pack->pmhd->id.srctrms);
  if (stage_task_id == none) {
    AntennaFatal("Unable to insert antenna source into the MHD stage task list");
  }
}

//----------------------------------------------------------------------------------------
//! \brief Measure the initial guide field/enthalpy and freeze the reference Alfven speed.

void AntennaDriver::InitializeReferenceState() {
  if (reference_initialized) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  auto w = pmy_pack->pmhd->w0;
  auto bcc = pmy_pack->pmhd->bcc0;
  auto &size = pmy_pack->pmb->mb_size;
  const Real gamma = pmy_pack->pmhd->peos->eos_data.gamma;

  Real volume = 0.0, guide_flux = 0.0, enthalpy = 0.0;
  Kokkos::parallel_reduce("antenna_reference_state",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum_volume,
                Real &sum_guide, Real &sum_enthalpy) {
    const Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    sum_volume += dv;
    sum_guide += bcc(m, IBZ, k, j, i)*dv;
    sum_enthalpy += (w(m, IDN, k, j, i) + gamma*w(m, IEN, k, j, i))*dv;
  }, Kokkos::Sum<Real>(volume), Kokkos::Sum<Real>(guide_flux),
     Kokkos::Sum<Real>(enthalpy));

#if MPI_PARALLEL_ENABLED
  Real local[3] = {volume, guide_flux, enthalpy};
  Real global[3];
  MPI_Allreduce(local, global, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  volume = global[0];
  guide_flux = global[1];
  enthalpy = global[2];
#endif

  if (volume <= 0.0 || enthalpy <= 0.0) {
    AntennaFatal("Cannot initialize antenna reference state from nonpositive volume or "
                 "enthalpy");
  }
  const Real b0 = guide_flux/volume;
  const Real w0 = enthalpy/volume;
  magnetization_reference = SQR(b0)/w0;
  if (!fixed_alfven_speed) {
    alfven_speed_reference = std::sqrt(SQR(b0)/(w0 + SQR(b0)));
  }
  if (alfven_speed_reference <= 0.0) {
    AntennaFatal("The antenna requires a nonzero mean guide field");
  }
  const Real box_length = pmy_pack->pmesh->mesh_size.x1max
                          - pmy_pack->pmesh->mesh_size.x1min;
  if (zhdankin_amplitude) {
    // Zhdankin et al. set |a_j|=B0 L/(8 pi) and multiply the current by
    // 2 pi/L^2, giving a component amplitude B0/(4 L).  Compensating the two
    // discrete curl symbols makes that current amplitude resolution independent.
    const Real dx = box_length/pmy_pack->pmesh->mesh_indcs.nx1;
    const Real q = std::sin(2.0*M_PI*dx/box_length)/dx;
    const Real baseline = std::abs(b0)/(4.0*box_length*q*q);
    for (int family = 0; family < num_families; ++family) {
      apar_rms[family] = amplitude_fraction[family]*baseline;
    }
  }
  angular_frequency_reference = 2.0*M_PI*alfven_speed_reference
                                /(std::sqrt(3.0)*box_length);
  reference_initialized = true;
}

//----------------------------------------------------------------------------------------
//! \brief Draw the stationary complex Gaussian state, or start exactly from zero.

void AntennaDriver::InitializeModeState() {
  constexpr Real inv_sqrt_two = 0.70710678118654752440;
  for (int family = 0; family < num_families; ++family) {
    for (int n = 0; n < num_modes; ++n) {
      for (int q = 0; q < num_quadratures; ++q) {
        mode_state.h_view(family, n, q) = stationary_initial_state
            ? apar_rms[family]*inv_sqrt_two*AntennaGaussian(&rstate) : 0.0;
      }
    }
  }
  mode_state.template modify<HostMemSpace>();
  mode_state.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \brief Apply the exact finite-timestep transition of the complex oscillating OU mode.

void AntennaDriver::AdvanceModeState(Real dt) {
  constexpr Real inv_sqrt_two = 0.70710678118654752440;
  mode_state.template sync<HostMemSpace>();
  const Real gamma_rate = decorrelation_factor*angular_frequency_reference;
  const Real decay = std::exp(-gamma_rate*dt);
  const Real noise_fraction = std::sqrt(std::max(0.0, 1.0 - decay*decay));
  for (int family = 0; family < num_families; ++family) {
    const Real sign = (family == 0) ? 1.0 : -1.0;
    const Real theta = sign*frequency_factor*angular_frequency_reference*dt;
    const Real cosine = std::cos(theta);
    const Real sine = std::sin(theta);
    for (int n = 0; n < num_modes; ++n) {
      const Real old_real = mode_state.h_view(family, n, 0);
      const Real old_imag = mode_state.h_view(family, n, 1);
      const Real noise_scale = apar_rms[family]*noise_fraction*inv_sqrt_two;
      mode_state.h_view(family, n, 0) = decay*(cosine*old_real + sine*old_imag)
          + noise_scale*AntennaGaussian(&rstate);
      mode_state.h_view(family, n, 1) = decay*(cosine*old_imag - sine*old_real)
          + noise_scale*AntennaGaussian(&rstate);
    }
  }
  mode_state.template modify<HostMemSpace>();
  mode_state.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \brief Build J=discrete curl curl(alpha zhat) on cells and compatible E faces.

void AntennaDriver::SynthesizeCurrent() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto state = mode_state;
  auto wavevector = mode_wavevector;
  auto jcell = current;
  auto jface = current_face;

  par_for("antenna_current_cell", DevExeSpace(), 0, nmb - 1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                              size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                              size.d_view(m).x3max);
    Real current1 = 0.0, current2 = 0.0, current3 = 0.0;
    for (int n = 0; n < num_modes; ++n) {
      const Real k1 = wavevector.d_view(n, 0);
      const Real k2 = wavevector.d_view(n, 1);
      const Real k3 = wavevector.d_view(n, 2);
      const Real q1 = sin(k1*size.d_view(m).dx1)/size.d_view(m).dx1;
      const Real q2 = sin(k2*size.d_view(m).dx2)/size.d_view(m).dx2;
      const Real q3 = sin(k3*size.d_view(m).dx3)/size.d_view(m).dx3;
      const Real phase = k1*x + k2*y + k3*z;
      Real alpha = 0.0;
      for (int family = 0; family < num_families; ++family) {
        alpha += state.d_view(family, n, 0)*cos(phase)
                 - state.d_view(family, n, 1)*sin(phase);
      }
      current1 -= q1*q3*alpha;
      current2 -= q2*q3*alpha;
      current3 += (q1*q1 + q2*q2)*alpha;
    }
    jcell(m, 0, k, j, i) = current1;
    jcell(m, 1, k, j, i) = current2;
    jcell(m, 2, k, j, i) = current3;
  });

  par_for("antenna_current_face1", DevExeSpace(), 0, nmb - 1, ks, ke, js, je,
          is, ie + 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = LeftEdgeX(i - is, nx1, size.d_view(m).x1min,
                            size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                              size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                              size.d_view(m).x3max);
    Real value = 0.0;
    for (int n = 0; n < num_modes; ++n) {
      const Real k1 = wavevector.d_view(n, 0);
      const Real k2 = wavevector.d_view(n, 1);
      const Real k3 = wavevector.d_view(n, 2);
      const Real q1 = sin(k1*size.d_view(m).dx1)/size.d_view(m).dx1;
      const Real q3 = sin(k3*size.d_view(m).dx3)/size.d_view(m).dx3;
      const Real phase = k1*x + k2*y + k3*z;
      Real alpha = 0.0;
      for (int family = 0; family < num_families; ++family) {
        alpha += state.d_view(family, n, 0)*cos(phase)
                 - state.d_view(family, n, 1)*sin(phase);
      }
      value -= q1*q3*alpha/cos(0.5*k1*size.d_view(m).dx1);
    }
    jface.x1f(m, k, j, i) = value;
  });

  par_for("antenna_current_face2", DevExeSpace(), 0, nmb - 1, ks, ke, js, je + 1,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
    const Real y = LeftEdgeX(j - js, nx2, size.d_view(m).x2min,
                            size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                              size.d_view(m).x3max);
    Real value = 0.0;
    for (int n = 0; n < num_modes; ++n) {
      const Real k1 = wavevector.d_view(n, 0);
      const Real k2 = wavevector.d_view(n, 1);
      const Real k3 = wavevector.d_view(n, 2);
      const Real q2 = sin(k2*size.d_view(m).dx2)/size.d_view(m).dx2;
      const Real q3 = sin(k3*size.d_view(m).dx3)/size.d_view(m).dx3;
      const Real phase = k1*x + k2*y + k3*z;
      Real alpha = 0.0;
      for (int family = 0; family < num_families; ++family) {
        alpha += state.d_view(family, n, 0)*cos(phase)
                 - state.d_view(family, n, 1)*sin(phase);
      }
      value -= q2*q3*alpha/cos(0.5*k2*size.d_view(m).dx2);
    }
    jface.x2f(m, k, j, i) = value;
  });

  par_for("antenna_current_face3", DevExeSpace(), 0, nmb - 1, ks, ke + 1, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                              size.d_view(m).x2max);
    const Real z = LeftEdgeX(k - ks, nx3, size.d_view(m).x3min,
                            size.d_view(m).x3max);
    Real value = 0.0;
    for (int n = 0; n < num_modes; ++n) {
      const Real k1 = wavevector.d_view(n, 0);
      const Real k2 = wavevector.d_view(n, 1);
      const Real k3 = wavevector.d_view(n, 2);
      const Real q1 = sin(k1*size.d_view(m).dx1)/size.d_view(m).dx1;
      const Real q2 = sin(k2*size.d_view(m).dx2)/size.d_view(m).dx2;
      const Real phase = k1*x + k2*y + k3*z;
      Real alpha = 0.0;
      for (int family = 0; family < num_families; ++family) {
        alpha += state.d_view(family, n, 0)*cos(phase)
                 - state.d_view(family, n, 1)*sin(phase);
      }
      value += (q1*q1 + q2*q2)*alpha/cos(0.5*k3*size.d_view(m).dx3);
    }
    jface.x3f(m, k, j, i) = value;
  });

  Real volume = 0.0, current_squared = 0.0;
  Kokkos::parallel_reduce("antenna_current_rms",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum_volume, Real &sum_j2) {
    const Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    sum_volume += dv;
    sum_j2 += dv*(SQR(jcell(m, 0, k, j, i)) + SQR(jcell(m, 1, k, j, i))
                  + SQR(jcell(m, 2, k, j, i)));
  }, Kokkos::Sum<Real>(volume), Kokkos::Sum<Real>(current_squared));

  Real max_divergence = 0.0;
  Kokkos::parallel_reduce("antenna_current_divergence",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum) {
    const Real divergence =
        (jface.x1f(m, k, j, i + 1) - jface.x1f(m, k, j, i))
            /size.d_view(m).dx1
        + (jface.x2f(m, k, j + 1, i) - jface.x2f(m, k, j, i))
            /size.d_view(m).dx2
        + (jface.x3f(m, k + 1, j, i) - jface.x3f(m, k, j, i))
            /size.d_view(m).dx3;
    maximum = fmax(maximum, fabs(divergence));
  }, Kokkos::Max<Real>(max_divergence));

  Real max_face_cell_mismatch = 0.0;
  Kokkos::parallel_reduce("antenna_face_cell_compatibility",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum) {
    const Real face_average1 =
        0.5*(jface.x1f(m, k, j, i) + jface.x1f(m, k, j, i + 1));
    const Real face_average2 =
        0.5*(jface.x2f(m, k, j, i) + jface.x2f(m, k, j + 1, i));
    const Real face_average3 =
        0.5*(jface.x3f(m, k, j, i) + jface.x3f(m, k + 1, j, i));
    maximum = fmax(maximum, fabs(face_average1 - jcell(m, 0, k, j, i)));
    maximum = fmax(maximum, fabs(face_average2 - jcell(m, 1, k, j, i)));
    maximum = fmax(maximum, fabs(face_average3 - jcell(m, 2, k, j, i)));
  }, Kokkos::Max<Real>(max_face_cell_mismatch));

#if MPI_PARALLEL_ENABLED
  Real local_sum[2] = {volume, current_squared};
  Real global_sum[2];
  MPI_Allreduce(local_sum, global_sum, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  volume = global_sum[0];
  current_squared = global_sum[1];
  MPI_Allreduce(MPI_IN_PLACE, &max_divergence, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_face_cell_mismatch, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
#endif
  last_current_rms = (volume > 0.0) ? std::sqrt(current_squared/volume) : 0.0;
  last_divergence = max_divergence;
  last_face_cell_mismatch = max_face_cell_mismatch;
  current_ready = true;
}

//----------------------------------------------------------------------------------------
//! \brief Finalize host mode coefficients and reference data loaded from a restart.

void AntennaDriver::MarkRestarted() {
  mode_state.template modify<HostMemSpace>();
  mode_state.template sync<DevExeSpace>();
  initialized = true;
  reference_initialized = true;
  SynthesizeCurrent();
}

//----------------------------------------------------------------------------------------
//! \brief Update coefficients once per full step and synthesize the held antenna current.

TaskStatus AntennaDriver::UpdateAntenna(Driver *pdrive, int stage) {
  (void)pdrive;
  (void)stage;
  InitializeReferenceState();
  if (!initialized) {
    InitializeModeState();
    initialized = true;
  } else {
    AdvanceModeState(pmy_pack->pmesh->dt);
  }
  SynthesizeCurrent();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Apply S_E=-J_ant and -F^{nu lambda}J_ant,lambda at one RK stage.

TaskStatus AntennaDriver::ApplyAntenna(Driver *pdrive, int stage) {
  if (!apply_source) return TaskStatus::complete;
  if (!current_ready) {
    AntennaFatal("Antenna source task executed before current synthesis");
  }

  auto *pmhd = pmy_pack->pmhd;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto u = pmhd->u0;
  auto w = pmhd->w0;
  auto bcc = pmhd->bcc0;
  auto jant = current;
  auto jface = current_face;
  auto eface = pmhd->e0;
  const bool use_electric_ct = pmhd->use_electric_ct;
  const Real beta_dt = pdrive->beta[stage - 1]*pmy_pack->pmesh->dt;

  Real power = 0.0, momentum1 = 0.0, momentum2 = 0.0, momentum3 = 0.0;
  Kokkos::parallel_reduce("antenna_source_diagnostics",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum_power, Real &sum_m1,
                Real &sum_m2, Real &sum_m3) {
    const Real current1 = use_electric_ct
        ? 0.5*(jface.x1f(m, k, j, i) + jface.x1f(m, k, j, i + 1))
        : jant(m, 0, k, j, i);
    const Real current2 = use_electric_ct
        ? 0.5*(jface.x2f(m, k, j, i) + jface.x2f(m, k, j + 1, i))
        : jant(m, 1, k, j, i);
    const Real current3 = use_electric_ct
        ? 0.5*(jface.x3f(m, k, j, i) + jface.x3f(m, k + 1, j, i))
        : jant(m, 2, k, j, i);
    const Real se1 = -current1;
    const Real se2 = -current2;
    const Real se3 = -current3;
    const Real e1 = w(m, srrmhd::IRE1, k, j, i);
    const Real e2 = w(m, srrmhd::IRE2, k, j, i);
    const Real e3 = w(m, srrmhd::IRE3, k, j, i);
    const Real b1 = bcc(m, IBX, k, j, i);
    const Real b2 = bcc(m, IBY, k, j, i);
    const Real b3 = bcc(m, IBZ, k, j, i);
    const Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    sum_power += (se1*e1 + se2*e2 + se3*e3)*dv;
    sum_m1 += (se2*b3 - se3*b2)*dv;
    sum_m2 += (se3*b1 - se1*b3)*dv;
    sum_m3 += (se1*b2 - se2*b1)*dv;
  }, Kokkos::Sum<Real>(power), Kokkos::Sum<Real>(momentum1),
     Kokkos::Sum<Real>(momentum2), Kokkos::Sum<Real>(momentum3));

#if MPI_PARALLEL_ENABLED
  Real local[4] = {power, momentum1, momentum2, momentum3};
  Real global[4];
  MPI_Allreduce(local, global, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  power = global[0];
  momentum1 = global[1];
  momentum2 = global[2];
  momentum3 = global[3];
#endif
  last_power = power;
  last_momentum1 = momentum1;
  last_momentum2 = momentum2;
  last_momentum3 = momentum3;

  if (stage == 1) {
    injected_energy_start = injected_energy;
    injected_momentum1_start = injected_momentum1;
    injected_momentum2_start = injected_momentum2;
    injected_momentum3_start = injected_momentum3;
  }
  const Real gam0 = pdrive->gam0[stage - 1];
  const Real gam1 = pdrive->gam1[stage - 1];
  injected_energy = gam0*injected_energy + gam1*injected_energy_start
                    + beta_dt*last_power;
  injected_momentum1 = gam0*injected_momentum1 + gam1*injected_momentum1_start
                       + beta_dt*last_momentum1;
  injected_momentum2 = gam0*injected_momentum2 + gam1*injected_momentum2_start
                       + beta_dt*last_momentum2;
  injected_momentum3 = gam0*injected_momentum3 + gam1*injected_momentum3_start
                       + beta_dt*last_momentum3;

  par_for("antenna_total_four_force", DevExeSpace(), 0, nmb - 1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real current1 = use_electric_ct
        ? 0.5*(jface.x1f(m, k, j, i) + jface.x1f(m, k, j, i + 1))
        : jant(m, 0, k, j, i);
    const Real current2 = use_electric_ct
        ? 0.5*(jface.x2f(m, k, j, i) + jface.x2f(m, k, j + 1, i))
        : jant(m, 1, k, j, i);
    const Real current3 = use_electric_ct
        ? 0.5*(jface.x3f(m, k, j, i) + jface.x3f(m, k + 1, j, i))
        : jant(m, 2, k, j, i);
    const Real se1 = -current1;
    const Real se2 = -current2;
    const Real se3 = -current3;
    const Real e1 = w(m, srrmhd::IRE1, k, j, i);
    const Real e2 = w(m, srrmhd::IRE2, k, j, i);
    const Real e3 = w(m, srrmhd::IRE3, k, j, i);
    const Real b1 = bcc(m, IBX, k, j, i);
    const Real b2 = bcc(m, IBY, k, j, i);
    const Real b3 = bcc(m, IBZ, k, j, i);
    u(m, IM1, k, j, i) += beta_dt*(se2*b3 - se3*b2);
    u(m, IM2, k, j, i) += beta_dt*(se3*b1 - se1*b3);
    u(m, IM3, k, j, i) += beta_dt*(se1*b2 - se2*b1);
    u(m, IEN, k, j, i) += beta_dt*(se1*e1 + se2*e2 + se3*e3);
    if (!use_electric_ct) {
      u(m, srrmhd::IRE1, k, j, i) += beta_dt*se1;
      u(m, srrmhd::IRE2, k, j, i) += beta_dt*se2;
      u(m, srrmhd::IRE3, k, j, i) += beta_dt*se3;
    }
  });

  if (use_electric_ct) {
    par_for("antenna_face_e1", DevExeSpace(), 0, nmb - 1, ks, ke, js, je,
            is, ie + 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      eface.x1f(m, k, j, i) -= beta_dt*jface.x1f(m, k, j, i);
    });
    par_for("antenna_face_e2", DevExeSpace(), 0, nmb - 1, ks, ke, js, je + 1,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      eface.x2f(m, k, j, i) -= beta_dt*jface.x2f(m, k, j, i);
    });
    par_for("antenna_face_e3", DevExeSpace(), 0, nmb - 1, ks, ke + 1, js, je,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      eface.x3f(m, k, j, i) -= beta_dt*jface.x3f(m, k, j, i);
    });
    par_for("antenna_face_to_cell_e", DevExeSpace(), 0, nmb - 1, ks, ke, js, je,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real e1, e2, e3;
      srrmhd::ElectricFaceToCell(eface, m, k, j, i, e1, e2, e3);
      u(m, srrmhd::IRE1, k, j, i) = e1;
      u(m, srrmhd::IRE2, k, j, i) = e2;
      u(m, srrmhd::IRE3, k, j, i) = e3;
    });
  }

  return TaskStatus::complete;
}
