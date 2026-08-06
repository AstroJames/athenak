//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file antenna_driver.cpp
//! \brief Electromagnetic oscillating-Langevin antenna for resistive SRMHD.

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "bvals/bvals.hpp"
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
#include "spectral_mode_catalog.hpp"

namespace {

[[noreturn]] void AntennaFatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

bool NearlyEqual(Real a, Real b) {
  const Real scale = std::max(std::abs(a), std::abs(b));
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

void HashByte(std::uint64_t &hash, std::uint8_t value) {
  constexpr std::uint64_t fnv_prime = 1099511628211ULL;
  hash ^= value;
  hash *= fnv_prime;
}

void HashUint32(std::uint64_t &hash, std::uint32_t value) {
  for (int byte = 0; byte < 4; ++byte) {
    HashByte(hash, static_cast<std::uint8_t>((value >> (8*byte)) & 0xffU));
  }
}

void HashUint64(std::uint64_t &hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    HashByte(hash, static_cast<std::uint8_t>((value >> (8*byte)) & 0xffULL));
  }
}

void HashString(std::uint64_t &hash, const std::string &value) {
  for (char character : value) {
    HashByte(hash, static_cast<std::uint8_t>(character));
  }
  HashByte(hash, 0);
}

void HashReal(std::uint64_t &hash, Real value) {
  if constexpr (sizeof(Real) == sizeof(std::uint64_t)) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    HashUint64(hash, bits);
  } else {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    HashUint32(hash, bits);
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \brief Construct a discrete, divergence-free oscillating-OU current antenna.

AntennaDriver::AntennaDriver(MeshBlockPack *pp, ParameterInput *pin) :
    current("antenna_current", 1, 1, 1, 1, 1),
    current_face("antenna_current_face", 1, 1, 1, 1),
    applied_current("antenna_applied_current", 1, 1, 1, 1, 1),
    mode_state("antenna_mode_state", num_families, 1, num_quadratures),
    mode_wavevector("antenna_mode_wavevector", 1, 3),
    mode_polarization("antenna_mode_polarization", 1, 3),
    mode_alpha_weight("antenna_mode_alpha_weight", 1),
    stage_task_id(0),
    pmy_pack(pp) {
  if (pmy_pack->pmhd == nullptr || !pmy_pack->pmhd->is_resistive_rel
      || !pmy_pack->pcoord->is_special_relativistic) {
    AntennaFatal("<antenna_driving> requires special-relativistic resistive MHD with "
                 "an evolved electric field");
  }
  if (!pmy_pack->pmesh->three_d) {
    AntennaFatal("The Fourier antenna requires a three-dimensional mesh");
  }
  if (pmy_pack->pmesh->multilevel) {
    AntennaFatal("The Fourier antenna supports uniform meshes only");
  }
  const std::string model = pin->GetOrAddString(
      "antenna_driving", "model", "fourier_oscillating_ou");
  if (model != "fourier_oscillating_ou") {
    AntennaFatal("Unknown <antenna_driving>/model='" + model
                 + "'; the implemented model is fourier_oscillating_ou");
  }
  auto &mesh_indcs = pmy_pack->pmesh->mesh_indcs;
  const int mesh_cells[3] = {
      mesh_indcs.nx1, mesh_indcs.nx2, mesh_indcs.nx3};
  for (int d = 0; d < 3; ++d) {
    if (mesh_cells[d] < 4) {
      AntennaFatal("The Fourier antenna requires at least four cells per direction");
    }
  }

  mode_set = pin->GetOrAddString("antenna_driving", "mode_set", "zhdankin8");
  const std::string guide_axis_name = pin->GetOrAddString(
      "antenna_driving", "guide_axis", "z");
  const auto parsed_guide_axis = spectral_modes::ParseGuideAxis(guide_axis_name);
  guide_axis = static_cast<int>(parsed_guide_axis);
  if (guide_axis < 0 || guide_axis > 2) {
    AntennaFatal("<antenna_driving>/guide_axis must be x1, x2, or x3");
  }
  const std::string current_geometry = pin->GetOrAddString(
      "antenna_driving", "current_geometry", "apar_double_curl");
  if (current_geometry != "apar_double_curl") {
    AntennaFatal("The antenna implementation requires "
                 "current_geometry=apar_double_curl");
  }

  const char *boundary_names[6] = {
      "ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"};
  for (const char *name : boundary_names) {
    if (pin->GetString("mesh", name) != "periodic") {
      AntennaFatal("The Fourier antenna requires periodic boundaries in all directions");
    }
  }

  const Real box_length[3] = {
      pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min,
      pmy_pack->pmesh->mesh_size.x2max - pmy_pack->pmesh->mesh_size.x2min,
      pmy_pack->pmesh->mesh_size.x3max - pmy_pack->pmesh->mesh_size.x3min};
  for (int d = 0; d < 3; ++d) {
    if (!std::isfinite(box_length[d]) || box_length[d] <= 0.0) {
      AntennaFatal("Antenna box lengths must be finite and positive");
    }
  }

  std::vector<spectral_modes::Mode> selected_modes;
  if (mode_set == "zhdankin8") {
    if (mesh_indcs.nx1 != mesh_indcs.nx2 || mesh_indcs.nx1 != mesh_indcs.nx3
        || !NearlyEqual(box_length[0], box_length[1])
        || !NearlyEqual(box_length[0], box_length[2])) {
      AntennaFatal("mode_set=zhdankin8 requires a cubic domain and equal resolution");
    }
    selected_modes = spectral_modes::Zhdankin8(parsed_guide_axis);
  } else if (mode_set == "anisotropic_band") {
    const int nperp_min = pin->GetOrAddInteger(
        "antenna_driving", "nperp_min", 1);
    const int nperp_max = pin->GetOrAddInteger(
        "antenna_driving", "nperp_max", 2);
    const int nparallel_min = pin->GetOrAddInteger(
        "antenna_driving", "nparallel_min", 1);
    const int nparallel_max = pin->GetOrAddInteger(
        "antenna_driving", "nparallel_max", 1);
    if (nperp_min < 1 || nperp_max < nperp_min || nparallel_min < 1
        || nparallel_max < nparallel_min) {
      AntennaFatal("Invalid anisotropic antenna mode bounds");
    }
    const auto transverse = spectral_modes::TransverseAxes(parsed_guide_axis);
    if (nparallel_max >= (mesh_cells[guide_axis] + 1)/2
        || nperp_max >= (mesh_cells[transverse[0]] + 1)/2
        || nperp_max >= (mesh_cells[transverse[1]] + 1)/2) {
      AntennaFatal("Anisotropic antenna bounds reach an unresolved or Nyquist mode");
    }
    selected_modes = spectral_modes::AnisotropicBand(
        parsed_guide_axis, nperp_min, nperp_max, nparallel_min, nparallel_max);
  } else {
    AntennaFatal("Unknown <antenna_driving>/mode_set='" + mode_set + "'");
  }
  if (selected_modes.empty()) AntennaFatal("The antenna mode set is empty");

  current_envelope = pin->GetOrAddString(
      "antenna_driving", "current_envelope", "band");
  if (current_envelope == "parabolic") {
    current_parabola_peak = pin->GetOrAddReal(
        "antenna_driving", "current_parabola_peak", 1.0);
    current_parabola_width = pin->GetOrAddReal(
        "antenna_driving", "current_parabola_width", 1.0);
    if (!std::isfinite(current_parabola_peak)
        || !std::isfinite(current_parabola_width)
        || current_parabola_width <= 0.0) {
      AntennaFatal("Parabolic-envelope parameters must be finite, with positive width");
    }
  } else if (current_envelope == "powerlaw") {
    current_exponent_perp = pin->GetOrAddReal(
        "antenna_driving", "current_exponent_perp", 0.0);
    current_exponent_parallel = pin->GetOrAddReal(
        "antenna_driving", "current_exponent_parallel", 0.0);
    current_reference_perp = pin->GetOrAddReal(
        "antenna_driving", "current_reference_perp", 1.0);
    current_reference_parallel = pin->GetOrAddReal(
        "antenna_driving", "current_reference_parallel", 1.0);
    if (!std::isfinite(current_exponent_perp)
        || !std::isfinite(current_exponent_parallel)
        || !std::isfinite(current_reference_perp)
        || !std::isfinite(current_reference_parallel)
        || current_reference_perp <= 0.0 || current_reference_parallel <= 0.0) {
      AntennaFatal("Power-law antenna parameters must be finite, with positive "
                   "reference indices");
    }
  } else if (current_envelope != "band") {
    AntennaFatal("current_envelope must be band, parabolic, or powerlaw");
  }

  apply_source = pin->GetOrAddBoolean("antenna_driving", "apply_source", true);
  const std::string default_frequency = (mode_set == "zhdankin8")
      ? "zhdankin2018" : "alfven_parallel";
  frequency_model = pin->GetOrAddString(
      "antenna_driving", "frequency_model", default_frequency);
  if (frequency_model == "alfven_parallel" || frequency_model == "alfven") {
    alfven_parallel_frequency = true;
    frequency_model = "alfven_parallel";
  } else if (frequency_model != "zhdankin2018") {
    AntennaFatal("<antenna_driving>/frequency_model must be zhdankin2018 or "
                 "alfven_parallel");
  }
  if (frequency_model == "zhdankin2018" && mode_set != "zhdankin8") {
    AntennaFatal("frequency_model=zhdankin2018 is restricted to mode_set=zhdankin8");
  }
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
    if (mode_set != "zhdankin8") {
      AntennaFatal("amplitude_normalization=zhdankin requires mode_set=zhdankin8");
    }
    zhdankin_amplitude = true;
    amplitude_fraction[0] = pin->GetOrAddReal(
        "antenna_driving", "amplitude_fraction_plus", 1.0);
    amplitude_fraction[1] = pin->GetOrAddReal(
        "antenna_driving", "amplitude_fraction_minus", 1.0);
  } else {
    AntennaFatal("<antenna_driving>/amplitude_normalization must be apar_rms or "
                 "zhdankin");
  }
  if (!std::isfinite(frequency_factor) || !std::isfinite(decorrelation_factor)
      || !std::isfinite(apar_rms[0]) || !std::isfinite(apar_rms[1])
      || !std::isfinite(amplitude_fraction[0])
      || !std::isfinite(amplitude_fraction[1])
      || frequency_factor < 0.0 || decorrelation_factor < 0.0
      || apar_rms[0] < 0.0 || apar_rms[1] < 0.0
      || amplitude_fraction[0] < 0.0 || amplitude_fraction[1] < 0.0) {
    AntennaFatal("Antenna frequency, decorrelation, and RMS amplitudes must be finite "
                 "and nonnegative");
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
    if (!std::isfinite(alfven_speed_reference)
        || alfven_speed_reference <= 0.0 || alfven_speed_reference >= 1.0) {
      AntennaFatal("A fixed antenna Alfven speed must be finite and lie strictly "
                   "between zero and one");
    }
  } else if (va_reference != "initial_mean") {
    AntennaFatal("<antenna_driving>/va_reference must be initial_mean or fixed");
  }

  int seed = pin->GetOrAddInteger("antenna_driving", "seed", 210989);
  if (seed == 0) seed = 1;
  rstate = {};
  rstate.idum = -std::abs(static_cast<int64_t>(seed));

  num_modes = static_cast<int>(selected_modes.size());
  mode_integer.resize(num_modes);
  mode_frequency_reference.assign(num_modes, 0.0);
  Kokkos::realloc(mode_state, num_families, num_modes, num_quadratures);
  Kokkos::realloc(mode_wavevector, num_modes, 3);
  Kokkos::realloc(mode_polarization, num_modes, 3);
  Kokkos::realloc(mode_alpha_weight, num_modes);

  std::vector<Real> raw_alpha_weight(num_modes);
  Real signed_weight_sum = 0.0;
  Real signed_weight_squared_sum = 0.0;
  for (int n = 0; n < num_modes; ++n) {
    mode_integer[n] = selected_modes[n].n;
    Real q[3];
    for (int d = 0; d < 3; ++d) {
      if (2*std::abs(mode_integer[n][d]) >= mesh_cells[d]) {
        AntennaFatal("Antenna mode set contains a Nyquist or unresolved mode");
      }
      const Real k = 2.0*M_PI*mode_integer[n][d]/box_length[d];
      const Real dx = box_length[d]/mesh_cells[d];
      mode_wavevector.h_view(n, d) = k;
      q[d] = std::sin(k*dx)/dx;
    }
    const Real q_squared = SQR(q[0]) + SQR(q[1]) + SQR(q[2]);
    if (!std::isfinite(q_squared) || q_squared <= 0.0) {
      AntennaFatal("Antenna mode set produced a non-finite or zero wavevector");
    }
    Real polarization_squared = 0.0;
    for (int d = 0; d < 3; ++d) {
      const Real polarization = ((d == guide_axis) ? q_squared : 0.0)
                                - q[d]*q[guide_axis];
      mode_polarization.h_view(n, d) = polarization;
      polarization_squared += SQR(polarization);
    }
    if (!std::isfinite(polarization_squared)) {
      AntennaFatal("Antenna mode set produced a non-finite polarization");
    }
    if (polarization_squared
        <= SQR(std::numeric_limits<Real>::epsilon()*q_squared)) {
      AntennaFatal("Antenna mode set contains a zero-current mode");
    }

    const int t1 = (guide_axis == 0) ? 1 : 0;
    const int t2 = (guide_axis == 2) ? 1 : 2;
    const Real nperp = std::sqrt(
        static_cast<Real>(SQR(mode_integer[n][t1]) + SQR(mode_integer[n][t2])));
    const Real nparallel = std::abs(mode_integer[n][guide_axis]);
    Real envelope = 1.0;
    if (current_envelope == "parabolic") {
      envelope = std::max(
          0.0, 1.0 - SQR((nperp - current_parabola_peak)/current_parabola_width));
    } else if (current_envelope == "powerlaw") {
      envelope = std::pow(nperp/current_reference_perp, -current_exponent_perp)
                 *std::pow(nparallel/current_reference_parallel,
                           -current_exponent_parallel);
    }
    if (!std::isfinite(envelope) || envelope < 0.0
        || !std::isfinite(selected_modes[n].signed_degeneracy)
        || selected_modes[n].signed_degeneracy <= 0.0) {
      AntennaFatal("Antenna envelope and mode degeneracy must be finite and "
                   "nonnegative");
    }
    raw_alpha_weight[n] = envelope/std::sqrt(polarization_squared);
    if (!std::isfinite(raw_alpha_weight[n])) {
      AntennaFatal("Antenna current-envelope compensation produced a non-finite "
                   "mode weight");
    }
    if (envelope > 0.0) {
      signed_weight_sum += selected_modes[n].signed_degeneracy;
    }
    signed_weight_squared_sum += selected_modes[n].signed_degeneracy
                                 *SQR(raw_alpha_weight[n]);
  }
  if (!std::isfinite(signed_weight_sum)
      || !std::isfinite(signed_weight_squared_sum)
      || signed_weight_sum <= 0.0 || signed_weight_squared_sum <= 0.0) {
    AntennaFatal("The selected current envelope is zero for every antenna mode");
  }
  const bool preserve_zhdankin_weights =
      (mode_set == "zhdankin8" && current_envelope == "band");
  const Real alpha_normalization = preserve_zhdankin_weights
      ? 1.0 : std::sqrt(signed_weight_sum/signed_weight_squared_sum);
  if (!std::isfinite(alpha_normalization) || alpha_normalization <= 0.0) {
    AntennaFatal("Antenna current-envelope normalization must be finite and positive");
  }
  for (int n = 0; n < num_modes; ++n) {
    mode_alpha_weight.h_view(n) = preserve_zhdankin_weights
        ? 1.0 : alpha_normalization*raw_alpha_weight[n];
    if (!std::isfinite(mode_alpha_weight.h_view(n))
        || mode_alpha_weight.h_view(n) < 0.0) {
      AntennaFatal("Antenna normalized mode weights must be finite and nonnegative");
    }
  }
  mode_wavevector.template modify<HostMemSpace>();
  mode_wavevector.template sync<DevExeSpace>();
  mode_polarization.template modify<HostMemSpace>();
  mode_polarization.template sync<DevExeSpace>();
  mode_alpha_weight.template modify<HostMemSpace>();
  mode_alpha_weight.template sync<DevExeSpace>();
  ComputeModeSetHash();

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
  Kokkos::realloc(applied_current, nmb, 3, n3, n2, n1);
  Kokkos::realloc(current_face.x1f, nmb, n3, n2, n1 + 1);
  Kokkos::realloc(current_face.x2f, nmb, n3, n2 + 1, n1);
  Kokkos::realloc(current_face.x3f, nmb, n3 + 1, n2, n1);
  Kokkos::deep_copy(current, 0.0);
  Kokkos::deep_copy(applied_current, 0.0);
  Kokkos::deep_copy(current_face.x1f, 0.0);
  Kokkos::deep_copy(current_face.x2f, 0.0);
  Kokkos::deep_copy(current_face.x3f, 0.0);

  pbval_current = new MeshBoundaryValuesCC(pp, pin, false);
  pbval_current->InitializeBuffers(3);
}

AntennaDriver::~AntennaDriver() {
  if (pbval_current != nullptr) delete pbval_current;
}

//----------------------------------------------------------------------------------------
//! \brief Hash the ordered mode set and stochastic dynamics for restart validation.

void AntennaDriver::ComputeModeSetHash() {
  constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
  mode_set_hash = fnv_offset_basis;
  HashByte(mode_set_hash, 1);  // antenna mode-set hash schema
  HashString(mode_set_hash, mode_set);
  HashString(mode_set_hash, frequency_model);
  HashString(mode_set_hash, current_envelope);
  HashUint32(mode_set_hash, static_cast<std::uint32_t>(guide_axis));
  HashUint32(mode_set_hash, static_cast<std::uint32_t>(num_modes));
  HashReal(mode_set_hash, frequency_factor);
  HashReal(mode_set_hash, decorrelation_factor);
  HashReal(mode_set_hash, current_parabola_peak);
  HashReal(mode_set_hash, current_parabola_width);
  HashReal(mode_set_hash, current_exponent_perp);
  HashReal(mode_set_hash, current_exponent_parallel);
  HashReal(mode_set_hash, current_reference_perp);
  HashReal(mode_set_hash, current_reference_parallel);
  const int mesh_cells[3] = {pmy_pack->pmesh->mesh_indcs.nx1,
                             pmy_pack->pmesh->mesh_indcs.nx2,
                             pmy_pack->pmesh->mesh_indcs.nx3};
  const Real box_lengths[3] = {
      pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min,
      pmy_pack->pmesh->mesh_size.x2max - pmy_pack->pmesh->mesh_size.x2min,
      pmy_pack->pmesh->mesh_size.x3max - pmy_pack->pmesh->mesh_size.x3min};
  for (int d = 0; d < 3; ++d) {
    HashUint32(mode_set_hash, static_cast<std::uint32_t>(mesh_cells[d]));
    HashReal(mode_set_hash, box_lengths[d]);
  }
  for (int n = 0; n < num_modes; ++n) {
    for (int d = 0; d < 3; ++d) {
      HashUint32(mode_set_hash,
                 static_cast<std::uint32_t>(mode_integer[n][d]));
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Return whether the runtime mode set matches the historical v2 record.

bool AntennaDriver::IsLegacyV2ModeSet() const {
  const std::array<std::array<int, 3>, 4> historical = {
      std::array<int, 3>{1, 0, 1},
      std::array<int, 3>{0, 1, 1},
      std::array<int, 3>{-1, 0, 1},
      std::array<int, 3>{0, -1, 1},
  };
  if (guide_axis != 2 || mode_set != "zhdankin8" || num_modes != 4
      || frequency_model != "zhdankin2018" || current_envelope != "band") {
    return false;
  }
  for (int n = 0; n < num_modes; ++n) {
    if (mode_integer[n] != historical[n] || mode_alpha_weight.h_view(n) != 1.0) {
      return false;
    }
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \brief Flatten complex mode coefficients in family-mode-quadrature order.

void AntennaDriver::PackRestartModes(std::vector<Real> &values) {
  mode_state.template sync<HostMemSpace>();
  values.resize(num_families*num_modes*num_quadratures);
  int index = 0;
  for (int family = 0; family < num_families; ++family) {
    for (int mode = 0; mode < num_modes; ++mode) {
      for (int q = 0; q < num_quadratures; ++q) {
        values[index++] = mode_state.h_view(family, mode, q);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Restore flattened complex mode coefficients after record validation.

void AntennaDriver::RestoreRestartModes(const std::vector<Real> &values) {
  const int expected = num_families*num_modes*num_quadratures;
  if (static_cast<int>(values.size()) != expected) {
    AntennaFatal("Antenna restart coefficient count does not match the mode set");
  }
  int index = 0;
  for (int family = 0; family < num_families; ++family) {
    for (int mode = 0; mode < num_modes; ++mode) {
      for (int q = 0; q < num_quadratures; ++q) {
        mode_state.h_view(family, mode, q) = values[index++];
      }
    }
  }
  mode_state.template modify<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \brief Add the coefficient update/current synthesis before the time integrator.

TaskID AntennaDriver::IncludeUpdateTask(std::shared_ptr<TaskList> tl, TaskID start) {
  auto update = tl->AddTask(&AntennaDriver::UpdateAntenna, this, start);
  auto send = tl->AddTask(&AntennaDriver::StartCurrentExchange, this, update);
  auto finish = tl->AddTask(&AntennaDriver::FinishCurrentExchange, this, send);
  return finish;
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

TaskStatus AntennaDriver::InitializeReferenceState() {
  if (reference_initialized) return TaskStatus::complete;

  if (reference_exchange_phase == 0) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, ie = indcs.ie;
    const int js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    const int nmb = pmy_pack->nmb_thispack;
    auto w = pmy_pack->pmhd->w0;
    auto bcc = pmy_pack->pmhd->bcc0;
    auto &size = pmy_pack->pmb->mb_size;
    const Real gamma = pmy_pack->pmhd->peos->eos_data.gamma;

    Real volume = 0.0, magnetic_flux1 = 0.0, magnetic_flux2 = 0.0;
    Real magnetic_flux3 = 0.0, enthalpy = 0.0;
    Kokkos::parallel_reduce("antenna_reference_state",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
            {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum_volume, Real &sum_b1,
                  Real &sum_b2, Real &sum_b3, Real &sum_enthalpy) {
      const Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      sum_volume += dv;
      sum_b1 += bcc(m, IBX, k, j, i)*dv;
      sum_b2 += bcc(m, IBY, k, j, i)*dv;
      sum_b3 += bcc(m, IBZ, k, j, i)*dv;
      sum_enthalpy += (w(m, IDN, k, j, i) + gamma*w(m, IEN, k, j, i))*dv;
    }, Kokkos::Sum<Real>(volume), Kokkos::Sum<Real>(magnetic_flux1),
       Kokkos::Sum<Real>(magnetic_flux2), Kokkos::Sum<Real>(magnetic_flux3),
       Kokkos::Sum<Real>(enthalpy));
    reference_sum_local[0] = volume;
    reference_sum_local[1] = magnetic_flux1;
    reference_sum_local[2] = magnetic_flux2;
    reference_sum_local[3] = magnetic_flux3;
    reference_sum_local[4] = enthalpy;
#if MPI_PARALLEL_ENABLED
    MPI_Iallreduce(reference_sum_local, reference_sum_global, 5, MPI_ATHENA_REAL,
                   MPI_SUM, MPI_COMM_WORLD, &reference_reduction_request);
    reference_exchange_phase = 1;
#else
    for (int n = 0; n < 5; ++n) {
      reference_sum_global[n] = reference_sum_local[n];
    }
    reference_exchange_phase = 2;
#endif
  }
#if MPI_PARALLEL_ENABLED
  if (reference_exchange_phase == 1) {
    int complete = 0;
    MPI_Test(&reference_reduction_request, &complete, MPI_STATUS_IGNORE);
    if (complete == 0) return TaskStatus::incomplete;
    reference_exchange_phase = 2;
  }
#endif
  const Real volume = reference_sum_global[0];
  const Real magnetic_flux1 = reference_sum_global[1];
  const Real magnetic_flux2 = reference_sum_global[2];
  const Real magnetic_flux3 = reference_sum_global[3];
  const Real enthalpy = reference_sum_global[4];
  const int guide_component = guide_axis;

  if (!std::isfinite(volume) || !std::isfinite(enthalpy)
      || volume <= 0.0 || enthalpy <= 0.0) {
    AntennaFatal("Cannot initialize antenna reference state from non-finite or "
                 "nonpositive volume or enthalpy");
  }
  const Real mean_field[3] = {
      magnetic_flux1/volume, magnetic_flux2/volume, magnetic_flux3/volume};
  const Real b0 = mean_field[guide_component];
  if (!std::isfinite(mean_field[0]) || !std::isfinite(mean_field[1])
      || !std::isfinite(mean_field[2]) || b0 <= 0.0) {
    AntennaFatal("The Cartesian antenna guide direction must have finite, positive "
                 "mean field");
  }
  const Real transverse_tolerance =
      1000.0*std::numeric_limits<Real>::epsilon()*std::abs(b0);
  for (int d = 0; d < 3; ++d) {
    if (d != guide_component && std::abs(mean_field[d]) > transverse_tolerance) {
      AntennaFatal("The initial mean magnetic field must align with guide_axis");
    }
  }
  const Real w0 = enthalpy/volume;
  magnetization_reference = SQR(b0)/w0;
  if (!fixed_alfven_speed) {
    alfven_speed_reference = std::sqrt(SQR(b0)/(w0 + SQR(b0)));
  }
  if (!std::isfinite(w0) || !std::isfinite(magnetization_reference)
      || !std::isfinite(alfven_speed_reference) || alfven_speed_reference <= 0.0) {
    AntennaFatal("The antenna reference state must produce finite enthalpy, "
                 "magnetization, and nonzero Alfven speed");
  }
  const Real box_lengths[3] = {
      pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min,
      pmy_pack->pmesh->mesh_size.x2max - pmy_pack->pmesh->mesh_size.x2min,
      pmy_pack->pmesh->mesh_size.x3max - pmy_pack->pmesh->mesh_size.x3min};
  const Real box_length = box_lengths[guide_axis];
  if (zhdankin_amplitude) {
    // In Gaussian units, Zhdankin et al. set |a_j|=B0 L/(8 pi) and multiply
    // the current by 2 pi/L^2, giving J_G=B0_G/(4 L).  AthenaK uses
    // rationalized units: B_G=sqrt(4 pi) B and J=sqrt(4 pi) J_G, hence the
    // corresponding Ampere-source amplitude is J=pi B0/L.  Compensating the
    // two discrete curl symbols makes that amplitude resolution independent.
    const int mesh_cells[3] = {pmy_pack->pmesh->mesh_indcs.nx1,
                               pmy_pack->pmesh->mesh_indcs.nx2,
                               pmy_pack->pmesh->mesh_indcs.nx3};
    const Real dx = box_length/mesh_cells[guide_axis];
    const Real q = std::sin(2.0*M_PI*dx/box_length)/dx;
    const Real baseline = M_PI*std::abs(b0)/(box_length*q*q);
    if (!std::isfinite(baseline)) {
      AntennaFatal("The Zhdankin antenna amplitude is not finite");
    }
    for (int family = 0; family < num_families; ++family) {
      apar_rms[family] = amplitude_fraction[family]*baseline;
    }
  }
  UpdateModeFrequencies();
  reference_initialized = true;
  reference_exchange_phase = 0;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Set the positive modal frequency scales from the frozen Alfven speed.

void AntennaDriver::UpdateModeFrequencies() {
  mode_wavevector.template sync<HostMemSpace>();
  const Real box_lengths[3] = {
      pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min,
      pmy_pack->pmesh->mesh_size.x2max - pmy_pack->pmesh->mesh_size.x2min,
      pmy_pack->pmesh->mesh_size.x3max - pmy_pack->pmesh->mesh_size.x3min};
  angular_frequency_reference =
      2.0*M_PI*alfven_speed_reference/box_lengths[guide_axis];
  if (!alfven_parallel_frequency) {
    angular_frequency_reference /= std::sqrt(3.0);
  }
  for (int n = 0; n < num_modes; ++n) {
    mode_frequency_reference[n] = alfven_parallel_frequency
        ? std::abs(mode_wavevector.h_view(n, guide_axis))*alfven_speed_reference
        : angular_frequency_reference;
    if (!std::isfinite(mode_frequency_reference[n])
        || mode_frequency_reference[n] <= 0.0) {
      AntennaFatal("Antenna mode frequencies must be finite and positive");
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Draw the stationary complex Gaussian state, or start exactly from zero.

void AntennaDriver::InitializeModeState() {
  constexpr Real inv_sqrt_two = 0.70710678118654752440;
  mode_alpha_weight.template sync<HostMemSpace>();
  for (int family = 0; family < num_families; ++family) {
    for (int n = 0; n < num_modes; ++n) {
      const Real mode_rms = apar_rms[family]*mode_alpha_weight.h_view(n);
      for (int q = 0; q < num_quadratures; ++q) {
        mode_state.h_view(family, n, q) = (stationary_initial_state
                                          && mode_rms != 0.0)
            ? mode_rms*inv_sqrt_two*AntennaGaussian(&rstate) : 0.0;
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
  mode_alpha_weight.template sync<HostMemSpace>();
  for (int family = 0; family < num_families; ++family) {
    const Real sign = (family == 0) ? 1.0 : -1.0;
    for (int n = 0; n < num_modes; ++n) {
      const Real mode_rms = apar_rms[family]*mode_alpha_weight.h_view(n);
      if (mode_rms == 0.0) {
        mode_state.h_view(family, n, 0) = 0.0;
        mode_state.h_view(family, n, 1) = 0.0;
        continue;
      }
      const Real gamma_rate = decorrelation_factor*mode_frequency_reference[n];
      const Real decay = std::exp(-gamma_rate*dt);
      const Real noise_fraction = std::sqrt(std::max(0.0, 1.0 - decay*decay));
      const Real theta = sign*frequency_factor*mode_frequency_reference[n]*dt;
      const Real cosine = std::cos(theta);
      const Real sine = std::sin(theta);
      const Real old_real = mode_state.h_view(family, n, 0);
      const Real old_imag = mode_state.h_view(family, n, 1);
      const Real noise_scale = mode_rms*noise_fraction*inv_sqrt_two;
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
//! \brief Build the canonical cell current on active cells.

void AntennaDriver::SynthesizeCellCurrent() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const bool use_electric_ct = pmy_pack->pmhd->use_electric_ct;
  const int cell_is = use_electric_ct ? is : 0;
  const int cell_ie = use_electric_ct ? ie : nx1 + 2*indcs.ng - 1;
  const int cell_js = use_electric_ct ? js : 0;
  const int cell_je = use_electric_ct ? je : nx2 + 2*indcs.ng - 1;
  const int cell_ks = use_electric_ct ? ks : 0;
  const int cell_ke = use_electric_ct ? ke : nx3 + 2*indcs.ng - 1;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto state = mode_state;
  auto wavevector = mode_wavevector;
  auto polarization = mode_polarization;
  auto jcell = current;
  const int mode_count = num_modes;

  current_ready = false;
  par_for("antenna_current_cell", DevExeSpace(), 0, nmb - 1, cell_ks, cell_ke,
          cell_js, cell_je, cell_is, cell_ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                              size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                              size.d_view(m).x3max);
    Real current1 = 0.0, current2 = 0.0, current3 = 0.0;
    for (int n = 0; n < mode_count; ++n) {
      const Real k1 = wavevector.d_view(n, 0);
      const Real k2 = wavevector.d_view(n, 1);
      const Real k3 = wavevector.d_view(n, 2);
      const Real phase = k1*x + k2*y + k3*z;
      Real alpha = 0.0;
      for (int family = 0; family < num_families; ++family) {
        alpha += state.d_view(family, n, 0)*cos(phase)
                 - state.d_view(family, n, 1)*sin(phase);
      }
      current1 += polarization.d_view(n, 0)*alpha;
      current2 += polarization.d_view(n, 1)*alpha;
      current3 += polarization.d_view(n, 2)*alpha;
    }
    jcell(m, 0, k, j, i) = current1;
    jcell(m, 1, k, j, i) = current2;
    jcell(m, 2, k, j, i) = current3;
  });
}

//----------------------------------------------------------------------------------------
//! \brief Form J_f=A_i J_c, the layout source, and compatible diagnostics.

void AntennaDriver::FinalizeCurrent() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto jcell = current;
  auto jface = current_face;
  auto japplied = applied_current;

  // The exchanged cell ghosts make these shared faces single-valued.  On a
  // uniform mesh the exact operator identity is D_f^i A_i = D_c^i.
  par_for("antenna_current_face1", DevExeSpace(), 0, nmb - 1, ks, ke, js, je,
          is, ie + 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    jface.x1f(m, k, j, i) =
        0.5*(jcell(m, 0, k, j, i - 1) + jcell(m, 0, k, j, i));
  });
  par_for("antenna_current_face2", DevExeSpace(), 0, nmb - 1, ks, ke, js, je + 1,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    jface.x2f(m, k, j, i) =
        0.5*(jcell(m, 1, k, j - 1, i) + jcell(m, 1, k, j, i));
  });
  par_for("antenna_current_face3", DevExeSpace(), 0, nmb - 1, ks, ke + 1, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    jface.x3f(m, k, j, i) =
        0.5*(jcell(m, 2, k - 1, j, i) + jcell(m, 2, k, j, i));
  });

  const bool use_electric_ct = pmy_pack->pmhd->use_electric_ct;
  par_for("antenna_applied_current", DevExeSpace(), 0, nmb - 1, ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    if (use_electric_ct) {
      japplied(m, 0, k, j, i) =
          0.5*(jface.x1f(m, k, j, i) + jface.x1f(m, k, j, i + 1));
      japplied(m, 1, k, j, i) =
          0.5*(jface.x2f(m, k, j, i) + jface.x2f(m, k, j + 1, i));
      japplied(m, 2, k, j, i) =
          0.5*(jface.x3f(m, k, j, i) + jface.x3f(m, k + 1, j, i));
    } else {
      japplied(m, 0, k, j, i) = jcell(m, 0, k, j, i);
      japplied(m, 1, k, j, i) = jcell(m, 1, k, j, i);
      japplied(m, 2, k, j, i) = jcell(m, 2, k, j, i);
    }
  });

  Real volume = 0.0, current_squared = 0.0, applied_current_squared = 0.0;
  Kokkos::parallel_reduce("antenna_current_rms",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum_volume, Real &sum_j2,
                Real &sum_applied_j2) {
    const Real dv = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    sum_volume += dv;
    sum_j2 += dv*(SQR(jcell(m, 0, k, j, i)) + SQR(jcell(m, 1, k, j, i))
                  + SQR(jcell(m, 2, k, j, i)));
    sum_applied_j2 +=
        dv*(SQR(japplied(m, 0, k, j, i)) + SQR(japplied(m, 1, k, j, i))
            + SQR(japplied(m, 2, k, j, i)));
  }, Kokkos::Sum<Real>(volume), Kokkos::Sum<Real>(current_squared),
     Kokkos::Sum<Real>(applied_current_squared));

  Real max_divergence = 0.0, max_compatibility_error = 0.0;
  Kokkos::parallel_reduce("antenna_current_divergence",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum_divergence,
                Real &maximum_compatibility) {
    const Real face_divergence =
        (jface.x1f(m, k, j, i + 1) - jface.x1f(m, k, j, i))
            /size.d_view(m).dx1
        + (jface.x2f(m, k, j + 1, i) - jface.x2f(m, k, j, i))
            /size.d_view(m).dx2
        + (jface.x3f(m, k + 1, j, i) - jface.x3f(m, k, j, i))
            /size.d_view(m).dx3;
    const Real cell_divergence =
        0.5*(jcell(m, 0, k, j, i + 1) - jcell(m, 0, k, j, i - 1))
            /size.d_view(m).dx1
        + 0.5*(jcell(m, 1, k, j + 1, i) - jcell(m, 1, k, j - 1, i))
            /size.d_view(m).dx2
        + 0.5*(jcell(m, 2, k + 1, j, i) - jcell(m, 2, k - 1, j, i))
            /size.d_view(m).dx3;
    const Real active_divergence = use_electric_ct
        ? face_divergence : cell_divergence;
    maximum_divergence = fmax(maximum_divergence, fabs(active_divergence));
    maximum_compatibility = fmax(
        maximum_compatibility, fabs(face_divergence - cell_divergence));
  }, Kokkos::Max<Real>(max_divergence),
     Kokkos::Max<Real>(max_compatibility_error));

  Real max_layout_filter = 0.0;
  Kokkos::parallel_reduce("antenna_layout_filter",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum) {
    maximum = fmax(maximum,
                   fabs(japplied(m, 0, k, j, i) - jcell(m, 0, k, j, i)));
    maximum = fmax(maximum,
                   fabs(japplied(m, 1, k, j, i) - jcell(m, 1, k, j, i)));
    maximum = fmax(maximum,
                   fabs(japplied(m, 2, k, j, i) - jcell(m, 2, k, j, i)));
  }, Kokkos::Max<Real>(max_layout_filter));

  current_sum_local[0] = volume;
  current_sum_local[1] = current_squared;
  current_sum_local[2] = applied_current_squared;
  current_max_local[0] = max_divergence;
  current_max_local[1] = max_layout_filter;
  current_max_local[2] = max_compatibility_error;
#if MPI_PARALLEL_ENABLED
  MPI_Iallreduce(current_sum_local, current_sum_global, 3, MPI_ATHENA_REAL,
                 MPI_SUM, pbval_current->comm_vars,
                 &current_reduction_requests[0]);
  MPI_Iallreduce(current_max_local, current_max_global, 3, MPI_ATHENA_REAL,
                 MPI_MAX, pbval_current->comm_vars,
                 &current_reduction_requests[1]);
  current_reduction_pending = true;
#else
  for (int n = 0; n < 3; ++n) {
    current_sum_global[n] = current_sum_local[n];
    current_max_global[n] = current_max_local[n];
  }
  CommitCurrentDiagnostics();
#endif
}

//----------------------------------------------------------------------------------------
//! \brief Poll the nonblocking global current-diagnostic reductions.

TaskStatus AntennaDriver::FinishCurrentReductions() {
#if MPI_PARALLEL_ENABLED
  if (current_reduction_pending) {
    int complete = 0;
    MPI_Testall(2, current_reduction_requests, &complete, MPI_STATUSES_IGNORE);
    if (complete == 0) return TaskStatus::incomplete;
    current_reduction_pending = false;
    CommitCurrentDiagnostics();
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Commit globally reduced current norms and mimetic residuals.

void AntennaDriver::CommitCurrentDiagnostics() {
  const Real volume = current_sum_global[0];
  last_current_rms = (volume > 0.0)
      ? std::sqrt(current_sum_global[1]/volume) : 0.0;
  last_applied_current_rms = (volume > 0.0)
      ? std::sqrt(current_sum_global[2]/volume) : 0.0;
  last_divergence = current_max_global[0];
  last_layout_filter = current_max_global[1];
  last_compatibility_error = current_max_global[2];
  current_ready = true;
}

//----------------------------------------------------------------------------------------
//! \brief Finalize host mode coefficients and reference data loaded from a restart.

void AntennaDriver::MarkRestarted() {
  mode_state.template modify<HostMemSpace>();
  mode_state.template sync<DevExeSpace>();
  UpdateModeFrequencies();
  initialized = true;
  reference_initialized = true;
  RebuildCurrentBlocking();
}

//----------------------------------------------------------------------------------------
//! \brief Update coefficients once per full step and synthesize the held antenna current.

TaskStatus AntennaDriver::UpdateAntenna(Driver *pdrive, int stage) {
  (void)pdrive;
  (void)stage;
  TaskStatus status = InitializeReferenceState();
  if (status != TaskStatus::complete) return status;
  if (!initialized) {
    InitializeModeState();
    initialized = true;
  } else {
    AdvanceModeState(pmy_pack->pmesh->dt);
  }
  SynthesizeCellCurrent();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Post the canonical cell-current ghost exchange.

TaskStatus AntennaDriver::StartCurrentExchange(Driver *pdrive, int stage) {
  (void)pdrive;
  (void)stage;
  if (!pmy_pack->pmhd->use_electric_ct) {
    current_exchange_phase = 5;
    return TaskStatus::complete;
  }
  TaskStatus status = TaskStatus::complete;
  if (current_exchange_phase == 0) {
    status = pbval_current->InitRecv(3);
    if (status != TaskStatus::complete) return status;
    current_exchange_phase = 1;
  }
  if (current_exchange_phase == 1) {
    // Multilevel meshes are rejected, so the coarse-array argument is never used.
    status = pbval_current->PackAndSendCC(current, current);
    if (status != TaskStatus::complete) return status;
    current_exchange_phase = 2;
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Complete the cell-current exchange and build the compatible layout source.

TaskStatus AntennaDriver::FinishCurrentExchange(Driver *pdrive, int stage) {
  (void)pdrive;
  (void)stage;
  TaskStatus status = TaskStatus::complete;
  if (current_exchange_phase == 5) {
    FinalizeCurrent();
    current_exchange_phase = 6;
  }
  if (current_exchange_phase == 2) {
    status = pbval_current->RecvAndUnpackCC(current, current);
    if (status != TaskStatus::complete) return status;
    current_exchange_phase = 3;
  }
  if (current_exchange_phase == 3) {
    status = pbval_current->ClearSend();
    if (status != TaskStatus::complete) return status;
    current_exchange_phase = 4;
  }
  if (current_exchange_phase == 4) {
    status = pbval_current->ClearRecv();
    if (status != TaskStatus::complete) return status;
    FinalizeCurrent();
    current_exchange_phase = 6;
  }
  if (current_exchange_phase == 6) {
    status = FinishCurrentReductions();
    if (status != TaskStatus::complete) return status;
    current_exchange_phase = 0;
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Rebuild a restart current before the normal task lists begin.

void AntennaDriver::RebuildCurrentBlocking() {
  SynthesizeCellCurrent();
  while (StartCurrentExchange(nullptr, 0) == TaskStatus::incomplete) {}
  while (FinishCurrentExchange(nullptr, 0) == TaskStatus::incomplete) {}
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
  auto jant = applied_current;
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
    const Real current1 = jant(m, 0, k, j, i);
    const Real current2 = jant(m, 1, k, j, i);
    const Real current3 = jant(m, 2, k, j, i);
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
    const Real current1 = jant(m, 0, k, j, i);
    const Real current2 = jant(m, 1, k, j, i);
    const Real current3 = jant(m, 2, k, j, i);
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
