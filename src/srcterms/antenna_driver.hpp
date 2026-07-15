#ifndef SRCTERMS_ANTENNA_DRIVER_HPP_
#define SRCTERMS_ANTENNA_DRIVER_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file antenna_driver.hpp
//! \brief Electromagnetic oscillating-Langevin antenna for resistive SRMHD.

#include <memory>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \class AntennaDriver
//! \brief Evolves a neutral external current and couples its four-force to total SRMHD.

class AntennaDriver {
 public:
  static constexpr int num_modes = 4;
  static constexpr int num_families = 2;
  static constexpr int num_quadratures = 2;

  AntennaDriver(MeshBlockPack *pp, ParameterInput *pin);
  ~AntennaDriver() = default;

  // Cell current and a divergence-free face representation whose arithmetic
  // face-to-cell average recovers that same current exactly.
  DvceArray5D<Real> current;
  DvceFaceFld4D<Real> current_face;

  // Complex OU coefficients: [propagation family][independent mode][real/imaginary].
  DualArray3D<Real> mode_state;
  DualArray2D<Real> mode_wavevector;
  RNG_State rstate;

  TaskID stage_task_id;

  // Instantaneous and cumulative source diagnostics.
  Real last_power = 0.0;
  Real last_current_rms = 0.0;
  Real last_divergence = 0.0;
  Real last_face_cell_mismatch = 0.0;
  Real last_momentum1 = 0.0;
  Real last_momentum2 = 0.0;
  Real last_momentum3 = 0.0;
  Real injected_energy = 0.0;
  Real injected_momentum1 = 0.0;
  Real injected_momentum2 = 0.0;
  Real injected_momentum3 = 0.0;
  Real alfven_speed_reference = 0.0;
  Real magnetization_reference = 0.0;
  Real angular_frequency_reference = 0.0;
  Real apar_rms[num_families] = {0.0, 0.0};

  void IncludeUpdateTask(std::shared_ptr<TaskList> tl, TaskID start);
  void IncludeApplyTask(std::shared_ptr<TaskList> tl, TaskID start);
  TaskStatus UpdateAntenna(Driver *pdrive, int stage);
  TaskStatus ApplyAntenna(Driver *pdrive, int stage);
  void MarkRestarted();

 private:
  void InitializeReferenceState();
  void InitializeModeState();
  void AdvanceModeState(Real dt);
  void SynthesizeCurrent();

  MeshBlockPack *pmy_pack;
  bool apply_source;
  bool stationary_initial_state;
  bool fixed_alfven_speed = false;
  bool zhdankin_amplitude = false;
  bool initialized = false;
  bool reference_initialized = false;
  bool current_ready = false;
  std::string va_reference;
  Real frequency_factor;
  Real decorrelation_factor;
  Real amplitude_fraction[num_families] = {1.0, 1.0};
  Real injected_energy_start = 0.0;
  Real injected_momentum1_start = 0.0;
  Real injected_momentum2_start = 0.0;
  Real injected_momentum3_start = 0.0;
};

#endif  // SRCTERMS_ANTENNA_DRIVER_HPP_
