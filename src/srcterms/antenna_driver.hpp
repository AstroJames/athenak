#ifndef SRCTERMS_ANTENNA_DRIVER_HPP_
#define SRCTERMS_ANTENNA_DRIVER_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file antenna_driver.hpp
//! \brief Electromagnetic oscillating-Langevin antenna for resistive SRMHD.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"

class MeshBoundaryValuesCC;

//----------------------------------------------------------------------------------------
//! \class AntennaDriver
//! \brief Evolves a neutral external current and couples its four-force to total SRMHD.

class AntennaDriver {
 public:
  static constexpr int num_families = 2;
  static constexpr int num_quadratures = 2;

  AntennaDriver(MeshBlockPack *pp, ParameterInput *pin);
  ~AntennaDriver();

  // Canonical cell current, its compatible face representation, and the cell
  // current actually applied by the active electric-field layout.  For FC-E,
  // the latter includes the two arithmetic-average filters A_i^T A_i.
  DvceArray5D<Real> current;
  DvceFaceFld4D<Real> current_face;
  DvceArray5D<Real> applied_current;

  // Complex OU coefficients: [propagation family][independent mode][real/imaginary].
  // Synthesis uses Re[a exp(i k.x)], so a is twice the conventional positive-k
  // Fourier coefficient and <|a_k|^2> = apar_rms^2*mode_alpha_weight(k)^2.
  DualArray3D<Real> mode_state;
  DualArray2D<Real> mode_wavevector;
  DualArray2D<Real> mode_polarization;
  DualArray1D<Real> mode_alpha_weight;
  std::vector<std::array<int, 3>> mode_integer;
  int num_modes = 0;
  RNG_State rstate;

  TaskID stage_task_id;

  // Instantaneous and cumulative source diagnostics.
  Real last_power = 0.0;
  Real last_current_rms = 0.0;
  Real last_applied_current_rms = 0.0;
  Real last_divergence = 0.0;
  Real last_compatibility_error = 0.0;
  Real last_layout_filter = 0.0;
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

  TaskID IncludeUpdateTask(std::shared_ptr<TaskList> tl, TaskID start);
  void IncludeApplyTask(std::shared_ptr<TaskList> tl, TaskID start);
  TaskStatus UpdateAntenna(Driver *pdrive, int stage);
  TaskStatus StartCurrentExchange(Driver *pdrive, int stage);
  TaskStatus FinishCurrentExchange(Driver *pdrive, int stage);
  TaskStatus ApplyAntenna(Driver *pdrive, int stage);
  void MarkRestarted();
  int RestartModeCount() const {return num_modes;}
  std::uint64_t RestartModeSetHash() const {return mode_set_hash;}
  bool IsLegacyV2ModeSet() const;
  void PackRestartModes(std::vector<Real> &values);
  void RestoreRestartModes(const std::vector<Real> &values);

 private:
  TaskStatus InitializeReferenceState();
  void UpdateModeFrequencies();
  void ComputeModeSetHash();
  void InitializeModeState();
  void AdvanceModeState(Real dt);
  void SynthesizeCellCurrent();
  void FinalizeCurrent();
  TaskStatus FinishCurrentReductions();
  void CommitCurrentDiagnostics();
  void RebuildCurrentBlocking();

  MeshBlockPack *pmy_pack;
  MeshBoundaryValuesCC *pbval_current = nullptr;
  bool apply_source;
  bool stationary_initial_state;
  bool fixed_alfven_speed = false;
  bool zhdankin_amplitude = false;
  bool initialized = false;
  bool reference_initialized = false;
  bool current_ready = false;
  bool current_reduction_pending = false;
  bool alfven_parallel_frequency = false;
  int current_exchange_phase = 0;
  int reference_exchange_phase = 0;
  int guide_axis = 2;
  std::string mode_set;
  std::string frequency_model;
  std::string current_envelope;
  std::string va_reference;
  std::vector<Real> mode_frequency_reference;
  std::uint64_t mode_set_hash = 0;
  Real frequency_factor;
  Real decorrelation_factor;
  Real current_parabola_peak = 1.0;
  Real current_parabola_width = 1.0;
  Real current_exponent_perp = 0.0;
  Real current_exponent_parallel = 0.0;
  Real current_reference_perp = 1.0;
  Real current_reference_parallel = 1.0;
  Real amplitude_fraction[num_families] = {1.0, 1.0};
  Real injected_energy_start = 0.0;
  Real injected_momentum1_start = 0.0;
  Real injected_momentum2_start = 0.0;
  Real injected_momentum3_start = 0.0;
  Real current_sum_local[3] = {0.0, 0.0, 0.0};
  Real current_sum_global[3] = {0.0, 0.0, 0.0};
  Real current_max_local[3] = {0.0, 0.0, 0.0};
  Real current_max_global[3] = {0.0, 0.0, 0.0};
  Real reference_sum_local[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  Real reference_sum_global[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
#if MPI_PARALLEL_ENABLED
  MPI_Request current_reduction_requests[2] = {
      MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  MPI_Request reference_reduction_request = MPI_REQUEST_NULL;
#endif
};

#endif  // SRCTERMS_ANTENNA_DRIVER_HPP_
