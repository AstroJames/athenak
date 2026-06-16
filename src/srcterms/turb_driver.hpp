#ifndef SRCTERMS_TURB_DRIVER_HPP_
#define SRCTERMS_TURB_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.hpp
//  \brief defines turbulence driver class, which implements data and functions for
//  randomly forced turbulence which evolves via an Ornstein-Uhlenbeck stochastic process

#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \class TurbulenceDriver

class TurbulenceDriver {
 public:
  TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceDriver();

  DvceArray5D<Real> force, force_tmp;  // arrays used for turb forcing
  DvceArray6D<Real> force_component, force_tmp_component;
  RNG_State rstate;                    // random state

  DualArray2D<Real> xccc, xccs, xcsc, xcss, xscc, xscs, xssc, xsss;
  DualArray2D<Real> yccc, yccs, ycsc, ycss, yscc, yscs, yssc, ysss;
  DualArray2D<Real> zccc, zccs, zcsc, zcss, zscc, zscs, zssc, zsss;
  DualArray2D<Real> kx_mode, ky_mode, kz_mode;
  DvceArray4D<Real> xcos, xsin, ycos, ysin, zcos, zsin;

  // parameters of driving
  int num_components, max_mode_count;
  Real last_power = 0.0;
  std::vector<std::string> component_name;
  std::vector<int> nlow, nhigh, mode_count;
  std::vector<Real> tcorr, dedt;
  std::vector<Real> expo, exp_prl, exp_prp;
  std::vector<Real> sol_weight;
  std::vector<Real> parabola_peak, parabola_width;
  std::vector<Real> vertical_window_width, vertical_window_transition;
  std::vector<Real> transverse_window_radius, transverse_window_transition;
  std::vector<int> driving_geometry, driving_profile;
  std::vector<int> vertical_window, transverse_window;

  // functions
  void IncludeInitializeModesTask(std::shared_ptr<TaskList> tl, TaskID start);
  void IncludeAddForcingTask(std::shared_ptr<TaskList> tl, TaskID start);
  TaskStatus InitializeModes(Driver *pdrive, int stage);
  TaskStatus UpdateForcing(Driver *pdrive, int stage);
  TaskStatus AddForcing(Driver *pdrive, int stage);
  void Initialize();

 private:
  bool first_time = true;   // flag to enable initialization on first call
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this TurbulenceDriver
};

#endif  // SRCTERMS_TURB_DRIVER_HPP_
