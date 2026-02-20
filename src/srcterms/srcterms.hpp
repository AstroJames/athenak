#ifndef SRCTERMS_SRCTERMS_HPP_
#define SRCTERMS_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.hpp
//! \brief Data, functions, and classes to implement various source terms in the hydro
//! and/or MHD equations of motion.  Currently implemented:
//!  (1) constant (gravitational) acceleration - for RTI
//!  (2) optically-thin ISM cooling and heating
//!  (3) relativistic cooling
//!  (4) stochastic supernova driving
//!  (5) shearing box in 2D (x-z), for both hydro and MHD
//!  (6) random forcing to drive turbulence - implemented in TurbulenceDriver class

#include <map>
#include <random>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

// forward declarations
class TurbulenceDriver;
class Driver;

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms {
 public:
  SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~SourceTerms();

  // data
  // flags for various source terms
  bool const_accel;
  bool ism_cooling;
  bool rel_cooling;
  bool sn_driving;
  bool beam;
  bool shearing_box, shearing_box_r_phi;

  // new timestep
  Real dtnew;

  // magnitude and direction of constant accel
  Real const_accel_val;
  int const_accel_dir;

  // heating rate used with ISM cooling
  Real hrate;

  // cooling rate used with relativistic cooling
  Real crate_rel;
  Real cpower_rel;

  // stochastic supernova driving
  Real sn_rate;        // target events per unit code time
  Real sn_einj;        // thermal energy per event in code units
  Real sn_rinj;        // injection radius in code units
  Real sn_zmin;        // lower edge of z-driving band in code units
  Real sn_zmax;        // upper edge of z-driving band in code units
  int sn_seed;         // RNG seed
  bool sn_log_events;  // write event list to file
  std::string sn_log_file;

  // beam source
  Real dii_dt;

  // shearing box
  Real qshear, omega0;

  // functions
  void ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real dt, DvceArray5D<Real> &u0);
  void ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real dt, DvceArray5D<Real> &u0);
  void RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real dt, DvceArray5D<Real> &u0);
  void SupernovaDriving(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                        const Real dt, DvceArray5D<Real> &u0);
  void BeamSource(DvceArray5D<Real> &i0, const Real dt);
  void ShearingBox(const DvceArray5D<Real> &w0, const EOS_Data &eos_data, const Real bdt,
                   DvceArray5D<Real> &u0);
  void ShearingBox(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                   const EOS_Data &eos_data, const Real bdt, DvceArray5D<Real> &u0);
  // in 2D shearing box there is a source term for Ex and Ey
  void SBoxEField(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);

  void NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos);

 private:
  MeshBlockPack *pmy_pack;
  std::mt19937_64 sn_rng_;
  long long sn_event_count_;
  Real sn_event_accum_;
  bool sn_rinj_warned_;
};

#endif  // SRCTERMS_SRCTERMS_HPP_
