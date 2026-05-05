#ifndef UTILS_SPECTRAL_2D_FIELD_GEN_HPP_
#define UTILS_SPECTRAL_2D_FIELD_GEN_HPP_
//========================================================================================
// AthenaK: astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectral_2d_field_gen.hpp
//! \brief Utility for generating divergence-free 2D vector fields from a spectral scalar
//! potential concentrated in a prescribed Fourier shell or band.

#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"

class Spectral2DFieldGenerator {
 public:
  enum class SpectrumForm { gaussian, band };

  Spectral2DFieldGenerator(MeshBlockPack *pmbp, ParameterInput *pin,
                           const std::string &block = "spectral_ic");
  ~Spectral2DFieldGenerator() = default;

  void GenerateCurlField(DvceArray4D<Real> &vx, DvceArray4D<Real> &vy);

  int mode_count;

 private:
  MeshBlockPack *pmy_pack;

  Real k_peak;
  Real k_width;
  Real k_min;
  Real k_max;
  SpectrumForm spectrum_form;
  RNG_State rstate;

  DualArray1D<Real> kx_mode, ky_mode;
  DualArray1D<Real> az_cc, az_cs, az_sc, az_ss;
  DvceArray3D<Real> xcos, xsin;
  DvceArray3D<Real> ycos, ysin;

  Real ModeAmplitude(Real kmag) const;
  void CountModes();
  void GenerateModeCoefficients();
  void PrecomputeTrigTables();
};

void RemoveVectorFieldMean(MeshBlockPack *pmbp, DvceArray4D<Real> &vx,
                           DvceArray4D<Real> &vy, DvceArray4D<Real> &vz);
void NormalizeVectorFieldRms(MeshBlockPack *pmbp, DvceArray4D<Real> &vx,
                             DvceArray4D<Real> &vy, DvceArray4D<Real> &vz,
                             Real rms_target);

#endif  // UTILS_SPECTRAL_2D_FIELD_GEN_HPP_
