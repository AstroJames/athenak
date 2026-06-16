#ifndef UTILS_SPECTRAL_IC_GEN_HPP_
#define UTILS_SPECTRAL_IC_GEN_HPP_
//========================================================================================
// AthenaK: astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectral_ic_gen.hpp
//! \brief Defines SpectralICGenerator class for initializing turbulent magnetic (or
//! velocity) fields with specified power spectra.
//!
//! Two generation paths are available:
//!   - GenerateVectorPotential: direct Fourier synthesis (O(N_modes × N_cells)).
//!   - GenerateVectorPotentialFFT: FFT-based synthesis using one of three backends
//!     selected at compile time: heFFTe (MPI-distributed, HEFFTE_ENABLED) → KokkosFFT
//!     (serial per-rank, FFT_ENABLED) → direct synthesis fallback.
//!
//! The vector potential A is generated with spectral amplitudes set by the chosen
//! spectrum form.  B = curl(A) is then computed by the caller using a standard
//! second-order finite-difference stencil, guaranteeing div(B) = 0 discretely.
//!
//! Spectrum convention: for the power-law form, the parameter `spectral_index` sets
//! E_B(k) ∝ k^{-spectral_index}.  Each A-mode amplitude is scaled as
//! k^{-(spectral_index+4)/2} so that after the curl |B_k|^2 ∝ k^{-spectral_index-2}
//! and the shell-integrated spectrum k^2|B_k|^2 ∝ k^{-spectral_index}.
//!
//! References:
//!   Federrath et al. (2010, A&A 512, A81)

#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \class SpectralICGenerator

class SpectralICGenerator {
 public:
  // Spectrum form options
  enum class SpectrumForm { kBand, kParabolic, kPowerLaw };

  // Constructor: reads parameters from the given input-file block
  SpectralICGenerator(MeshBlockPack *pmbp, ParameterInput *pin,
                      const std::string &block = "spectral_ic");
  ~SpectralICGenerator() = default;

  // Fill ax, ay, az arrays (allocated by caller to size [nmb, ke+2, je+2, ie+2])
  // with the vector potential evaluated at node / face-edge positions.
  // Caller is then responsible for curling A to obtain face-centered B.
  void GenerateVectorPotential(DvceArray4D<Real> &ax, DvceArray4D<Real> &ay,
                               DvceArray4D<Real> &az);

  // FFT-based generation: heFFTe (MPI-distributed) → KokkosFFT (serial) → direct
  // synthesis fallback.  Returns the backend name used ("heffte", "kokkos_fft",
  // or "direct").
  std::string GenerateVectorPotentialFFT(DvceArray4D<Real> &ax, DvceArray4D<Real> &ay,
                                          DvceArray4D<Real> &az);

  int mode_count;  // number of modes (set in constructor, public for diagnostics)

 private:
  MeshBlockPack *pmy_pack;

  // Wavenumber range (integer mode indices, same convention as TurbulenceDriver)
  int nlow, nhigh;

  SpectrumForm spectrum_form;
  Real spectral_index;  // power-law exponent: E_B(k) ∝ k^{-spectral_index}

  RNG_State rstate;     // random state for mode coefficient generation

  // Wavenumber arrays
  DualArray1D<Real> kx_mode, ky_mode, kz_mode;

  // Trig-product coefficient arrays for A_x (3 components × 8 terms = 24 arrays).
  // Naming: a{x,y,z}_{c/s}{c/s}{c/s} where c=cos, s=sin and the three letters
  // correspond to the x, y, z factors respectively.
  DualArray1D<Real> ax_ccc, ax_ccs, ax_csc, ax_css, ax_scc, ax_scs, ax_ssc, ax_sss;
  DualArray1D<Real> ay_ccc, ay_ccs, ay_csc, ay_css, ay_scc, ay_scs, ay_ssc, ay_sss;
  DualArray1D<Real> az_ccc, az_ccs, az_csc, az_css, az_scc, az_scs, az_ssc, az_sss;

  // Precomputed trig tables at face/node positions [nmb, mode_count, ncells]
  // (ncells is large enough to cover indices is..ie+1, js..je+1, ks..ke+1)
  DvceArray3D<Real> xcos_f, xsin_f;
  DvceArray3D<Real> ycos_f, ysin_f;
  DvceArray3D<Real> zcos_f, zsin_f;

  // Returns spectral weight for A-field mode amplitude at mode magnitude n_mag
  // (= sqrt(nkx^2+nky^2+nkz^2) in mode-index units, consistent with nlow/nhigh).
  // For power-law form: 1/n^((s+4)/2), so E_B(n) ∝ n^{-s}.
  // For band form: 1.  For parabolic form: smooth bump centred on (nlow+nhigh)/2.
  Real ModeAmplitude(Real n_mag) const;

  void CountModes();
  void GenerateModeCoefficients();
  void PrecomputeTrigTables();
};

//----------------------------------------------------------------------------------------
//! \fn SubtractGlobalMeanB
//! \brief Subtracts the global (MPI-reduced) volume-averaged B from all face-field values.
void SubtractGlobalMeanB(MeshBlockPack *pmbp, DvceFaceFld4D<Real> &b0);

//----------------------------------------------------------------------------------------
//! \fn NormalizeRmsB
//! \brief Rescales all face-field values so that the global RMS |B| equals rms_target.
void NormalizeRmsB(MeshBlockPack *pmbp, DvceFaceFld4D<Real> &b0, Real rms_target);

#endif  // UTILS_SPECTRAL_IC_GEN_HPP_
