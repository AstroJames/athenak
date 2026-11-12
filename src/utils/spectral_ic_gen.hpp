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
//!     selected at compile time: heFFTe (MPI-distributed, HEFFTE_ENABLED), KokkosFFT
//!     (serial per-rank, KOKKOS_FFT_ENABLED), or direct synthesis fallback.
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
//! Isotropic narrow-shell seed (seed_type = shell; default seed_type = legacy keeps the
//! behaviour above bit-for-bit). n_dirs Fibonacci directions on the sphere (n_dirs/2 on
//! a half-sphere, conjugates implicit) are randomly rotated (iseed), snapped to lattice
//! vectors with ||n|-n0| <= shell_width, and weighted by their spherical-Voronoi solid
//! angles w_i (sum 4*pi over the antipodally symmetric set). Each mode has
//! |b_k|^2 proportional to w_i and a helical polarisation, with per-mode k_hat:
//!   helicity=+1/-1: b_k = r sqrt(w) e_h exp(i phi1), with i k_hat x e_h = h e_h,
//!   helicity=0:     b_k = r sqrt(w/2) (e_+ exp(i phi1) + e_- exp(i phi2)),
//! so h=+1 gives curl B = +|k| B (same sign as legacy force_free helicity=+1) and h=0
//! has exactly zero helicity per mode. r=1 (amplitude=equal) or Rayleigh with <r^2>=1
//! (amplitude=gaussian). A_k = i k x b_k/|k|^2 (Coulomb gauge) is synthesised at edge
//! midpoints with analytic edge averages, so the CT curl gives div B = 0 to round-off.
//!
//! References:
//!   Federrath et al. (2010, A&A 512, A81)

#include <array>
#include <string>
#include <vector>

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
  // force_free=true fills analytic edge averages of helical A on one shell.
  void GenerateVectorPotential(DvceArray4D<Real> &ax, DvceArray4D<Real> &ay,
                               DvceArray4D<Real> &az);

  // FFT-based generation: heFFTe (MPI-distributed) → KokkosFFT (serial) → direct
  // synthesis fallback.  Returns the backend name used ("heffte", "kokkos_fft",
  // or "direct"). force_free=true selects "direct_force_free" here.
  std::string GenerateVectorPotentialFFT(DvceArray4D<Real> &ax, DvceArray4D<Real> &ay,
                                          DvceArray4D<Real> &az);

  int mode_count;  // number of modes (set in constructor, public for diagnostics)

  // Sparse mode table: (k_x, k_y, k_z, Re A_x, Re A_y, Re A_z, Im A_x, Im A_y, Im A_z)
  // with analytic edge-average factors applied; one of each +/-k pair (2 Re[...]).
  using ModeTable = std::vector<std::array<Real, 9>>;

  // Public only because nvcc rejects device lambdas in private member functions.
  void PrecomputeTrigTables();
  // Legacy force-free (force_free=true) mode table; arithmetic unchanged.
  ModeTable BuildForceFreeModes();
  // Isotropic narrow-shell (seed_type=shell) mode table; host-only, O(n_dirs*Q).
  ModeTable BuildShellModes();
  // Edge-midpoint synthesis of a mode table into ax/ay/az (contains a device lambda).
  void SynthesizeModeTable(const ModeTable &modes, DvceArray4D<Real> &ax,
                           DvceArray4D<Real> &ay, DvceArray4D<Real> &az);

 private:
  MeshBlockPack *pmy_pack;

  // Wavenumber range (integer mode indices, same convention as TurbulenceDriver)
  int nlow, nhigh;
  bool force_free;  // optional single-shell helical projection; default false
  int helicity;    // +1 or -1 when force_free is enabled; +1, -1 or 0 for shell seeds

  // seed_type = shell (isotropic narrow-shell seed); see file header.
  bool shell_seed = false;
  Real shell_n0 = 0.0;       // shell radius in mode units
  Real shell_width = 0.5;    // accept lattice vectors with ||n|-n0| <= shell_width
  int shell_ndirs = 0;       // directions on the full sphere (even)
  bool shell_equal_amp = true;  // amplitude = equal (true) or gaussian (false)

  SpectrumForm spectrum_form;
  Real spectral_index;  // power-law exponent: E_B(k) ∝ k^{-spectral_index}

  RNG_State rstate{};   // random state for mode coefficient generation

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

  void ValidateForceFree() const;
  void ValidateShell() const;
  void CountModes();
  void GenerateModeCoefficients();
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
