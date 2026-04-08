//========================================================================================
// AthenaK: astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectral_ic_gen.cpp
//! \brief Implements SpectralICGenerator and helper functions SubtractGlobalMeanB and
//! NormalizeRmsB for spectral magnetic-field initial conditions.

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "utils/random.hpp"
#include "utils/spectral_ic_gen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#if FFT_ENABLED
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>
#endif

#if HEFFTE_ENABLED
#include <heffte.h>
#endif

//----------------------------------------------------------------------------------------
// constructor

SpectralICGenerator::SpectralICGenerator(MeshBlockPack *pmbp, ParameterInput *pin,
                                         const std::string &block) :
  pmy_pack(pmbp),
  kx_mode("kx_mode",1), ky_mode("ky_mode",1), kz_mode("kz_mode",1),
  ax_ccc("ax_ccc",1), ax_ccs("ax_ccs",1), ax_csc("ax_csc",1), ax_css("ax_css",1),
  ax_scc("ax_scc",1), ax_scs("ax_scs",1), ax_ssc("ax_ssc",1), ax_sss("ax_sss",1),
  ay_ccc("ay_ccc",1), ay_ccs("ay_ccs",1), ay_csc("ay_csc",1), ay_css("ay_css",1),
  ay_scc("ay_scc",1), ay_scs("ay_scs",1), ay_ssc("ay_ssc",1), ay_sss("ay_sss",1),
  az_ccc("az_ccc",1), az_ccs("az_ccs",1), az_csc("az_csc",1), az_css("az_css",1),
  az_scc("az_scc",1), az_scs("az_scs",1), az_ssc("az_ssc",1), az_sss("az_sss",1),
  xcos_f("xcos_f",1,1,1), xsin_f("xsin_f",1,1,1),
  ycos_f("ycos_f",1,1,1), ysin_f("ysin_f",1,1,1),
  zcos_f("zcos_f",1,1,1), zsin_f("zsin_f",1,1,1) {
  // Read wavenumber range (integer mode indices, same convention as TurbulenceDriver).
  // k_phys = n * 2π / L, so nlow=2, nhigh=4 → 2 ≤ n ≤ 4
  nlow = pin->GetOrAddInteger(block, "nlow", 2);
  nhigh = pin->GetOrAddInteger(block, "nhigh", 4);

  // Spectrum form
  std::string spec_str = pin->GetOrAddString(block, "spectrum", "power_law");
  if (spec_str.compare("band") == 0) {
    spectrum_form = SpectrumForm::kBand;
  } else if (spec_str.compare("parabolic") == 0) {
    spectrum_form = SpectrumForm::kParabolic;
  } else {
    spectrum_form = SpectrumForm::kPowerLaw;
  }
  spectral_index = pin->GetOrAddReal(block, "spectral_index", 5.0/3.0);

  // Initialize RNG
  int64_t iseed = pin->GetOrAddInteger(block, "iseed", -1234);
  rstate.idum = iseed;

  CountModes();

  // Reallocate coefficient and trig arrays to correct sizes.
  // The trig tables are evaluated at face (left-edge) positions up to ie+1/je+1/ke+1,
  // so we always use the full cell-count formula (never the nx==1 short-circuit).
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);

  Kokkos::realloc(kx_mode, mode_count);
  Kokkos::realloc(ky_mode, mode_count);
  Kokkos::realloc(kz_mode, mode_count);

  Kokkos::realloc(ax_ccc, mode_count); Kokkos::realloc(ax_ccs, mode_count);
  Kokkos::realloc(ax_csc, mode_count); Kokkos::realloc(ax_css, mode_count);
  Kokkos::realloc(ax_scc, mode_count); Kokkos::realloc(ax_scs, mode_count);
  Kokkos::realloc(ax_ssc, mode_count); Kokkos::realloc(ax_sss, mode_count);

  Kokkos::realloc(ay_ccc, mode_count); Kokkos::realloc(ay_ccs, mode_count);
  Kokkos::realloc(ay_csc, mode_count); Kokkos::realloc(ay_css, mode_count);
  Kokkos::realloc(ay_scc, mode_count); Kokkos::realloc(ay_scs, mode_count);
  Kokkos::realloc(ay_ssc, mode_count); Kokkos::realloc(ay_sss, mode_count);

  Kokkos::realloc(az_ccc, mode_count); Kokkos::realloc(az_ccs, mode_count);
  Kokkos::realloc(az_csc, mode_count); Kokkos::realloc(az_css, mode_count);
  Kokkos::realloc(az_scc, mode_count); Kokkos::realloc(az_scs, mode_count);
  Kokkos::realloc(az_ssc, mode_count); Kokkos::realloc(az_sss, mode_count);

  // Trig tables cover indices is..ie+1 / js..je+1 / ks..ke+1 (face positions).
  // The allocation of ncells{1,2,3} is sufficient since ie+1 < ncells1, etc.
  Kokkos::realloc(xcos_f, nmb, mode_count, ncells1);
  Kokkos::realloc(xsin_f, nmb, mode_count, ncells1);
  Kokkos::realloc(ycos_f, nmb, mode_count, ncells2);
  Kokkos::realloc(ysin_f, nmb, mode_count, ncells2);
  Kokkos::realloc(zcos_f, nmb, mode_count, ncells3);
  Kokkos::realloc(zsin_f, nmb, mode_count, ncells3);

  GenerateModeCoefficients();
  PrecomputeTrigTables();
}

//----------------------------------------------------------------------------------------
//! \fn SpectralICGenerator::CountModes
//! \brief Counts modes with nlow^2 ≤ nkx^2+nky^2+nkz^2 ≤ nhigh^2.

void SpectralICGenerator::CountModes() {
  Real nlow_sqr  = static_cast<Real>(nlow)  * static_cast<Real>(nlow);
  Real nhigh_sqr = static_cast<Real>(nhigh) * static_cast<Real>(nhigh);
  mode_count = 0;
  for (int nkx = 0; nkx <= nhigh; nkx++) {
    for (int nky = 0; nky <= nhigh; nky++) {
      for (int nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        Real nsqr = static_cast<Real>(SQR(nkx) + SQR(nky) + SQR(nkz));
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr) mode_count++;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn SpectralICGenerator::ModeAmplitude
//! \brief Returns the spectral weight for the A-field at mode magnitude n_mag.
//!
//! \param n_mag  Mode magnitude |n| = sqrt(nkx^2+nky^2+nkz^2) in mode-index units.
//!               Both nlow/nhigh and the parabolic formula are in mode-index units,
//!               so all comparisons are consistent regardless of box size.
//!               The power-law form uses n_mag as well; the overall amplitude scale is
//!               irrelevant because NormalizeRmsB rescales the field to the target RMS.

Real SpectralICGenerator::ModeAmplitude(Real n_mag) const {
  if (n_mag < 1.0e-16) return 0.0;
  switch (spectrum_form) {
    case SpectrumForm::kBand:
      return 1.0;
    case SpectrumForm::kParabolic: {
      // Smooth bump peaking at n_mid, zero at nlow and nhigh.
      Real n_mid = 0.5*(static_cast<Real>(nlow) + static_cast<Real>(nhigh));
      Real dn    = static_cast<Real>(nhigh - nlow);
      if (dn < 1.0e-16) return 1.0;
      Real val = 1.0 - 4.0*SQR(n_mag - n_mid)/SQR(dn);
      return std::max(0.0, val);
    }
    case SpectrumForm::kPowerLaw:
      // A-amplitude ∝ n^{-(spectral_index+4)/2} so that E_B(n) ∝ n^{-spectral_index}.
      // (NormalizeRmsB handles the overall amplitude scale.)
      return 1.0/std::pow(n_mag, (spectral_index + 4.0)/2.0);
    default:
      return 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn SpectralICGenerator::GenerateModeCoefficients
//! \brief On the host: generates Gaussian random coefficients for each Fourier mode of
//! the vector potential, scaled by ModeAmplitude(k).  Coefficients are then synced to
//! the device.  Each component of A (ax, ay, az) is generated independently — the
//! div(B)=0 constraint is satisfied automatically via B = curl(A).

void SpectralICGenerator::GenerateModeCoefficients() {
  Mesh *pm = pmy_pack->pmesh;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  Real dkx = 2.0*M_PI/lx;
  Real dky = 2.0*M_PI/ly;
  Real dkz = 2.0*M_PI/lz;

  Real nlow_sqr  = static_cast<Real>(nlow)  * static_cast<Real>(nlow);
  Real nhigh_sqr = static_cast<Real>(nhigh) * static_cast<Real>(nhigh);

  int n = 0;  // current mode index
  for (int nkx = 0; nkx <= nhigh; nkx++) {
    for (int nky = 0; nky <= nhigh; nky++) {
      for (int nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        Real nsqr = static_cast<Real>(SQR(nkx) + SQR(nky) + SQR(nkz));
        if (nsqr < nlow_sqr || nsqr > nhigh_sqr) continue;

        Real kx = dkx*nkx;
        Real ky = dky*nky;
        Real kz = dkz*nkz;
        // Mode magnitude in mode-index units (consistent with nlow/nhigh)
        Real n_mag = std::sqrt(static_cast<Real>(SQR(nkx) + SQR(nky) + SQR(nkz)));

        kx_mode.h_view(n) = kx;
        ky_mode.h_view(n) = ky;
        kz_mode.h_view(n) = kz;

        Real amp = ModeAmplitude(n_mag);

        // Generate 8 trig coefficients per component.
        // A term is non-zero only when the sin() factor involves a non-zero wavenumber;
        // otherwise sin(0) ≡ 0 and that term is identically zero.
        // Pattern for term {s/c}kx {s/c}ky {s/c}kz:
        //   - sin factor is zero when the corresponding n is 0
        //   - cos factor is always potentially non-zero (cos(0)=1)
        // We conditionally draw from the RNG only for non-zero terms to preserve a
        // compact, reproducible RNG sequence.

        // --- A_x ---
        ax_ccc.h_view(n) = amp * RanGaussianSt(&rstate);
        ax_ccs.h_view(n) = (nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ax_csc.h_view(n) = (nky>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ax_css.h_view(n) = (nky>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ax_scc.h_view(n) = (nkx>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ax_scs.h_view(n) = (nkx>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ax_ssc.h_view(n) = (nkx>0 && nky>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ax_sss.h_view(n) = (nkx>0 && nky>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;

        // --- A_y ---
        ay_ccc.h_view(n) = amp * RanGaussianSt(&rstate);
        ay_ccs.h_view(n) = (nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ay_csc.h_view(n) = (nky>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ay_css.h_view(n) = (nky>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ay_scc.h_view(n) = (nkx>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ay_scs.h_view(n) = (nkx>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ay_ssc.h_view(n) = (nkx>0 && nky>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        ay_sss.h_view(n) = (nkx>0 && nky>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;

        // --- A_z ---
        az_ccc.h_view(n) = amp * RanGaussianSt(&rstate);
        az_ccs.h_view(n) = (nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        az_csc.h_view(n) = (nky>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        az_css.h_view(n) = (nky>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        az_scc.h_view(n) = (nkx>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        az_scs.h_view(n) = (nkx>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        az_ssc.h_view(n) = (nkx>0 && nky>0) ? amp * RanGaussianSt(&rstate) : 0.0;
        az_sss.h_view(n) = (nkx>0 && nky>0 && nkz>0) ? amp * RanGaussianSt(&rstate) : 0.0;

        n++;
      }
    }
  }

  // Sync all coefficient arrays to device
  kx_mode.template modify<HostMemSpace>(); kx_mode.template sync<DevExeSpace>();
  ky_mode.template modify<HostMemSpace>(); ky_mode.template sync<DevExeSpace>();
  kz_mode.template modify<HostMemSpace>(); kz_mode.template sync<DevExeSpace>();

  ax_ccc.template modify<HostMemSpace>(); ax_ccc.template sync<DevExeSpace>();
  ax_ccs.template modify<HostMemSpace>(); ax_ccs.template sync<DevExeSpace>();
  ax_csc.template modify<HostMemSpace>(); ax_csc.template sync<DevExeSpace>();
  ax_css.template modify<HostMemSpace>(); ax_css.template sync<DevExeSpace>();
  ax_scc.template modify<HostMemSpace>(); ax_scc.template sync<DevExeSpace>();
  ax_scs.template modify<HostMemSpace>(); ax_scs.template sync<DevExeSpace>();
  ax_ssc.template modify<HostMemSpace>(); ax_ssc.template sync<DevExeSpace>();
  ax_sss.template modify<HostMemSpace>(); ax_sss.template sync<DevExeSpace>();

  ay_ccc.template modify<HostMemSpace>(); ay_ccc.template sync<DevExeSpace>();
  ay_ccs.template modify<HostMemSpace>(); ay_ccs.template sync<DevExeSpace>();
  ay_csc.template modify<HostMemSpace>(); ay_csc.template sync<DevExeSpace>();
  ay_css.template modify<HostMemSpace>(); ay_css.template sync<DevExeSpace>();
  ay_scc.template modify<HostMemSpace>(); ay_scc.template sync<DevExeSpace>();
  ay_scs.template modify<HostMemSpace>(); ay_scs.template sync<DevExeSpace>();
  ay_ssc.template modify<HostMemSpace>(); ay_ssc.template sync<DevExeSpace>();
  ay_sss.template modify<HostMemSpace>(); ay_sss.template sync<DevExeSpace>();

  az_ccc.template modify<HostMemSpace>(); az_ccc.template sync<DevExeSpace>();
  az_ccs.template modify<HostMemSpace>(); az_ccs.template sync<DevExeSpace>();
  az_csc.template modify<HostMemSpace>(); az_csc.template sync<DevExeSpace>();
  az_css.template modify<HostMemSpace>(); az_css.template sync<DevExeSpace>();
  az_scc.template modify<HostMemSpace>(); az_scc.template sync<DevExeSpace>();
  az_scs.template modify<HostMemSpace>(); az_scs.template sync<DevExeSpace>();
  az_ssc.template modify<HostMemSpace>(); az_ssc.template sync<DevExeSpace>();
  az_sss.template modify<HostMemSpace>(); az_sss.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn SpectralICGenerator::PrecomputeTrigTables
//! \brief Computes sin/cos tables at face (left-edge) positions for each mode.
//! Face positions: x1f = LeftEdgeX(i-is, nx1, x1min, x1max) for i in [is, ie+1].

void SpectralICGenerator::PrecomputeTrigTables() {
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  bool is_2d = (indcs.nx2 == 1);  // true for slab / 1D+z problems
  bool is_1d = (indcs.nx3 == 1);  // true when z extent is trivial

  auto &size = pmy_pack->pmb->mb_size;
  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto kz_mode_ = kz_mode;
  auto xcos_f_  = xcos_f;
  auto xsin_f_  = xsin_f;
  auto ycos_f_  = ycos_f;
  auto ysin_f_  = ysin_f;
  auto zcos_f_  = zcos_f;
  auto zsin_f_  = zsin_f;
  int mode_count_ = mode_count;

  // x-faces: i in [is, ie+1]
  par_for("spec_ic_xface", DevExeSpace(), 0, nmb-1, 0, mode_count_-1, is, ie+1,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real x1min = size.d_view(m).x1min;
    Real x1max = size.d_view(m).x1max;
    Real x1f   = LeftEdgeX(i-is, nx1, x1min, x1max);
    Real k1v   = kx_mode_.d_view(n);
    xcos_f_(m,n,i) = cos(k1v*x1f);
    xsin_f_(m,n,i) = sin(k1v*x1f);
  });

  // y-faces: j in [js, je+1]
  // For a problem with nx2==1 (slab geometry), force cos=1, sin=0 so modes with
  // nky>0 contribute only through their cos(ky*y)=cos(0)=1 factor.
  par_for("spec_ic_yface", DevExeSpace(), 0, nmb-1, 0, mode_count_-1, js, je+1,
  KOKKOS_LAMBDA(int m, int n, int j) {
    if (is_2d) {
      ycos_f_(m,n,j) = 1.0;
      ysin_f_(m,n,j) = 0.0;
    } else {
      Real x2min = size.d_view(m).x2min;
      Real x2max = size.d_view(m).x2max;
      Real x2f   = LeftEdgeX(j-js, nx2, x2min, x2max);
      Real k2v   = ky_mode_.d_view(n);
      ycos_f_(m,n,j) = cos(k2v*x2f);
      ysin_f_(m,n,j) = sin(k2v*x2f);
    }
  });

  // z-faces: k in [ks, ke+1]
  par_for("spec_ic_zface", DevExeSpace(), 0, nmb-1, 0, mode_count_-1, ks, ke+1,
  KOKKOS_LAMBDA(int m, int n, int k) {
    if (is_1d) {
      zcos_f_(m,n,k) = 1.0;
      zsin_f_(m,n,k) = 0.0;
    } else {
      Real x3min = size.d_view(m).x3min;
      Real x3max = size.d_view(m).x3max;
      Real x3f   = LeftEdgeX(k-ks, nx3, x3min, x3max);
      Real k3v   = kz_mode_.d_view(n);
      zcos_f_(m,n,k) = cos(k3v*x3f);
      zsin_f_(m,n,k) = sin(k3v*x3f);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn SpectralICGenerator::GenerateVectorPotential
//! \brief Accumulates the spectral vector potential into ax, ay, az at all face/node
//! positions (indices is..ie+1, js..je+1, ks..ke+1).
//! Caller must zero-initialise ax, ay, az before calling this function.

void SpectralICGenerator::GenerateVectorPotential(DvceArray4D<Real> &ax,
                                                   DvceArray4D<Real> &ay,
                                                   DvceArray4D<Real> &az) {
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int mode_count_ = mode_count;

  // Capture device views of coefficient arrays
  auto ax_ccc_ = ax_ccc; auto ax_ccs_ = ax_ccs; auto ax_csc_ = ax_csc; auto ax_css_ = ax_css;
  auto ax_scc_ = ax_scc; auto ax_scs_ = ax_scs; auto ax_ssc_ = ax_ssc; auto ax_sss_ = ax_sss;
  auto ay_ccc_ = ay_ccc; auto ay_ccs_ = ay_ccs; auto ay_csc_ = ay_csc; auto ay_css_ = ay_css;
  auto ay_scc_ = ay_scc; auto ay_scs_ = ay_scs; auto ay_ssc_ = ay_ssc; auto ay_sss_ = ay_sss;
  auto az_ccc_ = az_ccc; auto az_ccs_ = az_ccs; auto az_csc_ = az_csc; auto az_css_ = az_css;
  auto az_scc_ = az_scc; auto az_scs_ = az_scs; auto az_ssc_ = az_ssc; auto az_sss_ = az_sss;
  auto xcos_f_ = xcos_f; auto xsin_f_ = xsin_f;
  auto ycos_f_ = ycos_f; auto ysin_f_ = ysin_f;
  auto zcos_f_ = zcos_f; auto zsin_f_ = zsin_f;

  // Outer loop over modes (sequential), inner loop over cells (parallel).
  // This matches the TurbulenceDriver::InitializeModes pattern.
  for (int n = 0; n < mode_count_; n++) {
    par_for("spec_ic_A", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real xc = xcos_f_(m,n,i), xs = xsin_f_(m,n,i);
      Real yc = ycos_f_(m,n,j), ys = ysin_f_(m,n,j);
      Real zc = zcos_f_(m,n,k), zs = zsin_f_(m,n,k);

      ax(m,k,j,i) +=
          ax_ccc_.d_view(n)*xc*yc*zc + ax_ccs_.d_view(n)*xc*yc*zs
        + ax_csc_.d_view(n)*xc*ys*zc + ax_css_.d_view(n)*xc*ys*zs
        + ax_scc_.d_view(n)*xs*yc*zc + ax_scs_.d_view(n)*xs*yc*zs
        + ax_ssc_.d_view(n)*xs*ys*zc + ax_sss_.d_view(n)*xs*ys*zs;

      ay(m,k,j,i) +=
          ay_ccc_.d_view(n)*xc*yc*zc + ay_ccs_.d_view(n)*xc*yc*zs
        + ay_csc_.d_view(n)*xc*ys*zc + ay_css_.d_view(n)*xc*ys*zs
        + ay_scc_.d_view(n)*xs*yc*zc + ay_scs_.d_view(n)*xs*yc*zs
        + ay_ssc_.d_view(n)*xs*ys*zc + ay_sss_.d_view(n)*xs*ys*zs;

      az(m,k,j,i) +=
          az_ccc_.d_view(n)*xc*yc*zc + az_ccs_.d_view(n)*xc*yc*zs
        + az_csc_.d_view(n)*xc*ys*zc + az_css_.d_view(n)*xc*ys*zs
        + az_scc_.d_view(n)*xs*yc*zc + az_scs_.d_view(n)*xs*yc*zs
        + az_ssc_.d_view(n)*xs*ys*zc + az_sss_.d_view(n)*xs*ys*zs;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn SpectralICGenerator::GenerateVectorPotentialFFT
//! \brief FFT-based alternative to GenerateVectorPotential.
//!
//! Fills A_hat in Fourier space with the same Gaussian random coefficients as the
//! direct method, then IFFTs to real space and scatters the result into ax/ay/az.
//!
//! Backend priority (compile-time):
//!   1. heFFTe (HEFFTE_ENABLED) — MPI-distributed r2c IFFT.
//!   2. KokkosFFT (FFT_ENABLED) — all ranks compute independently (no MPI comms).
//!   3. Direct synthesis fallback (always available).
//!
//! Returns: "heffte", "kokkos_fft", or "direct".

std::string SpectralICGenerator::GenerateVectorPotentialFFT(DvceArray4D<Real> &ax,
                                                             DvceArray4D<Real> &ay,
                                                             DvceArray4D<Real> &az) {
  Mesh *pm = pmy_pack->pmesh;
  auto &gindcs = pm->mesh_indcs;
  const int nx = gindcs.nx1;
  const int ny = gindcs.nx2;
  const int nz = gindcs.nx3;
  const int nzp1 = nz/2 + 1;  // size of r2c Fourier half-space in z

  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  Real dk1 = 2.0*M_PI/lx;
  Real dk2 = 2.0*M_PI/ly;
  Real dk3 = 2.0*M_PI/lz;

  const int nlow2  = nlow  * nlow;
  const int nhigh2 = nhigh * nhigh;

  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx_mb = indcs.nx1;
  const int ny_mb = indcs.nx2;
  const int nz_mb = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;

//----------------------------------------------------------------------------------------
// heFFTe path: MPI-distributed r2c IFFT
#if HEFFTE_ENABLED
  {
    int world_rank, world_nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_nranks);
    const int fft_nranks = std::min(world_nranks, nx);
    const bool participates = (world_rank < fft_nranks);

    MPI_Comm fft_comm = MPI_COMM_NULL;
    const int color = (participates ? 0 : MPI_UNDEFINED);
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &fft_comm);

    int fft_rank = -1, slab_x0 = 0, slab_x1 = -1, local_nx = 0;
    if (participates) {
      MPI_Comm_rank(fft_comm, &fft_rank);
      slab_x0 = (fft_rank * nx) / fft_nranks;
      slab_x1 = ((fft_rank + 1) * nx) / fft_nranks - 1;
      local_nx = slab_x1 - slab_x0 + 1;
    }

    using box3d = heffte::box3d<int>;
    const std::array<int, 3> mem_order = {2, 1, 0};  // z-fastest (row-major)
    std::unique_ptr<heffte::fft3d_r2c<heffte::backend::stock>> plan;
    if (participates) {
      box3d in_box({slab_x0, 0, 0}, {slab_x1, ny-1, nz-1}, mem_order);
      box3d out_box({slab_x0, 0, 0}, {slab_x1, ny-1, nz/2}, mem_order);
      plan = std::make_unique<heffte::fft3d_r2c<heffte::backend::stock>>(
          in_box, out_box, 2, fft_comm);
    }

    const int outbox_size = participates ? static_cast<int>(plan->size_outbox()) : 0;
    std::vector<std::complex<Real>> ax_hat(outbox_size, {0,0});
    std::vector<std::complex<Real>> ay_hat(outbox_size, {0,0});
    std::vector<std::complex<Real>> az_hat(outbox_size, {0,0});

    // Helper: flat index in slab for (gx, gy, gz): layout [local_x][y][z], z-fastest
    // idx = ((gx - slab_x0) * ny + gy) * nzp1 + gz
    RNG_State rng = rstate;  // deterministic copy of member RNG

    // Full global loop on ALL ranks (for RNG consistency).
    // kz=0 and kz=nz/2 planes need Hermitian-symmetry treatment.
    for (int gz = 0; gz < nzp1; ++gz) {
      const int kz = gz;  // always >= 0 in the half-space
      for (int gy = 0; gy < ny; ++gy) {
        const int ky = (gy <= ny/2 ? gy : gy - ny);
        for (int gx = 0; gx < nx; ++gx) {
          const int kx = (gx <= nx/2 ? gx : gx - nx);

          // Skip DC mode
          if (kx == 0 && ky == 0 && kz == 0) continue;

          const int ksq = kx*kx + ky*ky + kz*kz;
          const bool in_band = (ksq >= nlow2 && ksq <= nhigh2);

          if (kz == 0 || (nz % 2 == 0 && kz == nz/2)) {
            // Hermitian-symmetry plane: each canonical mode represents itself and
            // its conjugate at (-kx,-ky,kz).  Only draw RNG for the canonical mode.
            const int gx_c = (kx == 0 ? 0 : nx - gx);
            const int gy_c = (ky == 0 ? 0 : ny - gy);
            const bool self_conj = (gx_c == gx && gy_c == gy);
            const bool is_canon = self_conj ||
                                  (gx_c > gx) ||
                                  (gx_c == gx && gy_c > gy);
            if (!is_canon) continue;  // conjugate already processed; no RNG draw

            if (in_band) {
              const Real n_mag = std::sqrt(static_cast<Real>(kx*kx + ky*ky + kz*kz));
              const Real amp = ModeAmplitude(n_mag);
              const Real Gr_x = amp * RanGaussianSt(&rng);
              const Real Gr_y = amp * RanGaussianSt(&rng);
              const Real Gr_z = amp * RanGaussianSt(&rng);

              if (participates && gx >= slab_x0 && gx <= slab_x1) {
                const int64_t id = (static_cast<int64_t>(gx - slab_x0) * ny + gy) * nzp1 + gz;
                ax_hat[id] = {Gr_x, 0.0};
                ay_hat[id] = {Gr_y, 0.0};
                az_hat[id] = {Gr_z, 0.0};
              }
              if (!self_conj) {
                if (participates && gx_c >= slab_x0 && gx_c <= slab_x1) {
                  const int64_t id_c = (static_cast<int64_t>(gx_c - slab_x0) * ny + gy_c) * nzp1 + gz;
                  ax_hat[id_c] = {Gr_x, 0.0};   // conj of (Gr_x+0i) = same
                  ay_hat[id_c] = {Gr_y, 0.0};
                  az_hat[id_c] = {Gr_z, 0.0};
                }
              }
            }
          } else {
            // Generic complex mode: 0 < kz < nz/2
            if (in_band) {
              const Real n_mag = std::sqrt(static_cast<Real>(kx*kx + ky*ky + kz*kz));
              const Real amp = ModeAmplitude(n_mag);
              const Real Gr_x = amp * RanGaussianSt(&rng);
              const Real Gi_x = amp * RanGaussianSt(&rng);
              const Real Gr_y = amp * RanGaussianSt(&rng);
              const Real Gi_y = amp * RanGaussianSt(&rng);
              const Real Gr_z = amp * RanGaussianSt(&rng);
              const Real Gi_z = amp * RanGaussianSt(&rng);

              if (participates && gx >= slab_x0 && gx <= slab_x1) {
                const int64_t id = (static_cast<int64_t>(gx - slab_x0) * ny + gy) * nzp1 + gz;
                ax_hat[id] = {Gr_x, Gi_x};
                ay_hat[id] = {Gr_y, Gi_y};
                az_hat[id] = {Gr_z, Gi_z};
              }
            }
          }
        }
      }
    }

    // Execute IFFT (complex outbox → real inbox) with full normalization
    const int inbox_size = participates ? static_cast<int>(plan->size_inbox()) : 0;
    std::vector<Real> ax_real(inbox_size, 0.0);
    std::vector<Real> ay_real(inbox_size, 0.0);
    std::vector<Real> az_real(inbox_size, 0.0);

    if (participates) {
      plan->backward(ax_hat.data(), ax_real.data(), heffte::scale::full);
      plan->backward(ay_hat.data(), ay_real.data(), heffte::scale::full);
      plan->backward(az_hat.data(), az_real.data(), heffte::scale::full);
    }

    // Build gx_to_owner lookup (consistent with slab boundaries)
    std::vector<int> gx_to_owner(nx);
    for (int r = 0; r < fft_nranks; ++r) {
      const int x0 = (r * nx) / fft_nranks;
      const int x1 = ((r + 1) * nx) / fft_nranks - 1;
      for (int gx = x0; gx <= x1; ++gx) gx_to_owner[gx] = r;
    }

    // Scatter from slab to MeshBlock owners via Alltoallv.
    // Each slab rank sends cells to the world rank that owns the corresponding MB.
    // We build the reverse: each MB rank sends requests by GID; slab owner replies.
    // Simpler approach: slab rank scatters to all MB owners (mirroring the gather).

    // Step A: each MB rank packs (gx, gy, gz) → world_rank (slab owner) for each cell
    struct CellSample {
      int gx, gy, gz;
      Real val_ax, val_ay, val_az;
    };

    // Each slab rank builds what it will send to each world rank.
    // We need: for each cell (gx,gy,gz) in this slab, who is the MB owner?
    // We cannot know this directly, so instead:
    // - Each MB rank announces which global cells it needs.
    // - Slab rank fills those requests.
    // This is equivalent to a gather from slab → MB owners.

    // Simpler: slab owners push data; MB owners receive.
    // We iterate over all slab cells and determine the target rank from MB layout.
    // Since mesh is uniform, cell (gx,gy,gz) → MB index (gx/nx_mb, gy/ny_mb, gz/nz_mb)
    // → the rank that owns that MB.

    // Build MB GID → world rank lookup from the mesh's loc_eachmb
    const int nmb_total = pm->nmb_total;
    std::vector<int> mb_owner(nmb_total, 0);

#if MPI_PARALLEL_ENABLED
    {
      // Each rank owns pmy_pack->nmb_thispack MBs; communicate owners to all
      std::vector<int> local_gids_v(nmb);
      for (int m = 0; m < nmb; ++m)
        local_gids_v[m] = pmy_pack->pmb->mb_gid.h_view(m);

      std::vector<int> nmb_per_rank(world_nranks, 0);
      MPI_Allgather(&nmb, 1, MPI_INT, nmb_per_rank.data(), 1, MPI_INT, MPI_COMM_WORLD);
      std::vector<int> displs(world_nranks, 0);
      for (int r = 1; r < world_nranks; ++r)
        displs[r] = displs[r-1] + nmb_per_rank[r-1];
      std::vector<int> all_gids(nmb_total);
      MPI_Allgatherv(local_gids_v.data(), nmb, MPI_INT,
                     all_gids.data(), nmb_per_rank.data(), displs.data(),
                     MPI_INT, MPI_COMM_WORLD);
      for (int r = 0; r < world_nranks; ++r)
        for (int b = 0; b < nmb_per_rank[r]; ++b)
          mb_owner[all_gids[displs[r] + b]] = r;
    }
#else
    for (int g = 0; g < nmb_total; ++g) mb_owner[g] = 0;
#endif

    // Build gx_to_mb_rank: for global x-index gx, which world rank owns the cell?
    // gx → lx1 = gx / nx_mb → look up MB with lx1, find its owner
    // We use pm->lloc_eachmb for this.
    // For a uniform mesh, each cell column gx maps to MB lx1 = gx/nx_mb (integer div).
    // We need a function: (lx1, lx2, lx3) → gid.  Use lloc_eachmb to build reverse.
    std::vector<int> lloc_to_gid(nmb_total, -1);
    // lloc_eachmb[gid] → {lx1, lx2, lx3, level}
    const int nmb_x = nx / nx_mb;   // number of MBs in x
    const int nmb_y = ny / ny_mb;
    const int nmb_z = nz / nz_mb;
    for (int gid = 0; gid < nmb_total; ++gid) {
      const auto &lloc = pm->lloc_eachmb[gid];
      const int lx1 = static_cast<int>(lloc.lx1);
      const int lx2 = static_cast<int>(lloc.lx2);
      const int lx3 = static_cast<int>(lloc.lx3);
      const int flat = (lx3 * nmb_y + lx2) * nmb_x + lx1;
      lloc_to_gid[flat] = gid;
    }

    // For each global cell (gx,gy,gz), world rank = mb_owner[gid of that cell's MB]
    auto cell_owner = [&](int gx_, int gy_, int gz_) -> int {
      const int lx1 = gx_ / nx_mb;
      const int lx2 = gy_ / ny_mb;
      const int lx3 = gz_ / nz_mb;
      const int flat = (lx3 * nmb_y + lx2) * nmb_x + lx1;
      return mb_owner[lloc_to_gid[flat]];
    };

    // Slab rank builds send lists for each world rank
    std::vector<std::vector<CellSample>> send_lists(world_nranks);

    if (participates) {
      for (int ix = 0; ix < local_nx; ++ix) {
        const int gx = slab_x0 + ix;
        for (int gy = 0; gy < ny; ++gy) {
          for (int gz = 0; gz < nz; ++gz) {
            const int64_t id = (static_cast<int64_t>(ix) * ny + gy) * nz + gz;
            const int tgt = cell_owner(gx, gy, gz);
            send_lists[tgt].push_back({gx, gy, gz,
                                       ax_real[id], ay_real[id], az_real[id]});
          }
        }
      }
    }

    // Exchange counts
    std::vector<int> send_counts(world_nranks, 0), recv_counts(world_nranks, 0);
    for (int r = 0; r < world_nranks; ++r)
      send_counts[r] = static_cast<int>(send_lists[r].size());
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> send_displs(world_nranks, 0), recv_displs(world_nranks, 0);
    for (int r = 1; r < world_nranks; ++r) {
      send_displs[r] = send_displs[r-1] + send_counts[r-1];
      recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
    }
    const int total_send = send_displs[world_nranks-1] + send_counts[world_nranks-1];
    const int total_recv = recv_displs[world_nranks-1] + recv_counts[world_nranks-1];

    std::vector<CellSample> send_flat(total_send);
    for (int r = 0; r < world_nranks; ++r)
      std::copy(send_lists[r].begin(), send_lists[r].end(),
                send_flat.begin() + send_displs[r]);
    send_lists.clear();

    std::vector<CellSample> recv_flat(total_recv);

    std::vector<int> sc_b(world_nranks), rc_b(world_nranks);
    std::vector<int> sd_b(world_nranks), rd_b(world_nranks);
    const int csz = static_cast<int>(sizeof(CellSample));
    for (int r = 0; r < world_nranks; ++r) {
      sc_b[r] = send_counts[r] * csz;
      rc_b[r] = recv_counts[r] * csz;
      sd_b[r] = send_displs[r] * csz;
      rd_b[r] = recv_displs[r] * csz;
    }
    MPI_Alltoallv(reinterpret_cast<char*>(send_flat.data()), sc_b.data(),
                  sd_b.data(), MPI_BYTE,
                  reinterpret_cast<char*>(recv_flat.data()), rc_b.data(),
                  rd_b.data(), MPI_BYTE, MPI_COMM_WORLD);

    // Build lookup: global (gx,gy,gz) → ax/ay/az values from received data
    // We need to scatter into ax/ay/az device arrays.
    // First collect into host maps indexed by (gx*ny+gy)*nz+gz for this rank's cells.
    // But cells can be large; use a flat host array indexed by MB-local position.
    // Copy received cells into a host mirror of ax/ay/az, then deep_copy to device.
    auto ax_h = Kokkos::create_mirror_view(ax);
    auto ay_h = Kokkos::create_mirror_view(ay);
    auto az_h = Kokkos::create_mirror_view(az);
    Kokkos::deep_copy(ax_h, ax);
    Kokkos::deep_copy(ay_h, ay);
    Kokkos::deep_copy(az_h, az);

    // Build a reverse map from (gx,gy,gz) → (m, k, j, i) for local MBs
    // For each MB m, x0_mb = lloc.lx1 * nx_mb, etc.
    // Cell (gx,gy,gz) → m: find MB where gx in [x0, x0+nx_mb-1], etc.
    // This is safe because we only receive cells that belong to our MBs.
    // Build x0/y0/z0 for each local MB:
    std::vector<int> mb_x0(nmb), mb_y0(nmb), mb_z0(nmb);
    for (int m = 0; m < nmb; ++m) {
      const int gid = pmy_pack->pmb->mb_gid.h_view(m);
      const auto &lloc = pm->lloc_eachmb[gid];
      mb_x0[m] = static_cast<int>(lloc.lx1) * nx_mb;
      mb_y0[m] = static_cast<int>(lloc.lx2) * ny_mb;
      mb_z0[m] = static_cast<int>(lloc.lx3) * nz_mb;
    }

    for (const auto &s : recv_flat) {
      // Find which local MB owns (s.gx, s.gy, s.gz)
      for (int m = 0; m < nmb; ++m) {
        const int li = s.gx - mb_x0[m];
        const int lj = s.gy - mb_y0[m];
        const int lk = s.gz - mb_z0[m];
        if (li < 0 || li >= nx_mb) continue;
        if (lj < 0 || lj >= ny_mb) continue;
        if (lk < 0 || lk >= nz_mb) continue;
        // Map to [is..ie+1, js..je+1, ks..ke+1]
        const int i = is + li;
        const int j = js + lj;
        const int k = ks + lk;
        ax_h(m, k, j, i) = s.val_ax;
        ay_h(m, k, j, i) = s.val_ay;
        az_h(m, k, j, i) = s.val_az;
        // Periodic wrap for extra face at upper boundary
        if (li == nx_mb - 1) {
          const int gx_w = (s.gx + 1) % nx;
          // Extra face value at i+1: same as gx_w in the global array
          // These will be filled when that cell's sample arrives.
          // The face at i=ie+1 gets ax(gx+1) which is a different global cell.
          // Just store at i+1 position using modular wrap of the FFT output.
          // We'll handle this after all cells are placed (see wrap pass below).
          (void)gx_w;
        }
        break;
      }
    }

    // Extra face values: ax at i=ie+1, ay at j=je+1, az at k=ke+1
    // These are ax_real[(gx+1) % nx][gy][gz], etc.  We need to handle the
    // periodic boundary wrap.  Because the received data only covers [0,nx_mb-1]
    // relative to MB origin, we need a second pass for upper-edge faces.
    // Easiest: build a host flat array for global values and index into it.
    // For the FFT path, we already have ax_real (only for participates rank).
    // Instead, we gather the extra face values via the same Alltoallv mechanism.
    // However, this is complex.  A simpler approach: repeat the scatter with
    // the periodic wrap indices.

    // Actually, for a uniform periodic mesh the extra face at i=ie+1 of MB m
    // equals the first interior face of MB m+1 (or MB 0 if m is the last in x).
    // We already receive that cell for the adjacent MB.
    // So we just need to copy: for each local MB m,
    //   ax_h(m, k, j, ie+1) = ax_h(m_right, k, j, is)
    // where m_right is the MB to the right of m (with periodic wrap).
    // This requires knowing the right-neighbour MB.  We use pm->lloc_eachmb.

    // Build a (lx1,lx2,lx3) → local_m map for local MBs
    std::vector<int> local_mb_lmap(nmb_total, -1);
    for (int m = 0; m < nmb; ++m) {
      const int gid = pmy_pack->pmb->mb_gid.h_view(m);
      local_mb_lmap[gid] = m;
    }

    // Fill upper-boundary faces for each local MB
    for (int m = 0; m < nmb; ++m) {
      const int gid = pmy_pack->pmb->mb_gid.h_view(m);
      const auto &lloc = pm->lloc_eachmb[gid];

      // Right neighbour in x (periodic)
      {
        const int lx1_r = (static_cast<int>(lloc.lx1) + 1) % nmb_x;
        const int flat_r = (static_cast<int>(lloc.lx3) * nmb_y
                           + static_cast<int>(lloc.lx2)) * nmb_x + lx1_r;
        const int gid_r = lloc_to_gid[flat_r];
        const int m_r = local_mb_lmap[gid_r];
        if (m_r >= 0) {
          for (int k = ks; k <= ke+1; ++k)
            for (int j = js; j <= je+1; ++j)
              ax_h(m, k, j, ie+1) = ax_h(m_r, k, j, is);
        }
      }

      // Top neighbour in y (periodic)
      {
        const int lx2_t = (static_cast<int>(lloc.lx2) + 1) % nmb_y;
        const int flat_t = (static_cast<int>(lloc.lx3) * nmb_y + lx2_t) * nmb_x
                          + static_cast<int>(lloc.lx1);
        const int gid_t = lloc_to_gid[flat_t];
        const int m_t = local_mb_lmap[gid_t];
        if (m_t >= 0) {
          for (int k = ks; k <= ke+1; ++k)
            for (int i = is; i <= ie+1; ++i)
              ay_h(m, k, je+1, i) = ay_h(m_t, k, js, i);
        }
      }

      // Front neighbour in z (periodic)
      {
        const int lx3_f = (static_cast<int>(lloc.lx3) + 1) % nmb_z;
        const int flat_f = (lx3_f * nmb_y + static_cast<int>(lloc.lx2)) * nmb_x
                          + static_cast<int>(lloc.lx1);
        const int gid_f = lloc_to_gid[flat_f];
        const int m_f = local_mb_lmap[gid_f];
        if (m_f >= 0) {
          for (int j = js; j <= je+1; ++j)
            for (int i = is; i <= ie+1; ++i)
              az_h(m, ke+1, j, i) = az_h(m_f, ks, j, i);
        }
      }
    }

    Kokkos::deep_copy(ax, ax_h);
    Kokkos::deep_copy(ay, ay_h);
    Kokkos::deep_copy(az, az_h);

    if (fft_comm != MPI_COMM_NULL) MPI_Comm_free(&fft_comm);
    return "heffte";
  }

//----------------------------------------------------------------------------------------
// KokkosFFT path: all ranks compute the full IFFT independently (no MPI comms needed)
#elif FFT_ENABLED
  {
    using h_complex_t = Kokkos::View<Kokkos::complex<Real>***,
                                     Kokkos::LayoutRight, Kokkos::HostSpace>;
    using d_complex_t = Kokkos::View<Kokkos::complex<Real>***,
                                     Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
    using d_real_t = Kokkos::View<Real***,
                                  Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

    // Allocate full host complex arrays [nx][ny][nzp1]
    h_complex_t ax_hat_h("ax_hat_h", nx, ny, nzp1);
    h_complex_t ay_hat_h("ay_hat_h", nx, ny, nzp1);
    h_complex_t az_hat_h("az_hat_h", nx, ny, nzp1);
    Kokkos::deep_copy(ax_hat_h, Kokkos::complex<Real>(0, 0));
    Kokkos::deep_copy(ay_hat_h, Kokkos::complex<Real>(0, 0));
    Kokkos::deep_copy(az_hat_h, Kokkos::complex<Real>(0, 0));

    RNG_State rng = rstate;  // deterministic copy of member RNG

    // Fill A_hat: same loop structure as heFFTe path but storing the full array.
    for (int gz = 0; gz < nzp1; ++gz) {
      const int kz = gz;
      for (int gy = 0; gy < ny; ++gy) {
        const int ky = (gy <= ny/2 ? gy : gy - ny);
        for (int gx = 0; gx < nx; ++gx) {
          const int kx = (gx <= nx/2 ? gx : gx - nx);

          if (kx == 0 && ky == 0 && kz == 0) continue;

          const int ksq = kx*kx + ky*ky + kz*kz;
          const bool in_band = (ksq >= nlow2 && ksq <= nhigh2);

          if (kz == 0 || (nz % 2 == 0 && kz == nz/2)) {
            const int gx_c = (kx == 0 ? 0 : nx - gx);
            const int gy_c = (ky == 0 ? 0 : ny - gy);
            const bool self_conj = (gx_c == gx && gy_c == gy);
            const bool is_canon = self_conj ||
                                  (gx_c > gx) ||
                                  (gx_c == gx && gy_c > gy);
            if (!is_canon) continue;

            if (in_band) {
              const Real n_mag = std::sqrt(static_cast<Real>(kx*kx + ky*ky + kz*kz));
              const Real amp = ModeAmplitude(n_mag);
              const Real Gr_x = amp * RanGaussianSt(&rng);
              const Real Gr_y = amp * RanGaussianSt(&rng);
              const Real Gr_z = amp * RanGaussianSt(&rng);

              ax_hat_h(gx, gy, gz) = Kokkos::complex<Real>(Gr_x, 0.0);
              ay_hat_h(gx, gy, gz) = Kokkos::complex<Real>(Gr_y, 0.0);
              az_hat_h(gx, gy, gz) = Kokkos::complex<Real>(Gr_z, 0.0);
              if (!self_conj) {
                ax_hat_h(gx_c, gy_c, gz) = Kokkos::complex<Real>(Gr_x, 0.0);
                ay_hat_h(gx_c, gy_c, gz) = Kokkos::complex<Real>(Gr_y, 0.0);
                az_hat_h(gx_c, gy_c, gz) = Kokkos::complex<Real>(Gr_z, 0.0);
              }
            }
          } else {
            // 0 < kz < nz/2
            if (in_band) {
              const Real n_mag = std::sqrt(static_cast<Real>(kx*kx + ky*ky + kz*kz));
              const Real amp = ModeAmplitude(n_mag);
              const Real Gr_x = amp * RanGaussianSt(&rng);
              const Real Gi_x = amp * RanGaussianSt(&rng);
              const Real Gr_y = amp * RanGaussianSt(&rng);
              const Real Gi_y = amp * RanGaussianSt(&rng);
              const Real Gr_z = amp * RanGaussianSt(&rng);
              const Real Gi_z = amp * RanGaussianSt(&rng);

              ax_hat_h(gx, gy, gz) = Kokkos::complex<Real>(Gr_x, Gi_x);
              ay_hat_h(gx, gy, gz) = Kokkos::complex<Real>(Gr_y, Gi_y);
              az_hat_h(gx, gy, gz) = Kokkos::complex<Real>(Gr_z, Gi_z);
            }
          }
        }
      }
    }

    // Copy to device
    d_complex_t ax_hat_d = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(), ax_hat_h);
    d_complex_t ay_hat_d = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(), ay_hat_h);
    d_complex_t az_hat_d = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(), az_hat_h);

    // Allocate output real arrays on device [nx][ny][nz]
    d_real_t ax_real_d("ax_real_d", nx, ny, nz);
    d_real_t ay_real_d("ay_real_d", nx, ny, nz);
    d_real_t az_real_d("az_real_d", nx, ny, nz);

    // Execute backward (c2r) 3D FFT using KokkosFFT.
    // Normalization::backward divides by N, giving the correct IFFT result.
    {
      using BackPlanType = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace,
                                           d_complex_t, d_real_t, 3>;
      BackPlanType plan_x(Kokkos::DefaultExecutionSpace(), ax_hat_d, ax_real_d,
                          KokkosFFT::Direction::backward,
                          std::array<int,3>{0,1,2});
      KokkosFFT::execute(plan_x, ax_hat_d, ax_real_d,
                         KokkosFFT::Normalization::backward);
    }
    {
      using BackPlanType = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace,
                                           d_complex_t, d_real_t, 3>;
      BackPlanType plan_y(Kokkos::DefaultExecutionSpace(), ay_hat_d, ay_real_d,
                          KokkosFFT::Direction::backward,
                          std::array<int,3>{0,1,2});
      KokkosFFT::execute(plan_y, ay_hat_d, ay_real_d,
                         KokkosFFT::Normalization::backward);
    }
    {
      using BackPlanType = KokkosFFT::Plan<Kokkos::DefaultExecutionSpace,
                                           d_complex_t, d_real_t, 3>;
      BackPlanType plan_z(Kokkos::DefaultExecutionSpace(), az_hat_d, az_real_d,
                          KokkosFFT::Direction::backward,
                          std::array<int,3>{0,1,2});
      KokkosFFT::execute(plan_z, az_hat_d, az_real_d,
                         KokkosFFT::Normalization::backward);
    }

    // Build MB offset arrays (host → device)
    Kokkos::View<int*, Kokkos::HostSpace> mb_x0_h("mb_x0_h", nmb);
    Kokkos::View<int*, Kokkos::HostSpace> mb_y0_h("mb_y0_h", nmb);
    Kokkos::View<int*, Kokkos::HostSpace> mb_z0_h("mb_z0_h", nmb);
    for (int m = 0; m < nmb; ++m) {
      const int gid = pmy_pack->pmb->mb_gid.h_view(m);
      const auto &lloc = pm->lloc_eachmb[gid];
      mb_x0_h(m) = static_cast<int>(lloc.lx1) * nx_mb;
      mb_y0_h(m) = static_cast<int>(lloc.lx2) * ny_mb;
      mb_z0_h(m) = static_cast<int>(lloc.lx3) * nz_mb;
    }
    auto mb_x0_d = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(), mb_x0_h);
    auto mb_y0_d = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(), mb_y0_h);
    auto mb_z0_d = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(), mb_z0_h);

    // Scatter IFFT result into ax/ay/az with periodic wrapping for the extra face
    const int nx_ = nx, ny_ = ny, nz_ = nz;
    par_for("spec_fft_scatter", DevExeSpace(),
            0, nmb-1, ks, ke+1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      int gx = (mb_x0_d(m) + (i - is)) % nx_;
      int gy = (mb_y0_d(m) + (j - js)) % ny_;
      int gz = (mb_z0_d(m) + (k - ks)) % nz_;
      ax(m, k, j, i) = ax_real_d(gx, gy, gz);
      ay(m, k, j, i) = ay_real_d(gx, gy, gz);
      az(m, k, j, i) = az_real_d(gx, gy, gz);
    });

    return "kokkos_fft";
  }

//----------------------------------------------------------------------------------------
// Fallback: direct Fourier synthesis
#else
  GenerateVectorPotential(ax, ay, az);
  return "direct";
#endif
}

//========================================================================================
// Free functions for post-processing the face-centered B field
//========================================================================================

//----------------------------------------------------------------------------------------
//! \fn SubtractGlobalMeanB
//! \brief Subtracts the MPI-global volume-averaged B from all face-field values.
//! This centres the turbulent field around zero net flux in each direction.

void SubtractGlobalMeanB(MeshBlockPack *pmbp, DvceFaceFld4D<Real> &b0) {
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &gindcs = pmbp->pmesh->mesh_indcs;
  int gnx1 = gindcs.nx1, gnx2 = gindcs.nx2, gnx3 = gindcs.nx3;
  Real ncells_total = static_cast<Real>(gnx1) * static_cast<Real>(gnx2)
                    * static_cast<Real>(gnx3);

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji  = nx3*nx2*nx1;
  const int nji   = nx2*nx1;

  // Sum cell-centred B over active cells (average of neighbouring face values)
  Real mean_bx = 0.0, mean_by = 0.0, mean_bz = 0.0;
  Kokkos::parallel_reduce("spec_ic_mean_B",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sbx, Real &sby, Real &sbz) {
      int m = idx / nkji;
      int k = (idx - m*nkji) / nji;
      int j = (idx - m*nkji - k*nji) / nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      sbx += 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      sby += 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      sbz += 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    },
    Kokkos::Sum<Real>(mean_bx),
    Kokkos::Sum<Real>(mean_by),
    Kokkos::Sum<Real>(mean_bz));

#if MPI_PARALLEL_ENABLED
  Real loc[3] = {mean_bx, mean_by, mean_bz};
  Real glb[3];
  MPI_Allreduce(loc, glb, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  mean_bx = glb[0]; mean_by = glb[1]; mean_bz = glb[2];
#endif

  mean_bx /= ncells_total;
  mean_by /= ncells_total;
  mean_bz /= ncells_total;

  par_for("spec_ic_sub_mean", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b0.x1f(m,k,j,i) -= mean_bx;
    b0.x2f(m,k,j,i) -= mean_by;
    b0.x3f(m,k,j,i) -= mean_bz;
    if (i == ie) b0.x1f(m,k,j,i+1) -= mean_bx;
    if (j == je) b0.x2f(m,k,j+1,i) -= mean_by;
    if (k == ke) b0.x3f(m,k+1,j,i) -= mean_bz;
  });
}

//----------------------------------------------------------------------------------------
//! \fn NormalizeRmsB
//! \brief Rescales all face-field values so that sqrt(<|B|^2>) = rms_target.

void NormalizeRmsB(MeshBlockPack *pmbp, DvceFaceFld4D<Real> &b0, Real rms_target) {
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &gindcs = pmbp->pmesh->mesh_indcs;
  int gnx1 = gindcs.nx1, gnx2 = gindcs.nx2, gnx3 = gindcs.nx3;
  Real ncells_total = static_cast<Real>(gnx1) * static_cast<Real>(gnx2)
                    * static_cast<Real>(gnx3);

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji  = nx3*nx2*nx1;
  const int nji   = nx2*nx1;

  Real sum_bsq = 0.0;
  Kokkos::parallel_reduce("spec_ic_rms_B",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sbsq) {
      int m = idx / nkji;
      int k = (idx - m*nkji) / nji;
      int j = (idx - m*nkji - k*nji) / nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
      sbsq += bx*bx + by*by + bz*bz;
    },
    Kokkos::Sum<Real>(sum_bsq));

#if MPI_PARALLEL_ENABLED
  Real glb;
  MPI_Allreduce(&sum_bsq, &glb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  sum_bsq = glb;
#endif

  Real rms = std::sqrt(sum_bsq / ncells_total);
  if (rms < 1.0e-20) {
    std::cout << "### WARNING SpectralICGenerator: RMS |B| is near zero; "
              << "skipping normalization." << std::endl;
    return;
  }
  Real scale = rms_target / rms;

  par_for("spec_ic_norm", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b0.x1f(m,k,j,i) *= scale;
    b0.x2f(m,k,j,i) *= scale;
    b0.x3f(m,k,j,i) *= scale;
    if (i == ie) b0.x1f(m,k,j,i+1) *= scale;
    if (j == je) b0.x2f(m,k,j+1,i) *= scale;
    if (k == ke) b0.x3f(m,k+1,j,i) *= scale;
  });
}
