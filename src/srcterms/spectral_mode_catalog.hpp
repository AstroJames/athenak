#ifndef SRCTERMS_SPECTRAL_MODE_CATALOG_HPP_
#define SRCTERMS_SPECTRAL_MODE_CATALOG_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectral_mode_catalog.hpp
//! \brief Deterministic complex Hermitian half-spectrum mode sets for source drivers.

#include <array>
#include <string>
#include <vector>

namespace spectral_modes {

enum class GuideAxis {invalid = -1, x1 = 0, x2 = 1, x3 = 2};

struct Mode {
  std::array<int, 3> n;
  double signed_degeneracy = 2.0;
};

inline GuideAxis ParseGuideAxis(const std::string &name) {
  if (name == "x" || name == "x1") return GuideAxis::x1;
  if (name == "y" || name == "x2") return GuideAxis::x2;
  if (name == "z" || name == "x3") return GuideAxis::x3;
  return GuideAxis::invalid;
}

inline std::array<int, 2> TransverseAxes(GuideAxis guide) {
  const int g = static_cast<int>(guide);
  if (g < 0 || g > 2) return {-1, -1};
  std::array<int, 2> transverse = {0, 1};
  int index = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis != g) transverse[index++] = axis;
  }
  return transverse;
}

inline std::vector<Mode> Zhdankin8(GuideAxis guide) {
  const int g = static_cast<int>(guide);
  if (g < 0 || g > 2) return {};
  const auto transverse = TransverseAxes(guide);
  std::vector<Mode> modes(4);
  modes[0].n = {0, 0, 0};
  modes[1].n = {0, 0, 0};
  modes[2].n = {0, 0, 0};
  modes[3].n = {0, 0, 0};
  modes[0].n[g] = 1;
  modes[1].n[g] = 1;
  modes[2].n[g] = 1;
  modes[3].n[g] = 1;
  modes[0].n[transverse[0]] = 1;
  modes[1].n[transverse[1]] = 1;
  modes[2].n[transverse[0]] = -1;
  modes[3].n[transverse[1]] = -1;
  return modes;
}

inline std::vector<Mode> AnisotropicBand(GuideAxis guide, int nperp_min,
                                         int nperp_max, int nparallel_min,
                                         int nparallel_max) {
  const int g = static_cast<int>(guide);
  if (g < 0 || g > 2) return {};
  const auto transverse = TransverseAxes(guide);
  const int nperp_min2 = nperp_min*nperp_min;
  const int nperp_max2 = nperp_max*nperp_max;
  std::vector<Mode> modes;
  for (int nparallel = nparallel_min; nparallel <= nparallel_max; ++nparallel) {
    for (int nt1 = -nperp_max; nt1 <= nperp_max; ++nt1) {
      for (int nt2 = -nperp_max; nt2 <= nperp_max; ++nt2) {
        const int nperp2 = nt1*nt1 + nt2*nt2;
        if (nperp2 < nperp_min2 || nperp2 > nperp_max2 || nperp2 == 0) continue;
        Mode mode;
        mode.n = {0, 0, 0};
        mode.n[g] = nparallel;
        mode.n[transverse[0]] = nt1;
        mode.n[transverse[1]] = nt2;
        modes.push_back(mode);
      }
    }
  }
  return modes;
}

} // namespace spectral_modes

#endif  // SRCTERMS_SPECTRAL_MODE_CATALOG_HPP_
