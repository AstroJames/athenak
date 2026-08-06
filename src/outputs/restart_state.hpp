#ifndef OUTPUTS_RESTART_STATE_HPP_
#define OUTPUTS_RESTART_STATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart_state.hpp
//! \brief Version and fixed sizes for optional restart state owned by source modules.

#include <cstdint>

namespace restart_state {

inline constexpr int version = 3;
inline constexpr int turbulence_diagnostics = 9;
inline constexpr int cooling_diagnostics = 10;
inline constexpr int antenna_diagnostics = 18;
inline constexpr int antenna_v2_diagnostics = 16;
inline constexpr int antenna_v2_modes = 16;
inline constexpr std::uint64_t antenna_values_per_mode = 4;
inline constexpr std::uint64_t antenna_record_magic = 0x414e54454e4e4133ULL;
inline constexpr std::uint64_t antenna_record_schema = 2;
inline constexpr std::uint64_t antenna_header_words = 7;
inline constexpr std::uint64_t antenna_header_bytes =
    antenna_header_words*sizeof(std::uint64_t);

inline constexpr std::uint64_t AntennaPayloadBytes(
    std::uint64_t num_modes, std::uint64_t rng_bytes,
    std::uint64_t real_bytes) {
  return rng_bytes
         + (antenna_diagnostics + antenna_values_per_mode*num_modes)*real_bytes;
}

} // namespace restart_state

#endif  // OUTPUTS_RESTART_STATE_HPP_
