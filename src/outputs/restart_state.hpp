#ifndef OUTPUTS_RESTART_STATE_HPP_
#define OUTPUTS_RESTART_STATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart_state.hpp
//! \brief Version and fixed sizes for optional restart state owned by source modules.

namespace restart_state {

inline constexpr int version = 1;
inline constexpr int turbulence_diagnostics = 9;
inline constexpr int cooling_diagnostics = 10;

} // namespace restart_state

#endif  // OUTPUTS_RESTART_STATE_HPP_
