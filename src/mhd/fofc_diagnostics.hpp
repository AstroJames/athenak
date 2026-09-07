#ifndef MHD_FOFC_DIAGNOSTICS_HPP_
#define MHD_FOFC_DIAGNOSTICS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2026 James Beattie and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

class Driver;
class MeshBlockPack;

namespace mhd {
class MHD;

// Failure-only host snapshot. No state is modified and no MPI collective is used.
void DumpFOFCFailure(MHD *mhd, MeshBlockPack *pack, Driver *driver, int stage,
                     int last_recheck);
} // namespace mhd
#endif // MHD_FOFC_DIAGNOSTICS_HPP_
