#ifndef MHD_FOFC_BOUNDARY_HPP_
#define MHD_FOFC_BOUNDARY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fofc_boundary.hpp
//! \brief One-cell correction-mask exchange for iterative FOFC on uniform meshes.

#include <cstddef>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"

namespace mhd {

// Only boundary masks are staged on the host. Interior states and flags stay on device.
// Fixed correction rounds allow nearest-neighbor exchanges without a global reduction.
class FOFCBoundary {
 public:
  FOFCBoundary(MeshBlockPack *pack, MeshBoundaryValuesCC *bvals);
  ~FOFCBoundary();
  int Exchange(const DvceArray4D<bool> &flags);

 private:
  struct Cell {
    std::size_t source, target;
    int peer;  // offset in send buffer for a same-rank neighbor; -1 for MPI
  };
  struct Transfer {
    int rank, send_tag, recv_tag, offset, count;
  };
  DvceArray1D<Cell> cells_;
  DvceArray1D<unsigned char> send_, recv_;
  HostArray1D<unsigned char> host_send_, host_recv_;
  std::vector<Transfer> transfers_;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<MPI_Request> requests_;
#endif
};

} // namespace mhd
#endif // MHD_FOFC_BOUNDARY_HPP_
