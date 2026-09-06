//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fofc_boundary.cpp
//! \brief Exchange one layer of FOFC flags, including edges and corners.

#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "mhd/fofc_boundary.hpp"

namespace mhd {

FOFCBoundary::FOFCBoundary(MeshBlockPack *pack, MeshBoundaryValuesCC *bvals) {
  const auto &ind = pack->pmesh->mb_indcs;
  const int nmb = pack->nmb_thispack;
  const int nn = pack->pmb->nnghbr;
  const int my_rank = global_variable::my_rank;
  const int gid0 = pack->pmb->mb_gid.h_view(0);
  const auto &neighbors = pack->pmb->nghbr.h_view;
  const int nc1 = ind.nx1 + 2*ind.ng;
  const int nc2 = pack->pmesh->multi_d ? ind.nx2 + 2*ind.ng : 1;
  const int nc3 = pack->pmesh->three_d ? ind.nx3 + 2*ind.ng : 1;
  const auto index = [=](int m, int k, int j, int i) -> std::size_t {
    return ((static_cast<std::size_t>(m)*nc3 + k)*nc2 + j)*nc1 + i;
  };

  std::vector<MeshBufferIndcs> ranges(nn);
  for (int n=0; n<nn; ++n) {
    auto r = bvals->recvbuf[n].isame[0];
    r.bis = std::max(r.bis, ind.is-1);
    r.bie = std::min(r.bie, ind.ie+1);
    r.bjs = std::max(r.bjs, pack->pmesh->multi_d ? ind.js-1 : ind.js);
    r.bje = std::min(r.bje, pack->pmesh->multi_d ? ind.je+1 : ind.je);
    r.bks = std::max(r.bks, pack->pmesh->three_d ? ind.ks-1 : ind.ks);
    r.bke = std::min(r.bke, pack->pmesh->three_d ? ind.ke+1 : ind.ke);
    ranges[n] = r;
  }

  std::vector<int> offsets(nmb*nn, -1);
  int total = 0;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nn; ++n) {
      if (neighbors(m,n).gid < 0) continue;
      const auto &r = ranges[n];
      offsets[m*nn+n] = total;
      total += (r.bie-r.bis+1)*(r.bje-r.bjs+1)*(r.bke-r.bks+1);
    }
  }
  cells_ = DvceArray1D<Cell>("fofc_boundary_cells", total);
  auto host_cells = Kokkos::create_mirror_view(cells_);
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nn; ++n) {
      const auto &nb = neighbors(m,n);
      if (nb.gid < 0) continue;
      const auto &r = ranges[n];
      const int offset = offsets[m*nn+n];
      const int si = r.bie < ind.is ? ind.is : (r.bis > ind.ie ? ind.ie : r.bis);
      const int sj = r.bje < ind.js ? ind.js : (r.bjs > ind.je ? ind.je : r.bjs);
      const int sk = r.bke < ind.ks ? ind.ks : (r.bks > ind.ke ? ind.ke : r.bks);
      const bool local = nb.rank == my_rank;
      const int peer = local ? offsets[(nb.gid-gid0)*nn+nb.dest] : -1;
      int q = 0;
      for (int k=r.bks; k<=r.bke; ++k) {
        for (int j=r.bjs; j<=r.bje; ++j) {
          for (int i=r.bis; i<=r.bie; ++i, ++q) {
            host_cells(offset+q) = {index(m,sk+k-r.bks,sj+j-r.bjs,si+i-r.bis),
                                    index(m,k,j,i), local ? peer+q : -1};
          }
        }
      }
      if (!local) {
        const int lid = nb.gid - pack->pmesh->gids_eachrank[nb.rank];
        transfers_.push_back({nb.rank, CreateBvals_MPI_Tag(lid, nb.dest),
                              CreateBvals_MPI_Tag(m, n), offset, q});
      }
    }
  }
  Kokkos::deep_copy(cells_, host_cells);
  send_ = DvceArray1D<unsigned char>("fofc_boundary_send", total);
  recv_ = DvceArray1D<unsigned char>("fofc_boundary_recv", total);
  if (!transfers_.empty()) {
    host_send_ = HostArray1D<unsigned char>("fofc_boundary_host_send", total);
    host_recv_ = HostArray1D<unsigned char>("fofc_boundary_host_recv", total);
  }
#if MPI_PARALLEL_ENABLED
  if (MPI_Comm_dup(MPI_COMM_WORLD, &comm_) != MPI_SUCCESS) {
    Kokkos::abort("Failed to create iterative FOFC communicator.");
  }
  requests_.resize(2*transfers_.size());
#endif
}

FOFCBoundary::~FOFCBoundary() {
#if MPI_PARALLEL_ENABLED
  if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
#endif
}

int FOFCBoundary::Exchange(const DvceArray4D<bool> &flags) {
  const auto cells = cells_;
  const auto send = send_;
  const auto recv = recv_;
  const int count = cells.extent_int(0);
  auto mask = flags.data();  // AthenaK views use contiguous LayoutRight storage
  Kokkos::parallel_for("fofc_pack_mask",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, count), KOKKOS_LAMBDA(const int q) {
    send(q) = mask[cells(q).source] ? 1 : 0;
  });

#if MPI_PARALLEL_ENABLED
  if (!transfers_.empty()) {
    Kokkos::deep_copy(host_send_, send_);
    int q = 0;
    for (const auto &t : transfers_) {
      if (MPI_Irecv(host_recv_.data()+t.offset, t.count, MPI_UNSIGNED_CHAR, t.rank,
                    t.recv_tag, comm_, &requests_[q++]) != MPI_SUCCESS) {
        Kokkos::abort("Failed to receive iterative FOFC mask.");
      }
    }
    for (const auto &t : transfers_) {
      if (MPI_Isend(host_send_.data()+t.offset, t.count, MPI_UNSIGNED_CHAR, t.rank,
                    t.send_tag, comm_, &requests_[q++]) != MPI_SUCCESS) {
        Kokkos::abort("Failed to send iterative FOFC mask.");
      }
    }
    if (MPI_Waitall(q, requests_.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
      Kokkos::abort("Failed to complete iterative FOFC mask exchange.");
    }
    Kokkos::deep_copy(recv_, host_recv_);
  }
#endif

  int changed = 0;
  Kokkos::parallel_reduce("fofc_unpack_mask",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, count),
      KOKKOS_LAMBDA(const int q, int &sum) {
    const auto c = cells(q);
    const bool value = (c.peer >= 0 ? send(c.peer) : recv(q)) != 0;
    if (mask[c.target] != value) ++sum;
    // The owning cell supplies the authoritative flag. Physical ghosts are untouched.
    mask[c.target] = value;
  }, changed);
  return changed;
}

} // namespace mhd
