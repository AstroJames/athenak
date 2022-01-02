//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file flux_correction_cc.cpp
//! \brief functions to pack/send and recv/unpack fluxes for cell-centered variables at
//! fine/coarse boundaries for the flux correction step.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValuesCC::PackAndSendFlux()
//! \brief Pack restricted fluxes of cell-centered variables at fine/coarse boundaries
//! into boundary buffers and send to neighbors for flux-correction step.
//!
//! This routine packs ALL the buffers on ALL the faces simultaneously for ALL the
//! MeshBlocks. Buffer data are then sent (via MPI) or copied directly for periodic or
//! block boundaries.

TaskStatus BoundaryValuesCC::PackAndSendFluxCC(DvceFaceFld5D<Real> &flx)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = flx.x1f.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR

  auto &cis = pmy_pack->pmesh->mb_indcs.cis;
  auto &cjs = pmy_pack->pmesh->mb_indcs.cjs;
  auto &cks = pmy_pack->pmesh->mb_indcs.cks;

  int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = send_buf;
  auto &rbuf = recv_buf;
  auto &one_d = pmy_pack->pmesh->one_d;
  auto &two_d = pmy_pack->pmesh->two_d;

  // Outer loop over (# of MeshBlocks)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only pack buffers for faces when neighbor is at coarser level
    if (nghbr.d_view(m,n).ccflx && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // Note send buffer flux indices are for the coarse mesh
      int il = sbuf[n].flux[0].bis;
      int iu = sbuf[n].flux[0].bie;
      int jl = sbuf[n].flux[0].bjs;
      int ju = sbuf[n].flux[0].bje;
      int kl = sbuf[n].flux[0].bks;
      int ku = sbuf[n].flux[0].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj  = nk*nj;

      // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
      // in this MeshBlockPack, so array index equals (target_id - first_id)
      int mm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int nn = nghbr.d_view(m,n).dest;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
        int fj = 2*j - cjs;
        int fk = 2*k - cks;

        // x1faces
        if (n<8) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            int fi = 2*i - cis;
            Real rflx;
            if (one_d) {
              rflx = flx.x1f(m,v,0,0,fi);
            } else if (two_d) {
              rflx = 0.5*(flx.x1f(m,v,0,fj,fi) + flx.x1f(m,v,0,fj+1,fi));
            } else {
              rflx = 0.25*(flx.x1f(m,v,fk  ,fj,fi) + flx.x1f(m,v,fk  ,fj+1,fi) +
                           flx.x1f(m,v,fk+1,fj,fi) + flx.x1f(m,v,fk+1,fj+1,fi));
            }
            // copy directly into recv buffer if MeshBlocks on same rank
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = rflx;
            // else copy into send buffer for MPI communication below
            } else {
              sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = rflx;
            }
/*****
std::cout << "x1send (m,n)="<<m<<","<<n<<"  (i,j,k)="<<fi<<","<<fj<<","<<fk << std::endl;
*****/
          });

        // x2faces
        } else if (n<16) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            int fi = 2*i - cis;
            Real rflx; 
            if (two_d) {
              rflx = 0.5*(flx.x2f(m,v,0,fj,fi) + flx.x2f(m,v,0,fj,fi+1));
            } else {
              rflx = 0.25*(flx.x2f(m,v,fk  ,fj,fi) + flx.x2f(m,v,fk  ,fj,fi+1) +
                           flx.x2f(m,v,fk+1,fj,fi) + flx.x2f(m,v,fk+1,fj,fi+1));
            }
            // copy directly into recv buffer if MeshBlocks on same rank
            if (nghbr.d_view(m,n).rank == my_rank) { 
              rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = rflx;
            // else copy into send buffer for MPI communication below
            } else {
              sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = rflx;
            }
/*********
std::cout << "x2send (m,n)="<<m<<","<<n<<"  (i,j,k)="<<fi<<","<<fj<<","<<fk << std::endl;
*****/
          });

        // x3faces
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            int fi = 2*i - cis;
            Real rflx = 0.25*(flx.x3f(m,v,fk,fj  ,fi) + flx.x3f(m,v,fk,fj  ,fi+1) +
                              flx.x3f(m,v,fk,fj+1,fi) + flx.x3f(m,v,fk,fj+1,fi+1));
            // copy directly into recv buffer if MeshBlocks on same rank
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = rflx;
            // else copy into send buffer for MPI communication below
            } else {
              sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = rflx;
            }
          });
        }
      });
    }  // end if-neighbor-exists block
  });  // end par_for_outer

  // Send boundary buffer to neighboring MeshBlocks using MPI

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;
  auto &mblev = pmy_pack->pmb->mb_lev;
#if MPI_PARALLEL_ENABLED
  auto &sbuf = send_buf;
#endif

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // compute indices of destination MeshBlock and Neighbor 
        int nn = nghbr.h_view(m,n).dest;
        // if MeshBlocks are on same rank, data already copied into receive buffer above
        // So simply set communication status tag as received.
        if (nghbr.h_view(m,n).rank == my_rank) {
          int mm = nghbr.h_view(m,n).gid - pmy_pack->gids;
          rbuf[nn].flx_stat[mm] = BoundaryCommStatus::received;

#if MPI_PARALLEL_ENABLED
        // Send boundary data using MPI
        } else {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid -
                    pmy_pack->pmesh->gidslist[nghbr.h_view(m,n).rank];
          int tag = CreateMPITag(lid, nn, key);
          auto send_data = Kokkos::subview(sbuf[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* send_ptr = send_data.data();
          int data_size = (sbuf[n].flux[0].ndat)*nvar;
          int ierr = MPI_Isend(send_ptr, data_size, MPI_ATHENA_REAL,
            nghbr.h_view(m,n).rank, tag, MPI_COMM_WORLD, &(sbuf[n].flx_req[m]));
#endif
        }
      }
    }
  }}

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus BoundaryValuesCC::RecvAndUnpackFluxCC(DvceFaceFld5D<Real> &flx)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  int nnghbr = pmy_pack->pmb->nnghbr;

  bool bflag = false;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;

#if MPI_PARALLEL_ENABLED
  // probe MPI communications.  This is a bit of black magic that seems to promote
  // communications to top of stack and gets them to complete more quickly
  int test;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, MPI_STATUS_IGNORE);
#endif

  //----- STEP 1: check that recv boundary buffer communications have all completed

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
        if (nghbr.h_view(m,n).rank == global_variable::my_rank) {
          if (rbuf[n].flx_stat[m] == BoundaryCommStatus::waiting) {bflag = true;}
#if MPI_PARALLEL_ENABLED
        } else {
          MPI_Test(&(rbuf[n].flx_req[m]), &test, MPI_STATUS_IGNORE);
          if (static_cast<bool>(test)) {
            rbuf[n].flx_stat[m] = BoundaryCommStatus::received;
          } else {
            bflag = true;
          }
#endif
        }
      }
    }
  }

  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  //----- STEP 2: buffers have all completed, so unpack

  int nvar = flx.x1f.extent_int(1); // TODO: 2nd index from L of input array must be NVAR
  auto &mblev = pmy_pack->pmb->mb_lev;

  // Outer loop over (# of MeshBlocks)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only unpack buffers for faces when neighbor is at finer level
    if (nghbr.d_view(m,n).ccflx && (nghbr.d_view(m,n).lev > mblev.d_view(m))) {
      // Recv buffer flux indices are for the regular mesh
      int il = rbuf[n].flux[0].bis;
      int iu = rbuf[n].flux[0].bie;
      int jl = rbuf[n].flux[0].bjs;
      int ju = rbuf[n].flux[0].bje;
      int kl = rbuf[n].flux[0].bks;
      int ku = rbuf[n].flux[0].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj  = nk*nj;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
         
        // x1faces
        if (n<8) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            flx.x1f(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
/*********
std::cout << "x1recv (m,n)="<<m<<","<<n<<"  (i,j,k)="<<i<<","<<j<<","<<k << std::endl;
********/
          });
        // x2faces
        } else if (n<16) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            flx.x2f(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
/*********
std::cout << "x2recv (m,n)="<<m<<","<<n<<"  (i,j,k)="<<i<<","<<j<<","<<k << std::endl;
********/
          });
        // x3faces
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            flx.x3f(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
          });
        }
      });
    }  // end if-neighbor-exists block
  });  // end par_for_outer

  return TaskStatus::complete;
}
