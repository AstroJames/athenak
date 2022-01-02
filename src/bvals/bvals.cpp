//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.cpp
//! \brief functions to pack/send and recv/unpack/prolongate boundary values for
//! cell-centered variables, implemented as part of the BValCC class.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// BoundaryValues constructor:

BoundaryValues::BoundaryValues(MeshBlockPack *pp, ParameterInput *pin)
 : pmy_pack(pp)
{
  // allocate vector of status flags and MPI requests (if needed)
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<nnghbr; ++n) {
    for (int m=0; m<nmb; ++m) {
      BoundaryCommStatus varsend_stat = BoundaryCommStatus::undef;
      BoundaryCommStatus varrecv_stat = BoundaryCommStatus::undef;
      send_buf[n].var_stat.push_back(varsend_stat);
      recv_buf[n].var_stat.push_back(varrecv_stat);

      BoundaryCommStatus flxsend_stat = BoundaryCommStatus::undef;
      BoundaryCommStatus flxrecv_stat = BoundaryCommStatus::undef;
      send_buf[n].flx_stat.push_back(flxsend_stat);
      recv_buf[n].flx_stat.push_back(flxrecv_stat);
#if MPI_PARALLEL_ENABLED
      // cannot create Kokkos::View of type MPI_Request (not POD) so use STL vector
      MPI_Request varsend_req, varrecv_req;
      send_buf[n].var_req.push_back(varsend_req);
      recv_buf[n].var_req.push_back(varrecv_req);

      MPI_Request flxsend_req, flxrecv_req;
      send_buf[n].flx_req.push_back(flxsend_req);
      recv_buf[n].flx_req.push_back(flxrecv_req);
#endif
    }
  }
} 

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::AllocateBuffers
//! \brief initialize components of send/recv BoundaryBuffers fixed-length arrays
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector
//! NOTE2: work here cannot be done in BoundaryValues constructor since it uses pure
//! virtual functions that get overridden in the derived classes

void BoundaryValues::InitializeBuffers(const int nvar)
{
  int nmb = pmy_pack->nmb_thispack;

  // initialize buffers used for uniform grid nd SMR/AMR calculations
  // set number of subblocks in x2- and x3-dirs
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  // x1 faces; NeighborIndex = [0,...,7]
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy = 0; fy<nfy; fy++) {
        int indx = pmy_pack->pmb->NeighborIndx(n,0,0,fy,fz);
        InitSendIndices(send_buf[indx],n, 0, 0, fy, fz);
        InitRecvIndices(recv_buf[indx],n, 0, 0, fy, fz);
        send_buf[indx].AllocateDataView(nmb, nvar);
        recv_buf[indx].AllocateDataView(nmb, nvar);
        indx++;
      }
    }
  }

  // add more buffers in 2D
  if (pmy_pack->pmesh->multi_d) {

    // x2 faces; NeighborIndex = [8,...,15]
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,m,0,fx,fz);
          InitSendIndices(send_buf[indx],0, m, 0, fx, fz);
          InitRecvIndices(recv_buf[indx],0, m, 0, fx, fz);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }

    // x1x2 edges; NeighborIndex = [16,...,23]
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int indx = pmy_pack->pmb->NeighborIndx(n,m,0,fz,0);
          InitSendIndices(send_buf[indx],n, m, 0, fz, 0);
          InitRecvIndices(recv_buf[indx],n, m, 0, fz, 0);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }
  }

  // add more buffers in 3D
  if (pmy_pack->pmesh->three_d) {

    // x3 faces; NeighborIndex = [24,...,31]
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) { 
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,0,l,fx,fy);
          InitSendIndices(send_buf[indx],0, 0, l, fx, fy);
          InitRecvIndices(recv_buf[indx],0, 0, l, fx, fy);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }

    // x3x1 edges; NeighborIndex = [32,...,39]
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int indx = pmy_pack->pmb->NeighborIndx(n,0,l,fy,0);
          InitSendIndices(send_buf[indx],n, 0, l, fy, 0);
          InitRecvIndices(recv_buf[indx],n, 0, l, fy, 0);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }

    // x2x3 edges; NeighborIndex = [40,...,47]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,m,l,fx,0);
          InitSendIndices(send_buf[indx],0, m, l, fx, 0);
          InitRecvIndices(recv_buf[indx],0, m, l, fx, 0);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }

    // corners; NeighborIndex = [48,...,55]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int indx = pmy_pack->pmb->NeighborIndx(n,m,l,0,0);
          InitSendIndices(send_buf[indx],n, m, l, 0, 0);
          InitRecvIndices(recv_buf[indx],n, m, l, 0, 0);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
        }
      }
    }
  }

/***************/
  int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<=nnghbr; ++n) {
std::cout << std::endl <<"CC_Buffer="<< n << std::endl;
std::cout <<"send_flux:" <<send_buf[n].flux[0].bis<<"  "<<send_buf[n].flux[0].bie<<
                "  "<<send_buf[n].flux[0].bjs<<"  "<<send_buf[n].flux[0].bje<<
                "  "<<send_buf[n].flux[0].bks<<"  "<<send_buf[n].flux[0].bke<< std::endl;
std::cout <<"recv_flux:" <<recv_buf[n].flux[0].bis<<"  "<<recv_buf[n].flux[0].bie<<
                "  "<<recv_buf[n].flux[0].bjs<<"  "<<recv_buf[n].flux[0].bje<<
                "  "<<recv_buf[n].flux[0].bks<<"  "<<recv_buf[n].flux[0].bke<< std::endl;
  }
/***************/



  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValues::InitRecv
//  \brief Posts non-blocking receives (with MPI), and initialize all boundary receive
//  status flags to waiting (with or without MPI) for boundary communication of CC vars.

TaskStatus BoundaryValues::InitRecv(int nvar)
{ 
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  
  // Initialize communications for cell-centered conserved variables
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) { 
      if (nghbr.h_view(m,n).gid >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n);
          auto recv_data = Kokkos::subview(recv_buf[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int data_size;
          // get data size if neighbor is at coarser/same/fine level
          if (nghbr.h_view(m,n).lev < mblev.h_view(m)) {
            data_size = (recv_buf[n].coar.ndat)*nvar;
          } else if (nghbr.h_view(m,n).lev == mblev.h_view(m)) {
            data_size = (recv_buf[n].same.ndat)*nvar;
          } else {
            data_size = (recv_buf[n].fine.ndat)*nvar;
          }
          (void) MPI_Irecv(recv_ptr, data_size, MPI_ATHENA_REAL, nghbr.h_view(m,n).rank,
                           tag, ccvar_comm, &(recv_buf[n].comm_req[m]));
        }
#endif  
        // initialize boundary receive status flags
        recv_buf[n].var_stat[m] = BoundaryCommStatus::waiting;
        recv_buf[n].flx_stat[m] = BoundaryCommStatus::waiting;
      }
    }
  }
  
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValues::ClearRecv
//  \brief Waits for all MPI receives associated with boundary communcations of CC vars
//  to complete before allowing execution to continue
  
TaskStatus BoundaryValues::ClearRecv()
{ 
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for CC vars to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          MPI_Wait(&(recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}       
          
//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValues::ClearSend
//  \brief Waits for all MPI sends associated with boundary communcations of CC vars to
//   complete before allowing execution to continue
  
TaskStatus BoundaryValues::ClearSend()
{ 
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for CC vars to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          MPI_Wait(&(send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}       
