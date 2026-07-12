//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_dual_ct.cpp
//! \brief Dynamic face-centered Ampere update for charge-conserving dual CT.

#include <iostream>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "driver/driver.hpp"
#include "eos/resistive_srmhd.hpp"
#include "globals.hpp"
#include "mhd/dual_ct.hpp"
#include "mhd/mhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::DualCTPrepare()
//! \brief Construct upwind edge B and face-centered qv before the Ampere update.

TaskStatus MHD::DualCTPrepare(Driver *pdriver, int stage) {
 if (!use_electric_ct)
  return TaskStatus::complete;

 auto &indcs  = pmy_pack->pmesh->mb_indcs;
 const int is = indcs.is, ie = indcs.ie;
 const int js = indcs.js, je = indcs.je;
 const int ks = indcs.ks, ke = indcs.ke;
 const int nmb1 = pmy_pack->nmb_thispack - 1;
 auto &size     = pmy_pack->pmb->mb_size;
 auto &nghbr    = pmy_pack->pmb->nghbr;
 auto e         = e0;
 auto j         = jfc;
 auto bedge     = bfld;
 auto w         = w0;
 auto flx1      = uflx.x1f;
 auto flx2      = uflx.x2f;
 auto flx3      = uflx.x3f;
 const int ix1m = NeighborIndex(-1, 0, 0, 0, 0);
 const int ix1p = NeighborIndex(1, 0, 0, 0, 0);
 const int ix2m = NeighborIndex(0, -1, 0, 0, 0);
 const int ix2p = NeighborIndex(0, 1, 0, 0, 0);
 const int ix3m = NeighborIndex(0, 0, -1, 0, 0);
 const int ix3p = NeighborIndex(0, 0, 1, 0, 0);

 if (pmy_pack->pmesh->three_d) {
  par_for(
    "dual_ct_3d_edge_b1", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je + 1,
    is, ie, KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     const Real by =
       flx2(m, srrmhd::IRE3, k - 1, j0, i) + flx2(m, srrmhd::IRE3, k, j0, i);
     const Real bz =
       -flx3(m, srrmhd::IRE2, k, j0 - 1, i) - flx3(m, srrmhd::IRE2, k, j0, i);
     bedge.x1e(m, k, j0, i) = 0.25 * (by + bz);
    });
  par_for(
    "dual_ct_3d_edge_b2", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je,
    is, ie + 1, KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     const Real bz =
       flx3(m, srrmhd::IRE1, k, j0, i - 1) + flx3(m, srrmhd::IRE1, k, j0, i);
     const Real bx =
       -flx1(m, srrmhd::IRE3, k - 1, j0, i) - flx1(m, srrmhd::IRE3, k, j0, i);
     bedge.x2e(m, k, j0, i) = 0.25 * (bz + bx);
    });
  par_for(
    "dual_ct_3d_edge_b3", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1,
    is, ie + 1, KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     const Real bx =
       flx1(m, srrmhd::IRE2, k, j0 - 1, i) + flx1(m, srrmhd::IRE2, k, j0, i);
     const Real by =
       -flx2(m, srrmhd::IRE1, k, j0, i - 1) - flx2(m, srrmhd::IRE1, k, j0, i);
     bedge.x3e(m, k, j0, i) = 0.25 * (bx + by);
    });

  par_for(
    "dual_ct_3d_j1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie + 1,
    KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     const bool use_left_ghost = (i == is) && (nghbr.d_view(m, ix1m).gid >= 0);
     const bool use_right_ghost =
       (i == ie + 1) && (nghbr.d_view(m, ix1p).gid >= 0);
     const int il = (i > is || use_left_ghost) ? i - 1 : is;
     const int ir = (i <= ie || use_right_ghost) ? i : ie;
     const Real idx1 = 1.0 / size.d_view(m).dx1;
     const Real idx2 = 1.0 / size.d_view(m).dx2;
     const Real idx3 = 1.0 / size.d_view(m).dx3;
     const Real ql =
       srrmhd::FaceDivergence(e, m, k, j0, il, idx1, idx2, idx3, true, true);
     const Real qr =
       srrmhd::FaceDivergence(e, m, k, j0, ir, idx1, idx2, idx3, true, true);
     const Real ux = 0.5 * (w(m, IVX, k, j0, il) + w(m, IVX, k, j0, ir));
     const Real uy = 0.5 * (w(m, IVY, k, j0, il) + w(m, IVY, k, j0, ir));
     const Real uz = 0.5 * (w(m, IVZ, k, j0, il) + w(m, IVZ, k, j0, ir));
     j.x1f(m, k, j0, i) =
       0.5 * (ql + qr) * ux / sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
    });
  par_for(
    "dual_ct_3d_j2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     const bool use_left_ghost =
       (j0 == js) && (nghbr.d_view(m, ix2m).gid >= 0);
     const bool use_right_ghost =
       (j0 == je + 1) && (nghbr.d_view(m, ix2p).gid >= 0);
     const int jl = (j0 > js || use_left_ghost) ? j0 - 1 : js;
     const int jr = (j0 <= je || use_right_ghost) ? j0 : je;
     const Real idx1 = 1.0 / size.d_view(m).dx1;
     const Real idx2 = 1.0 / size.d_view(m).dx2;
     const Real idx3 = 1.0 / size.d_view(m).dx3;
     const Real ql =
       srrmhd::FaceDivergence(e, m, k, jl, i, idx1, idx2, idx3, true, true);
     const Real qr =
       srrmhd::FaceDivergence(e, m, k, jr, i, idx1, idx2, idx3, true, true);
     const Real ux = 0.5 * (w(m, IVX, k, jl, i) + w(m, IVX, k, jr, i));
     const Real uy = 0.5 * (w(m, IVY, k, jl, i) + w(m, IVY, k, jr, i));
     const Real uz = 0.5 * (w(m, IVZ, k, jl, i) + w(m, IVZ, k, jr, i));
     j.x2f(m, k, j0, i) =
       0.5 * (ql + qr) * uy / sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
    });
  par_for(
    "dual_ct_3d_j3", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k0, int j0, int i) {
     const bool use_left_ghost =
       (k0 == ks) && (nghbr.d_view(m, ix3m).gid >= 0);
     const bool use_right_ghost =
       (k0 == ke + 1) && (nghbr.d_view(m, ix3p).gid >= 0);
     const int kl = (k0 > ks || use_left_ghost) ? k0 - 1 : ks;
     const int kr = (k0 <= ke || use_right_ghost) ? k0 : ke;
     const Real idx1 = 1.0 / size.d_view(m).dx1;
     const Real idx2 = 1.0 / size.d_view(m).dx2;
     const Real idx3 = 1.0 / size.d_view(m).dx3;
     const Real ql =
       srrmhd::FaceDivergence(e, m, kl, j0, i, idx1, idx2, idx3, true, true);
     const Real qr =
       srrmhd::FaceDivergence(e, m, kr, j0, i, idx1, idx2, idx3, true, true);
     const Real ux = 0.5 * (w(m, IVX, kl, j0, i) + w(m, IVX, kr, j0, i));
     const Real uy = 0.5 * (w(m, IVY, kl, j0, i) + w(m, IVY, kr, j0, i));
     const Real uz = 0.5 * (w(m, IVZ, kl, j0, i) + w(m, IVZ, kr, j0, i));
     j.x3f(m, k0, j0, i) =
       0.5 * (ql + qr) * uz / sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
    });
  return TaskStatus::complete;
 }

 if (pmy_pack->pmesh->two_d) {
  par_for(
    "dual_ct_2d_edge_b3", DevExeSpace(), 0, nmb1, js, je + 1, is, ie + 1,
    KOKKOS_LAMBDA(int m, int j, int i) {
     const Real bx =
       flx1(m, srrmhd::IRE2, ks, j - 1, i) + flx1(m, srrmhd::IRE2, ks, j, i);
     const Real by =
       -flx2(m, srrmhd::IRE1, ks, j, i - 1) - flx2(m, srrmhd::IRE1, ks, j, i);
     bedge.x3e(m, ks, j, i) = 0.25 * (bx + by);
    });
  par_for(
    "dual_ct_2d_edge_b12", DevExeSpace(), 0, nmb1, js, je + 1, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
     bedge.x1e(m, ks, j, i)     = flx2(m, srrmhd::IRE3, ks, j, i);
     bedge.x1e(m, ks + 1, j, i) = bedge.x1e(m, ks, j, i);
    });
  par_for(
    "dual_ct_2d_edge_b2", DevExeSpace(), 0, nmb1, js, je, is, ie + 1,
    KOKKOS_LAMBDA(int m, int j, int i) {
     bedge.x2e(m, ks, j, i)     = -flx1(m, srrmhd::IRE3, ks, j, i);
     bedge.x2e(m, ks + 1, j, i) = bedge.x2e(m, ks, j, i);
    });

  par_for(
    "dual_ct_2d_j1", DevExeSpace(), 0, nmb1, js, je, is, ie + 1,
    KOKKOS_LAMBDA(int m, int j0, int i) {
     const bool use_left_ghost = (i == is) && (nghbr.d_view(m, ix1m).gid >= 0);
     const bool use_right_ghost =
       (i == ie + 1) && (nghbr.d_view(m, ix1p).gid >= 0);
     const int il = (i > is || use_left_ghost) ? i - 1 : is;
     const int ir = (i <= ie || use_right_ghost) ? i : ie;
     const Real idx1 = 1.0 / size.d_view(m).dx1;
     const Real idx2 = 1.0 / size.d_view(m).dx2;
     const Real ql =
       srrmhd::FaceDivergence(e, m, ks, j0, il, idx1, idx2, 0.0, true, false);
     const Real qr =
       srrmhd::FaceDivergence(e, m, ks, j0, ir, idx1, idx2, 0.0, true, false);
     const Real ux = 0.5 * (w(m, IVX, ks, j0, il) + w(m, IVX, ks, j0, ir));
     const Real uy = 0.5 * (w(m, IVY, ks, j0, il) + w(m, IVY, ks, j0, ir));
     const Real uz = 0.5 * (w(m, IVZ, ks, j0, il) + w(m, IVZ, ks, j0, ir));
     j.x1f(m, ks, j0, i) =
       0.5 * (ql + qr) * ux / sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
    });
  par_for(
    "dual_ct_2d_j2", DevExeSpace(), 0, nmb1, js, je + 1, is, ie,
    KOKKOS_LAMBDA(int m, int j0, int i) {
     const bool use_left_ghost =
       (j0 == js) && (nghbr.d_view(m, ix2m).gid >= 0);
     const bool use_right_ghost =
       (j0 == je + 1) && (nghbr.d_view(m, ix2p).gid >= 0);
     const int jl = (j0 > js || use_left_ghost) ? j0 - 1 : js;
     const int jr = (j0 <= je || use_right_ghost) ? j0 : je;
     const Real idx1 = 1.0 / size.d_view(m).dx1;
     const Real idx2 = 1.0 / size.d_view(m).dx2;
     const Real ql =
       srrmhd::FaceDivergence(e, m, ks, jl, i, idx1, idx2, 0.0, true, false);
     const Real qr =
       srrmhd::FaceDivergence(e, m, ks, jr, i, idx1, idx2, 0.0, true, false);
     const Real ux = 0.5 * (w(m, IVX, ks, jl, i) + w(m, IVX, ks, jr, i));
     const Real uy = 0.5 * (w(m, IVY, ks, jl, i) + w(m, IVY, ks, jr, i));
     const Real uz = 0.5 * (w(m, IVZ, ks, jl, i) + w(m, IVZ, ks, jr, i));
     j.x2f(m, ks, j0, i) =
       0.5 * (ql + qr) * uy / sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
    });
  par_for(
    "dual_ct_2d_j3", DevExeSpace(), 0, nmb1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j0, int i) {
     const Real idx1 = 1.0 / size.d_view(m).dx1;
     const Real idx2 = 1.0 / size.d_view(m).dx2;
     const Real q =
       srrmhd::FaceDivergence(e, m, ks, j0, i, idx1, idx2, 0.0, true, false);
     const Real ux = w(m, IVX, ks, j0, i), uy = w(m, IVY, ks, j0, i),
                uz           = w(m, IVZ, ks, j0, i);
     const Real j3           = q * uz / sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
     j.x3f(m, ks, j0, i)     = j3;
     j.x3f(m, ks + 1, j0, i) = j3;
    });
  return TaskStatus::complete;
 }

 // In 1D, the Maxwell fluxes are already the required upwind line fields:
 // F_x(E_y)=B_z and F_x(E_z)=-B_y.
 par_for(
   "dual_ct_edge_b", DevExeSpace(), 0, nmb1, is, ie + 1,
   KOKKOS_LAMBDA(int m, int i) {
    bedge.x2e(m, ks, js, i) = -flx1(m, srrmhd::IRE3, ks, js, i);
    bedge.x3e(m, ks, js, i) = flx1(m, srrmhd::IRE2, ks, js, i);
   });

 // Normal current on x1 faces.  All quantities are collocated by symmetric
 // averaging, except the primary normal electric and magnetic components, which
 // already live here.
 par_for(
   "dual_ct_j1", DevExeSpace(), 0, nmb1, is, ie + 1,
   KOKKOS_LAMBDA(int m, int i) {
    const Real idx1 = 1.0 / size.d_view(m).dx1;
    const Real ql   = (e.x1f(m, ks, js, i) - e.x1f(m, ks, js, i - 1)) * idx1;
    const Real qr   = (e.x1f(m, ks, js, i + 1) - e.x1f(m, ks, js, i)) * idx1;
    const Real u1   = 0.5 * (w(m, IVX, ks, js, i - 1) + w(m, IVX, ks, js, i));
    const Real u2   = 0.5 * (w(m, IVY, ks, js, i - 1) + w(m, IVY, ks, js, i));
    const Real u3   = 0.5 * (w(m, IVZ, ks, js, i - 1) + w(m, IVZ, ks, js, i));
    const Real lor  = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
    j.x1f(m, ks, js, i) = 0.5 * (ql + qr) * u1 / lor;
   });

 // Transverse currents are naturally collocated with the degenerate x2/x3
 // faces.
 par_for(
   "dual_ct_j23", DevExeSpace(), 0, nmb1, is, ie, KOKKOS_LAMBDA(int m, int i) {
    const Real idx1 = 1.0 / size.d_view(m).dx1;
    const Real q    = (e.x1f(m, ks, js, i + 1) - e.x1f(m, ks, js, i)) * idx1;
    const Real u1   = w(m, IVX, ks, js, i);
    const Real u2   = w(m, IVY, ks, js, i);
    const Real u3   = w(m, IVZ, ks, js, i);
    const Real lor  = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
    j.x2f(m, ks, js, i)     = q * u2 / lor;
    j.x2f(m, ks, js + 1, i) = q * u2 / lor;
    j.x3f(m, ks, js, i)     = q * u3 / lor;
    j.x3f(m, ks + 1, js, i) = q * u3 / lor;
   });

 return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::DualCTUpdate()
//! \brief Advance face E with the synchronized edge B and face qv, then update
//! the cell-centered electric mirror.

TaskStatus MHD::DualCTUpdate(Driver *pdriver, int stage) {
 if (!use_electric_ct)
  return TaskStatus::complete;

 const Real dt = pmy_pack->pmesh->dt;
 auto &indcs   = pmy_pack->pmesh->mb_indcs;
 const int is = indcs.is, ie = indcs.ie;
 const int js = indcs.js, je = indcs.je;
 const int ks       = indcs.ks;
 const int nmb1     = pmy_pack->nmb_thispack - 1;
 const Real gam0    = pdriver->gam0[stage - 1];
 const Real gam1    = pdriver->gam1[stage - 1];
 const Real beta_dt = pdriver->beta[stage - 1] * dt;
 auto &size         = pmy_pack->pmb->mb_size;
 auto e             = e0;
 auto eold          = e1;
 auto j             = jfc;
 auto bedge         = bfld;
 auto u             = u0;

 if (pmy_pack->pmesh->three_d) {
  const int ke = indcs.ke;
  par_for(
    "dual_ct_3d_update_e1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie + 1,
    KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     const Real curl = srrmhd::EdgeCurl1(
       bedge, m, k, j0, i, 1.0 / size.d_view(m).dx2,
       1.0 / size.d_view(m).dx3, true, true);
     e.x1f(m, k, j0, i) = gam0 * e.x1f(m, k, j0, i)
       + gam1 * eold.x1f(m, k, j0, i) + beta_dt * (curl - j.x1f(m, k, j0, i));
    });
  par_for(
    "dual_ct_3d_update_e2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     const Real curl = srrmhd::EdgeCurl2(
       bedge, m, k, j0, i, 1.0 / size.d_view(m).dx1,
       1.0 / size.d_view(m).dx3, true);
     e.x2f(m, k, j0, i) = gam0 * e.x2f(m, k, j0, i)
       + gam1 * eold.x2f(m, k, j0, i) + beta_dt * (curl - j.x2f(m, k, j0, i));
    });
  par_for(
    "dual_ct_3d_update_e3", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     const Real curl = srrmhd::EdgeCurl3(
       bedge, m, k, j0, i, 1.0 / size.d_view(m).dx1,
       1.0 / size.d_view(m).dx2, true);
     e.x3f(m, k, j0, i) = gam0 * e.x3f(m, k, j0, i)
       + gam1 * eold.x3f(m, k, j0, i) + beta_dt * (curl - j.x3f(m, k, j0, i));
    });
  par_for(
    "dual_ct_3d_sync", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j0, int i) {
     Real a, bv, c;
     srrmhd::ElectricFaceToCell(e, m, k, j0, i, a, bv, c);
     u(m, srrmhd::IRE1, k, j0, i) = a;
     u(m, srrmhd::IRE2, k, j0, i) = bv;
     u(m, srrmhd::IRE3, k, j0, i) = c;
    });
  return TaskStatus::complete;
 }

 if (pmy_pack->pmesh->two_d) {
  par_for(
    "dual_ct_2d_update_e1", DevExeSpace(), 0, nmb1, js, je, is, ie + 1,
    KOKKOS_LAMBDA(int m, int j0, int i) {
     const Real curl = srrmhd::EdgeCurl1(
       bedge, m, ks, j0, i, 1.0 / size.d_view(m).dx2, 0.0, true, false);
     e.x1f(m, ks, j0, i) = gam0 * e.x1f(m, ks, j0, i)
       + gam1 * eold.x1f(m, ks, j0, i) + beta_dt * (curl - j.x1f(m, ks, j0, i));
    });
  par_for(
    "dual_ct_2d_update_e2", DevExeSpace(), 0, nmb1, js, je + 1, is, ie,
    KOKKOS_LAMBDA(int m, int j0, int i) {
     const Real curl = srrmhd::EdgeCurl2(
       bedge, m, ks, j0, i, 1.0 / size.d_view(m).dx1, 0.0, false);
     e.x2f(m, ks, j0, i) = gam0 * e.x2f(m, ks, j0, i)
       + gam1 * eold.x2f(m, ks, j0, i) + beta_dt * (curl - j.x2f(m, ks, j0, i));
    });
  par_for(
    "dual_ct_2d_update_e3", DevExeSpace(), 0, nmb1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j0, int i) {
     const Real curl = srrmhd::EdgeCurl3(
       bedge, m, ks, j0, i, 1.0 / size.d_view(m).dx1, 1.0 / size.d_view(m).dx2,
       true);
     const Real v = gam0 * e.x3f(m, ks, j0, i) + gam1 * eold.x3f(m, ks, j0, i)
       + beta_dt * (curl - j.x3f(m, ks, j0, i));
     e.x3f(m, ks, j0, i)     = v;
     e.x3f(m, ks + 1, j0, i) = v;
    });
  par_for(
    "dual_ct_2d_sync", DevExeSpace(), 0, nmb1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j0, int i) {
     Real a, bv, c;
     srrmhd::ElectricFaceToCell(e, m, ks, j0, i, a, bv, c);
     u(m, srrmhd::IRE1, ks, j0, i) = a;
     u(m, srrmhd::IRE2, ks, j0, i) = bv;
     u(m, srrmhd::IRE3, ks, j0, i) = c;
    });
  return TaskStatus::complete;
 }

 par_for(
   "dual_ct_update_e1", DevExeSpace(), 0, nmb1, is, ie + 1,
   KOKKOS_LAMBDA(int m, int i) {
    e.x1f(m, ks, js, i) = gam0 * e.x1f(m, ks, js, i)
      + gam1 * eold.x1f(m, ks, js, i) - beta_dt * j.x1f(m, ks, js, i);
   });
 par_for(
   "dual_ct_update_e23", DevExeSpace(), 0, nmb1, is, ie,
   KOKKOS_LAMBDA(int m, int i) {
    const Real idx1  = 1.0 / size.d_view(m).dx1;
    const Real curl2 = srrmhd::EdgeCurl2(bedge, m, ks, js, i, idx1, 0.0, false);
    const Real curl3 = srrmhd::EdgeCurl3(bedge, m, ks, js, i, idx1, 0.0, false);
    const Real newe2 = gam0 * e.x2f(m, ks, js, i)
      + gam1 * eold.x2f(m, ks, js, i) + beta_dt * (curl2 - j.x2f(m, ks, js, i));
    const Real newe3 = gam0 * e.x3f(m, ks, js, i)
      + gam1 * eold.x3f(m, ks, js, i) + beta_dt * (curl3 - j.x3f(m, ks, js, i));
    e.x2f(m, ks, js, i)     = newe2;
    e.x2f(m, ks, js + 1, i) = newe2;
    e.x3f(m, ks, js, i)     = newe3;
    e.x3f(m, ks + 1, js, i) = newe3;
   });

 // The packed cell-centered E remains a derived mirror for fluid fluxes and
 // recovery.
 par_for(
   "dual_ct_sync_cell_e", DevExeSpace(), 0, nmb1, is, ie,
   KOKKOS_LAMBDA(int m, int i) {
    Real ec1, ec2, ec3;
    srrmhd::ElectricFaceToCell(e, m, ks, js, i, ec1, ec2, ec3);
    u(m, srrmhd::IRE1, ks, js, i) = ec1;
    u(m, srrmhd::IRE2, ks, js, i) = ec2;
    u(m, srrmhd::IRE3, ks, js, i) = ec3;
   });

 return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::FreezeFaceResistivity()
//! \brief Evaluate and freeze the nonuniform coefficient on every primary E face.
//!
//! The stage-star electric field and the most recently accepted fluid primitives are
//! collocated with the same symmetric stencils used by FaceImpRKUpdate.  This routine
//! is called exactly once per diagonal IMEX stage, outside the Picard iteration.

void MHD::FreezeFaceResistivity() {
 if (resistivity_data.model == srrmhd::ResistivityModel::uniform) return;

 auto &indcs      = pmy_pack->pmesh->mb_indcs;
 const int is = indcs.is, ie = indcs.ie;
 const int js = indcs.js, je = indcs.je;
 const int ks = indcs.ks, ke = indcs.ke;
 const int nmb1 = pmy_pack->nmb_thispack - 1;
 const bool three_d = pmy_pack->pmesh->three_d;
 const int e3ke = three_d ? ke + 1 : ks;
 auto eta       = eta_face;
 auto es        = estar;
 auto b         = b0;
 auto bcc       = bcc0;
 auto w         = w0;
 auto &nghbr    = pmy_pack->pmb->nghbr;
 const auto data = resistivity_data;
 const int ix1m = NeighborIndex(-1, 0, 0, 0, 0);
 const int ix1p = NeighborIndex(1, 0, 0, 0, 0);
 const int ix2m = NeighborIndex(0, -1, 0, 0, 0);
 const int ix2p = NeighborIndex(0, 1, 0, 0, 0);
 const int ix3m = NeighborIndex(0, 0, -1, 0, 0);
 const int ix3p = NeighborIndex(0, 0, 1, 0, 0);

 if (pmy_pack->pmesh->multi_d) {
  par_for(
    "dual_ct_freeze_eta1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie + 1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
     const bool use_left_ghost =
       (i == is) && (nghbr.d_view(m, ix1m).gid >= 0);
     const bool use_right_ghost =
       (i == ie + 1) && (nghbr.d_view(m, ix1p).gid >= 0);
     const int il = (i > is || use_left_ghost) ? i - 1 : is;
     const int ir = (i <= ie || use_right_ghost) ? i : ie;
     const Real rho = 0.5*(w(m, IDN, k, j, il) + w(m, IDN, k, j, ir));
     const Real u1 = 0.5*(w(m, IVX, k, j, il) + w(m, IVX, k, j, ir));
     const Real u2 = 0.5*(w(m, IVY, k, j, il) + w(m, IVY, k, j, ir));
     const Real u3 = 0.5*(w(m, IVZ, k, j, il) + w(m, IVZ, k, j, ir));
     const Real e2 = 0.25
       * (es.x2f(m, k, j, il) + es.x2f(m, k, j + 1, il)
          + es.x2f(m, k, j, ir) + es.x2f(m, k, j + 1, ir));
     const Real e3 = 0.25
       * (es.x3f(m, k, j, il) + es.x3f(m, k + 1, j, il)
          + es.x3f(m, k, j, ir) + es.x3f(m, k + 1, j, ir));
     const Real b2 = 0.5*(bcc(m, IBY, k, j, il) + bcc(m, IBY, k, j, ir));
     const Real b3 = 0.5*(bcc(m, IBZ, k, j, il) + bcc(m, IBZ, k, j, ir));
     eta.x1f(m, k, j, i) = srrmhd::EvaluateResistivity(
       data, rho, u1, u2, u3, es.x1f(m, k, j, i), e2, e3,
       b.x1f(m, k, j, i), b2, b3);
    });
  par_for(
    "dual_ct_freeze_eta2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
     const bool use_left_ghost =
       (j == js) && (nghbr.d_view(m, ix2m).gid >= 0);
     const bool use_right_ghost =
       (j == je + 1) && (nghbr.d_view(m, ix2p).gid >= 0);
     const int jl = (j > js || use_left_ghost) ? j - 1 : js;
     const int jr = (j <= je || use_right_ghost) ? j : je;
     const Real rho = 0.5*(w(m, IDN, k, jl, i) + w(m, IDN, k, jr, i));
     const Real u1 = 0.5*(w(m, IVX, k, jl, i) + w(m, IVX, k, jr, i));
     const Real u2 = 0.5*(w(m, IVY, k, jl, i) + w(m, IVY, k, jr, i));
     const Real u3 = 0.5*(w(m, IVZ, k, jl, i) + w(m, IVZ, k, jr, i));
     const Real e1 = 0.25
       * (es.x1f(m, k, jl, i) + es.x1f(m, k, jl, i + 1)
          + es.x1f(m, k, jr, i) + es.x1f(m, k, jr, i + 1));
     const Real e3 = 0.25
       * (es.x3f(m, k, jl, i) + es.x3f(m, k + 1, jl, i)
          + es.x3f(m, k, jr, i) + es.x3f(m, k + 1, jr, i));
     const Real b1 = 0.5*(bcc(m, IBX, k, jl, i) + bcc(m, IBX, k, jr, i));
     const Real b3 = 0.5*(bcc(m, IBZ, k, jl, i) + bcc(m, IBZ, k, jr, i));
     eta.x2f(m, k, j, i) = srrmhd::EvaluateResistivity(
       data, rho, u1, u2, u3, e1, es.x2f(m, k, j, i), e3,
       b1, b.x2f(m, k, j, i), b3);
    });
  par_for(
    "dual_ct_freeze_eta3", DevExeSpace(), 0, nmb1, ks, e3ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
     const bool use_left_ghost =
       three_d && (k == ks) && (nghbr.d_view(m, ix3m).gid >= 0);
     const bool use_right_ghost =
       three_d && (k == ke + 1) && (nghbr.d_view(m, ix3p).gid >= 0);
     const int kl = three_d ? ((k > ks || use_left_ghost) ? k - 1 : ks) : ks;
     const int kr = three_d ? ((k <= ke || use_right_ghost) ? k : ke) : ks;
     const Real rho = 0.5*(w(m, IDN, kl, j, i) + w(m, IDN, kr, j, i));
     const Real u1 = 0.5*(w(m, IVX, kl, j, i) + w(m, IVX, kr, j, i));
     const Real u2 = 0.5*(w(m, IVY, kl, j, i) + w(m, IVY, kr, j, i));
     const Real u3 = 0.5*(w(m, IVZ, kl, j, i) + w(m, IVZ, kr, j, i));
     const Real e1 = 0.25
       * (es.x1f(m, kl, j, i) + es.x1f(m, kl, j, i + 1)
          + es.x1f(m, kr, j, i) + es.x1f(m, kr, j, i + 1));
     const Real e2 = 0.25
       * (es.x2f(m, kl, j, i) + es.x2f(m, kl, j + 1, i)
          + es.x2f(m, kr, j, i) + es.x2f(m, kr, j + 1, i));
     const Real b1 = 0.5*(bcc(m, IBX, kl, j, i) + bcc(m, IBX, kr, j, i));
     const Real b2 = 0.5*(bcc(m, IBY, kl, j, i) + bcc(m, IBY, kr, j, i));
     const Real value = srrmhd::EvaluateResistivity(
       data, rho, u1, u2, u3, e1, e2, es.x3f(m, k, j, i),
       b1, b2, b.x3f(m, k, j, i));
     eta.x3f(m, k, j, i) = value;
     if (!three_d) eta.x3f(m, ks + 1, j, i) = value;
    });
  return;
 }

 par_for(
   "dual_ct_freeze_eta1_1d", DevExeSpace(), 0, nmb1, is, ie + 1,
   KOKKOS_LAMBDA(int m, int i) {
    const Real rho = 0.5*(w(m, IDN, ks, js, i - 1) + w(m, IDN, ks, js, i));
    const Real u1 = 0.5*(w(m, IVX, ks, js, i - 1) + w(m, IVX, ks, js, i));
    const Real u2 = 0.5*(w(m, IVY, ks, js, i - 1) + w(m, IVY, ks, js, i));
    const Real u3 = 0.5*(w(m, IVZ, ks, js, i - 1) + w(m, IVZ, ks, js, i));
    const Real e2 = 0.5*(es.x2f(m, ks, js, i - 1) + es.x2f(m, ks, js, i));
    const Real e3 = 0.5*(es.x3f(m, ks, js, i - 1) + es.x3f(m, ks, js, i));
    const Real b2 = 0.5*(bcc(m, IBY, ks, js, i - 1) + bcc(m, IBY, ks, js, i));
    const Real b3 = 0.5*(bcc(m, IBZ, ks, js, i - 1) + bcc(m, IBZ, ks, js, i));
    eta.x1f(m, ks, js, i) = srrmhd::EvaluateResistivity(
      data, rho, u1, u2, u3, es.x1f(m, ks, js, i), e2, e3,
      b.x1f(m, ks, js, i), b2, b3);
   });
 par_for(
   "dual_ct_freeze_eta23_1d", DevExeSpace(), 0, nmb1, is, ie,
   KOKKOS_LAMBDA(int m, int i) {
    const Real e1 = 0.5*(es.x1f(m, ks, js, i) + es.x1f(m, ks, js, i + 1));
    const Real eta2 = srrmhd::EvaluateResistivity(
      data, w(m, IDN, ks, js, i), w(m, IVX, ks, js, i),
      w(m, IVY, ks, js, i), w(m, IVZ, ks, js, i), e1,
      es.x2f(m, ks, js, i), es.x3f(m, ks, js, i), bcc(m, IBX, ks, js, i),
      bcc(m, IBY, ks, js, i), bcc(m, IBZ, ks, js, i));
    eta.x2f(m, ks, js, i) = eta2;
    eta.x2f(m, ks, js + 1, i) = eta2;
    eta.x3f(m, ks, js, i) = eta2;
    eta.x3f(m, ks + 1, js, i) = eta2;
   });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::FaceImpRKUpdate()
//! \brief Apply one conductive IMEX stage to primary face-centered electric
//! fields.
//!
//! For fixed face velocity, ImplicitElectricField() gives the exact
//! backward-Euler conductive update.  Face E is then averaged to cells and the
//! fluid is recovered at fixed total momentum and energy.  Picard iteration
//! makes the interpolated face velocity and recovered cell velocity mutually
//! consistent.  Every stored source is computed from the discrete face update,
//! preserving the matching charge continuity equation under the IMEX tableau.

TaskStatus MHD::FaceImpRKUpdate(Driver *pdriver, int estage) {
 const int istage = estage + 2;
 auto &indcs      = pmy_pack->pmesh->mb_indcs;
 const int is = indcs.is, ie = indcs.ie;
 const int js = indcs.js, je = indcs.je;
 const int ks = indcs.ks, ke = indcs.ke;
 const int n1 = indcs.nx1 + 2*indcs.ng;
 const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
 const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
 const int nmb1 = pmy_pack->nmb_thispack - 1;
 const Real dt  = pmy_pack->pmesh->dt;
 auto e         = e0;
 auto es        = estar;
 auto re        = ect_src;
 auto delta     = jfc;
 auto b         = b0;
 auto bcc       = bcc0;
 auto u         = u0;
 auto w         = w0;
 auto uprev     = ect_u_prev;
 auto eta_f     = eta_face;
 auto &a_twid   = pdriver->a_twid;
 auto &nghbr    = pmy_pack->pmb->nghbr;
 const bool multi_block = pmy_pack->pmesh->nmb_total > 1;
 const int ix1m = NeighborIndex(-1, 0, 0, 0, 0);
 const int ix1p = NeighborIndex(1, 0, 0, 0, 0);
 const int ix2m = NeighborIndex(0, -1, 0, 0, 0);
 const int ix2p = NeighborIndex(0, 1, 0, 0, 0);
 const int ix3m = NeighborIndex(0, 0, -1, 0, 0);
 const int ix3p = NeighborIndex(0, 0, 1, 0, 0);
 const bool nonuniform_eta =
   resistivity_data.model != srrmhd::ResistivityModel::uniform;
 const Real uniform_eta = resistivity;

 auto sync_active_e = [&]() {
  par_for(
    "dual_ct_nonuniform_lagged_sync", DevExeSpace(), 0, nmb1, ks, ke,
    js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
     Real ec1, ec2, ec3;
     srrmhd::ElectricFaceToCell(e, m, k, j, i, ec1, ec2, ec3);
     u(m, srrmhd::IRE1, k, j, i) = ec1;
     u(m, srrmhd::IRE2, k, j, i) = ec2;
     u(m, srrmhd::IRE3, k, j, i) = ec3;
    });
 };

 auto recover_lagged_state = [&]() {
  peos->ConsToPrim(u, b, w, bcc, false, 0, n1 - 1, 0, n2 - 1, 0, n3 - 1);
 };

 if (pmy_pack->pmesh->multi_d) {
  constexpr int max_iterations = 60;
  constexpr Real relaxation    = 0.5;
  const Real tolerance = (sizeof(Real) == sizeof(float)) ? 2.0e-6 : 2.0e-12;
  const Real adt       = pdriver->a_impl * dt;
  const bool diagonal_solve = estage < pdriver->nexp_stages;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int e3ke = three_d ? ke + 1 : ks;

  auto assemble_star_faces = [&]() {
   par_for(
     "dual_ct_md_star_e1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie + 1,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real v = e.x1f(m, k, j, i);
      if (istage > 1) {
       for (int s = 0; s <= istage - 2; ++s) {
        v += a_twid[istage - 2][s] * dt * re.x1f(m, s, k, j, i);
       }
      }
      es.x1f(m, k, j, i) = v;
      e.x1f(m, k, j, i)  = v;
     });
   par_for(
     "dual_ct_md_star_e2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real v = e.x2f(m, k, j, i);
      if (istage > 1) {
       for (int s = 0; s <= istage - 2; ++s) {
        v += a_twid[istage - 2][s] * dt * re.x2f(m, s, k, j, i);
       }
      }
      es.x2f(m, k, j, i) = v;
      e.x2f(m, k, j, i)  = v;
     });
   par_for(
     "dual_ct_md_star_e3", DevExeSpace(), 0, nmb1, ks, e3ke, js, je, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real v = e.x3f(m, k, j, i);
      if (istage > 1) {
       for (int s = 0; s <= istage - 2; ++s) {
        v += a_twid[istage - 2][s] * dt * re.x3f(m, s, k, j, i);
       }
      }
      es.x3f(m, k, j, i) = v;
      e.x3f(m, k, j, i)  = v;
      if (!three_d) {
       es.x3f(m, ks + 1, j, i) = v;
       e.x3f(m, ks + 1, j, i)  = v;
      }
     });
  };

  if (ect_impl_state != ECTImplicitState::idle && ect_impl_estage != estage) {
   Kokkos::abort("Dual-CT implicit task re-entered with a different stage");
   return TaskStatus::fail;
  }

  auto sync_interior = [&]() {
   par_for(
     "dual_ct_md_imp_sync", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real ec1, ec2, ec3;
      srrmhd::ElectricFaceToCell(e, m, k, j, i, ec1, ec2, ec3);
      u(m, srrmhd::IRE1, k, j, i) = ec1;
      u(m, srrmhd::IRE2, k, j, i) = ec2;
      u(m, srrmhd::IRE3, k, j, i) = ec3;
     });
   peos->ConsToPrim(u, b, w, bcc, false, is, ie, js, je, ks, ke);
  };

  auto measure_residual = [&]() {
   Real max_change = 0.0;
   Kokkos::parallel_reduce(
     "dual_ct_md_velocity_change",
     Kokkos::MDRangePolicy<Kokkos::Rank<4> >(
       DevExeSpace(), {0, ks, js, is}, {nmb1 + 1, ke + 1, je + 1, ie + 1}),
     KOKKOS_LAMBDA(int m, int k, int j, int i, Real &max_value) {
      Real change = fabs(w(m, IVX, k, j, i) - uprev(m, 0, k, j, i));
      change = fmax(change, fabs(w(m, IVY, k, j, i) - uprev(m, 1, k, j, i)));
      change = fmax(change, fabs(w(m, IVZ, k, j, i) - uprev(m, 2, k, j, i)));
      max_value = fmax(max_value, change);
     },
     Kokkos::Max<Real>(max_change));
   Real max_e1_residual = 0.0;
   Real max_e2_residual = 0.0;
   Real max_e3_residual = 0.0;
   Kokkos::parallel_reduce(
     "dual_ct_md_e1_residual",
     Kokkos::MDRangePolicy<Kokkos::Rank<4> >(
       DevExeSpace(), {0, ks, js, is}, {nmb1 + 1, ke + 1, je + 1, ie + 2}),
     KOKKOS_LAMBDA(int m, int k, int j, int i, Real &max_value) {
      max_value = fmax(max_value, delta.x1f(m, k, j, i));
     },
     Kokkos::Max<Real>(max_e1_residual));
   Kokkos::parallel_reduce(
     "dual_ct_md_e2_residual",
     Kokkos::MDRangePolicy<Kokkos::Rank<4> >(
       DevExeSpace(), {0, ks, js, is}, {nmb1 + 1, ke + 1, je + 2, ie + 1}),
     KOKKOS_LAMBDA(int m, int k, int j, int i, Real &max_value) {
      max_value = fmax(max_value, delta.x2f(m, k, j, i));
     },
     Kokkos::Max<Real>(max_e2_residual));
   Kokkos::parallel_reduce(
     "dual_ct_md_e3_residual",
     Kokkos::MDRangePolicy<Kokkos::Rank<4> >(
       DevExeSpace(), {0, ks, js, is}, {nmb1 + 1, e3ke + 1, je + 1, ie + 1}),
     KOKKOS_LAMBDA(int m, int k, int j, int i, Real &max_value) {
      max_value = fmax(max_value, delta.x3f(m, k, j, i));
     },
     Kokkos::Max<Real>(max_e3_residual));
   return fmax(max_change,
               fmax(max_e1_residual, fmax(max_e2_residual, max_e3_residual)));
  };

  auto store_sources = [&]() {
   const int ss = istage - 1;
   par_for(
     "dual_ct_md_src_e1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie + 1,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      re.x1f(m, ss, k, j, i) =
        (e.x1f(m, k, j, i) - es.x1f(m, k, j, i)) / adt;
     });
   par_for(
     "dual_ct_md_src_e2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      re.x2f(m, ss, k, j, i) =
        (e.x2f(m, k, j, i) - es.x2f(m, k, j, i)) / adt;
     });
   par_for(
     "dual_ct_md_src_e3", DevExeSpace(), 0, nmb1, ks, e3ke, js, je, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real r = (e.x3f(m, k, j, i) - es.x3f(m, k, j, i)) / adt;
      re.x3f(m, ss, k, j, i) = r;
      if (!three_d) re.x3f(m, ss, ks + 1, j, i) = r;
     });
  };

  auto finish_stage = [&]() {
   par_for(
     "dual_ct_md_final_sync", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real ec1, ec2, ec3;
      srrmhd::ElectricFaceToCell(e, m, k, j, i, ec1, ec2, ec3);
      u(m, srrmhd::IRE1, k, j, i) = ec1;
      u(m, srrmhd::IRE2, k, j, i) = ec2;
      u(m, srrmhd::IRE3, k, j, i) = ec3;
     });
   ect_impl_state = ECTImplicitState::idle;
   ect_impl_estage = -1;
   ect_picard_iteration = 0;
   ect_comm_phase = 0;
   return TaskStatus::complete;
  };

  auto fail_iteration = [&]() {
   if (global_variable::my_rank == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Multidimensional face-centered electric IMEX iteration "
              << "failed to converge" << std::endl;
   }
   Kokkos::abort("Multidimensional face-centered electric IMEX iteration failed");
   return TaskStatus::fail;
  };

  auto check_iteration = [&]() {
   if (ect_global_residual < tolerance) {
    store_sources();
    if (multi_block) {
     TaskStatus tstat = StartElectricFaceExchange(e);
     if (tstat != TaskStatus::complete) return tstat;
     ect_impl_state = ECTImplicitState::final_face_recv;
     return TaskStatus::incomplete;
    }
    return finish_stage();
   }
   ++ect_picard_iteration;
   if (ect_picard_iteration >= max_iterations) return fail_iteration();
   ect_impl_state = ECTImplicitState::picard_compute;
   return TaskStatus::incomplete;
  };

  if (ect_impl_state == ECTImplicitState::idle) {
   ect_impl_estage = estage;
   ect_picard_iteration = 0;
   if (nonuniform_eta && diagonal_solve) {
    // Eta must sample an accepted, decomposition-independent cell state.  Refresh
    // active cell E from the primary face field, then communicate U before recovery.
    sync_active_e();
    if (multi_block) {
     TaskStatus tstat = StartElectricCellExchange();
     if (tstat != TaskStatus::complete) return tstat;
     ect_impl_state = ECTImplicitState::lagged_cell_recv;
     return TaskStatus::incomplete;
    }
    recover_lagged_state();
   }
   assemble_star_faces();
   if (multi_block) {
    TaskStatus tstat = StartElectricFaceExchange(es);
    if (tstat != TaskStatus::complete) return tstat;
    ect_impl_state = ECTImplicitState::star_face_recv;
    return TaskStatus::incomplete;
   }
   ect_impl_state = diagonal_solve ? ECTImplicitState::picard_compute
                                   : ECTImplicitState::final_face_recv;
   if (!diagonal_solve) return finish_stage();
   FreezeFaceResistivity();
  }

  if (ect_impl_state == ECTImplicitState::lagged_cell_recv) {
   TaskStatus tstat = FinishElectricCellExchange();
   if (tstat != TaskStatus::complete) return tstat;
   assemble_star_faces();
   tstat = StartElectricFaceExchange(es);
   if (tstat != TaskStatus::complete) return tstat;
   ect_impl_state = ECTImplicitState::star_face_recv;
   return TaskStatus::incomplete;
  }

  if (ect_impl_state == ECTImplicitState::star_face_recv) {
   TaskStatus tstat = FinishElectricFaceExchange(es);
   if (tstat != TaskStatus::complete) return tstat;
   if (diagonal_solve) {
    FreezeFaceResistivity();
    ect_impl_state = ECTImplicitState::picard_compute;
    return TaskStatus::incomplete;
   }
   tstat = StartSharedElectricAverage(e);
   if (tstat != TaskStatus::complete) return tstat;
   ect_impl_state = ECTImplicitState::final_average_recv;
   return TaskStatus::incomplete;
  }

  if (ect_impl_state == ECTImplicitState::picard_compute) {
   par_for(
     "dual_ct_md_save_velocity", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      uprev(m, 0, k, j, i) = w(m, IVX, k, j, i);
      uprev(m, 1, k, j, i) = w(m, IVY, k, j, i);
      uprev(m, 2, k, j, i) = w(m, IVZ, k, j, i);
     });
   par_for(
     "dual_ct_md_imp_e1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie + 1,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const bool use_left_ghost =
        (i == is) && (nghbr.d_view(m, ix1m).gid >= 0);
      const bool use_right_ghost =
        (i == ie + 1) && (nghbr.d_view(m, ix1p).gid >= 0);
      const int il = (i > is || use_left_ghost) ? i - 1 : is;
      const int ir = (i <= ie || use_right_ghost) ? i : ie;
      const Real ux = 0.5 * (w(m, IVX, k, j, il) + w(m, IVX, k, j, ir));
      const Real uy = 0.5 * (w(m, IVY, k, j, il) + w(m, IVY, k, j, ir));
      const Real uz = 0.5 * (w(m, IVZ, k, j, il) + w(m, IVZ, k, j, ir));
      const Real ey = 0.25
        * (es.x2f(m, k, j, il) + es.x2f(m, k, j + 1, il)
           + es.x2f(m, k, j, ir) + es.x2f(m, k, j + 1, ir));
      const Real ez = 0.25
        * (es.x3f(m, k, j, il) + es.x3f(m, k + 1, j, il)
           + es.x3f(m, k, j, ir) + es.x3f(m, k + 1, j, ir));
      Real ex_new, ey_new, ez_new;
      const Real local_eta =
        nonuniform_eta ? eta_f.x1f(m, k, j, i) : uniform_eta;
      const Real kap = adt/local_eta;
      srrmhd::ImplicitElectricField(
        ux, uy, uz, es.x1f(m, k, j, i), ey, ez, b.x1f(m, k, j, i),
        0.5 * (bcc(m, IBY, k, j, il) + bcc(m, IBY, k, j, ir)),
        0.5 * (bcc(m, IBZ, k, j, il) + bcc(m, IBZ, k, j, ir)), kap, ex_new,
        ey_new, ez_new);
      const Real old_e = e.x1f(m, k, j, i);
      delta.x1f(m, k, j, i) = fabs(ex_new - old_e);
      e.x1f(m, k, j, i) = (1.0 - relaxation) * old_e + relaxation * ex_new;
     });
   par_for(
     "dual_ct_md_imp_e2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const bool use_left_ghost =
        (j == js) && (nghbr.d_view(m, ix2m).gid >= 0);
      const bool use_right_ghost =
        (j == je + 1) && (nghbr.d_view(m, ix2p).gid >= 0);
      const int jl = (j > js || use_left_ghost) ? j - 1 : js;
      const int jr = (j <= je || use_right_ghost) ? j : je;
      const Real ux = 0.5 * (w(m, IVX, k, jl, i) + w(m, IVX, k, jr, i));
      const Real uy = 0.5 * (w(m, IVY, k, jl, i) + w(m, IVY, k, jr, i));
      const Real uz = 0.5 * (w(m, IVZ, k, jl, i) + w(m, IVZ, k, jr, i));
      const Real ex = 0.25
        * (es.x1f(m, k, jl, i) + es.x1f(m, k, jl, i + 1)
           + es.x1f(m, k, jr, i) + es.x1f(m, k, jr, i + 1));
      const Real ez = 0.25
        * (es.x3f(m, k, jl, i) + es.x3f(m, k + 1, jl, i)
           + es.x3f(m, k, jr, i) + es.x3f(m, k + 1, jr, i));
      Real ex_new, ey_new, ez_new;
      const Real local_eta =
        nonuniform_eta ? eta_f.x2f(m, k, j, i) : uniform_eta;
      const Real kap = adt/local_eta;
      srrmhd::ImplicitElectricField(
        ux, uy, uz, ex, es.x2f(m, k, j, i), ez,
        0.5 * (bcc(m, IBX, k, jl, i) + bcc(m, IBX, k, jr, i)),
        b.x2f(m, k, j, i),
        0.5 * (bcc(m, IBZ, k, jl, i) + bcc(m, IBZ, k, jr, i)), kap, ex_new,
        ey_new, ez_new);
      const Real old_e = e.x2f(m, k, j, i);
      delta.x2f(m, k, j, i) = fabs(ey_new - old_e);
      e.x2f(m, k, j, i) = (1.0 - relaxation) * old_e + relaxation * ey_new;
     });
   par_for(
     "dual_ct_md_imp_e3", DevExeSpace(), 0, nmb1, ks, e3ke, js, je, is, ie,
     KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const bool use_left_ghost =
        three_d && (k == ks) && (nghbr.d_view(m, ix3m).gid >= 0);
      const bool use_right_ghost =
        three_d && (k == ke + 1) && (nghbr.d_view(m, ix3p).gid >= 0);
      const int kl = three_d ? ((k > ks || use_left_ghost) ? k - 1 : ks) : ks;
      const int kr = three_d ? ((k <= ke || use_right_ghost) ? k : ke) : ks;
      const Real ux = 0.5 * (w(m, IVX, kl, j, i) + w(m, IVX, kr, j, i));
      const Real uy = 0.5 * (w(m, IVY, kl, j, i) + w(m, IVY, kr, j, i));
      const Real uz = 0.5 * (w(m, IVZ, kl, j, i) + w(m, IVZ, kr, j, i));
      const Real ex = 0.25
        * (es.x1f(m, kl, j, i) + es.x1f(m, kl, j, i + 1)
           + es.x1f(m, kr, j, i) + es.x1f(m, kr, j, i + 1));
      const Real ey = 0.25
        * (es.x2f(m, kl, j, i) + es.x2f(m, kl, j + 1, i)
           + es.x2f(m, kr, j, i) + es.x2f(m, kr, j + 1, i));
      Real ex_new, ey_new, ez_new;
      const Real local_eta =
        nonuniform_eta ? eta_f.x3f(m, k, j, i) : uniform_eta;
      const Real kap = adt/local_eta;
      srrmhd::ImplicitElectricField(
        ux, uy, uz, ex, ey, es.x3f(m, k, j, i),
        0.5 * (bcc(m, IBX, kl, j, i) + bcc(m, IBX, kr, j, i)),
        0.5 * (bcc(m, IBY, kl, j, i) + bcc(m, IBY, kr, j, i)),
        b.x3f(m, k, j, i), kap, ex_new, ey_new, ez_new);
      const Real old_e    = e.x3f(m, k, j, i);
      const Real relaxed  = (1.0 - relaxation) * old_e + relaxation * ez_new;
      const Real residual = fabs(ez_new - old_e);
      e.x3f(m, k, j, i) = relaxed;
      delta.x3f(m, k, j, i) = residual;
      if (!three_d) {
       e.x3f(m, ks + 1, j, i) = relaxed;
       delta.x3f(m, ks + 1, j, i) = residual;
      }
     });

   if (multi_block) {
    TaskStatus tstat = StartSharedElectricAverage(e);
    if (tstat != TaskStatus::complete) return tstat;
    ect_impl_state = ECTImplicitState::picard_face_recv;
    return TaskStatus::incomplete;
   }
   sync_interior();
   ect_local_residual = measure_residual();
#if MPI_PARALLEL_ENABLED
   ect_global_residual = 0.0;
   int ierr = MPI_Iallreduce(&ect_local_residual, &ect_global_residual, 1,
                             MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD,
                             &ect_reduce_request);
   if (ierr != MPI_SUCCESS) return TaskStatus::fail;
   ect_impl_state = ECTImplicitState::picard_reduce;
   return TaskStatus::incomplete;
#else
   ect_global_residual = ect_local_residual;
   return check_iteration();
#endif
  }

  if (ect_impl_state == ECTImplicitState::picard_face_recv) {
   TaskStatus tstat = FinishSharedElectricAverage(e);
   if (tstat != TaskStatus::complete) return tstat;
   sync_interior();
   tstat = StartElectricCellExchange();
   if (tstat != TaskStatus::complete) return tstat;
   ect_impl_state = ECTImplicitState::picard_cell_recv;
   return TaskStatus::incomplete;
  }

  if (ect_impl_state == ECTImplicitState::picard_cell_recv) {
   TaskStatus tstat = FinishElectricCellExchange();
   if (tstat != TaskStatus::complete) return tstat;
   ect_local_residual = measure_residual();
#if MPI_PARALLEL_ENABLED
   ect_global_residual = 0.0;
   int ierr = MPI_Iallreduce(&ect_local_residual, &ect_global_residual, 1,
                             MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD,
                             &ect_reduce_request);
   if (ierr != MPI_SUCCESS) return TaskStatus::fail;
   ect_impl_state = ECTImplicitState::picard_reduce;
   return TaskStatus::incomplete;
#else
   ect_global_residual = ect_local_residual;
   return check_iteration();
#endif
  }

  if (ect_impl_state == ECTImplicitState::picard_reduce) {
#if MPI_PARALLEL_ENABLED
   int complete = 0;
   int ierr = MPI_Test(&ect_reduce_request, &complete, MPI_STATUS_IGNORE);
   if (ierr != MPI_SUCCESS) return TaskStatus::fail;
   if (complete == 0) return TaskStatus::incomplete;
   ect_reduce_request = MPI_REQUEST_NULL;
   return check_iteration();
#else
   return TaskStatus::fail;
#endif
  }

  if (ect_impl_state == ECTImplicitState::final_average_recv) {
   TaskStatus tstat = FinishSharedElectricAverage(e);
   if (tstat != TaskStatus::complete) return tstat;
   tstat = StartElectricFaceExchange(e);
   if (tstat != TaskStatus::complete) return tstat;
   ect_impl_state = ECTImplicitState::final_face_recv;
   return TaskStatus::incomplete;
  }

  if (ect_impl_state == ECTImplicitState::final_face_recv) {
   TaskStatus tstat = FinishElectricFaceExchange(e);
   if (tstat != TaskStatus::complete) return tstat;
   return finish_stage();
  }
  return TaskStatus::fail;
 }

 // Assemble and freeze the explicit-plus-history right-hand side for this
 // stage.
 if (nonuniform_eta) {
  sync_active_e();
  recover_lagged_state();
 }
 par_for(
   "dual_ct_imex_star_e1", DevExeSpace(), 0, nmb1, is, ie + 1,
   KOKKOS_LAMBDA(int m, int i) {
    Real value = e.x1f(m, ks, js, i);
    if (istage > 1) {
     for (int s = 0; s <= istage - 2; ++s) {
      value += a_twid[istage - 2][s] * dt * re.x1f(m, s, ks, js, i);
     }
    }
    es.x1f(m, ks, js, i) = value;
    e.x1f(m, ks, js, i)  = value;
   });
 par_for(
   "dual_ct_imex_star_e23", DevExeSpace(), 0, nmb1, is, ie,
   KOKKOS_LAMBDA(int m, int i) {
    Real value2 = e.x2f(m, ks, js, i);
    Real value3 = e.x3f(m, ks, js, i);
    if (istage > 1) {
     for (int s = 0; s <= istage - 2; ++s) {
      const Real adt = a_twid[istage - 2][s] * dt;
      value2 += adt * re.x2f(m, s, ks, js, i);
      value3 += adt * re.x3f(m, s, ks, js, i);
     }
    }
    es.x2f(m, ks, js, i)     = value2;
    es.x2f(m, ks, js + 1, i) = value2;
    es.x3f(m, ks, js, i)     = value3;
    es.x3f(m, ks + 1, js, i) = value3;
    e.x2f(m, ks, js, i)      = value2;
    e.x2f(m, ks, js + 1, i)  = value2;
    e.x3f(m, ks, js, i)      = value3;
    e.x3f(m, ks + 1, js, i)  = value3;
   });

 const bool diagonal_solve = estage < pdriver->nexp_stages;
 if (diagonal_solve) {
  constexpr int max_iterations = 30;
  const Real tolerance = (sizeof(Real) == sizeof(float)) ? 2.0e-6 : 2.0e-12;
  const Real a_dt      = pdriver->a_impl * dt;
  bool converged       = false;

  FreezeFaceResistivity();

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
   par_for(
     "dual_ct_save_velocity", DevExeSpace(), 0, nmb1, is, ie,
     KOKKOS_LAMBDA(int m, int i) {
      uprev(m, 0, ks, js, i) = w(m, IVX, ks, js, i);
      uprev(m, 1, ks, js, i) = w(m, IVY, ks, js, i);
      uprev(m, 2, ks, js, i) = w(m, IVZ, ks, js, i);
     });

   par_for(
     "dual_ct_implicit_e1", DevExeSpace(), 0, nmb1, is, ie + 1,
     KOKKOS_LAMBDA(int m, int i) {
      const Real ux = 0.5 * (w(m, IVX, ks, js, i - 1) + w(m, IVX, ks, js, i));
      const Real uy = 0.5 * (w(m, IVY, ks, js, i - 1) + w(m, IVY, ks, js, i));
      const Real uz = 0.5 * (w(m, IVZ, ks, js, i - 1) + w(m, IVZ, ks, js, i));
      const Real by =
        0.5 * (bcc(m, IBY, ks, js, i - 1) + bcc(m, IBY, ks, js, i));
      const Real bz =
        0.5 * (bcc(m, IBZ, ks, js, i - 1) + bcc(m, IBZ, ks, js, i));
      const Real ey_star =
        0.5 * (es.x2f(m, ks, js, i - 1) + es.x2f(m, ks, js, i));
      const Real ez_star =
        0.5 * (es.x3f(m, ks, js, i - 1) + es.x3f(m, ks, js, i));
      const Real local_eta =
        nonuniform_eta ? eta_f.x1f(m, ks, js, i) : uniform_eta;
      const Real kappa = a_dt/local_eta;
      Real ex_new, ey_new, ez_new;
      srrmhd::ImplicitElectricField(
        ux, uy, uz, es.x1f(m, ks, js, i), ey_star, ez_star, b.x1f(m, ks, js, i),
        by, bz, kappa, ex_new, ey_new, ez_new);
      e.x1f(m, ks, js, i) = ex_new;
     });

   par_for(
     "dual_ct_implicit_e23", DevExeSpace(), 0, nmb1, is, ie,
     KOKKOS_LAMBDA(int m, int i) {
      const Real ex_star =
        0.5 * (es.x1f(m, ks, js, i) + es.x1f(m, ks, js, i + 1));
      const Real local_eta =
        nonuniform_eta ? eta_f.x2f(m, ks, js, i) : uniform_eta;
      const Real kappa = a_dt/local_eta;
      Real ex_new, ey_new, ez_new;
      srrmhd::ImplicitElectricField(
        w(m, IVX, ks, js, i), w(m, IVY, ks, js, i), w(m, IVZ, ks, js, i),
        ex_star, es.x2f(m, ks, js, i), es.x3f(m, ks, js, i),
        bcc(m, IBX, ks, js, i), bcc(m, IBY, ks, js, i), bcc(m, IBZ, ks, js, i),
        kappa, ex_new, ey_new, ez_new);
      e.x2f(m, ks, js, i)     = ey_new;
      e.x2f(m, ks, js + 1, i) = ey_new;
      e.x3f(m, ks, js, i)     = ez_new;
      e.x3f(m, ks + 1, js, i) = ez_new;
     });

   par_for(
     "dual_ct_implicit_sync", DevExeSpace(), 0, nmb1, is, ie,
     KOKKOS_LAMBDA(int m, int i) {
      Real ec1, ec2, ec3;
      srrmhd::ElectricFaceToCell(e, m, ks, js, i, ec1, ec2, ec3);
      u(m, srrmhd::IRE1, ks, js, i) = ec1;
      u(m, srrmhd::IRE2, ks, js, i) = ec2;
      u(m, srrmhd::IRE3, ks, js, i) = ec3;
     });
   peos->ConsToPrim(u, b, w, bcc, false, is, ie, js, js, ks, ks);

   Real max_change = 0.0;
   Kokkos::parallel_reduce(
     "dual_ct_velocity_change",
     Kokkos::MDRangePolicy<Kokkos::Rank<2> >(
       DevExeSpace(), {0, is}, {nmb1 + 1, ie + 1}),
     KOKKOS_LAMBDA(int m, int i, Real &max_value) {
      Real change = fabs(w(m, IVX, ks, js, i) - uprev(m, 0, ks, js, i));
      change =
        fmax(change, fabs(w(m, IVY, ks, js, i) - uprev(m, 1, ks, js, i)));
      change =
        fmax(change, fabs(w(m, IVZ, ks, js, i) - uprev(m, 2, ks, js, i)));
      max_value = fmax(max_value, change);
     },
     Kokkos::Max<Real>(max_change));
   if (max_change < tolerance) {
    converged = true;
    break;
   }
  }

  if (!converged) {
   if (global_variable::my_rank == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Face-centered electric IMEX iteration failed to "
              << "converge" << std::endl;
   }
   Kokkos::abort("Face-centered electric IMEX iteration failed");
   return TaskStatus::fail;
  }

  const int source_stage = istage - 1;
  par_for(
    "dual_ct_store_source_e1", DevExeSpace(), 0, nmb1, is, ie + 1,
    KOKKOS_LAMBDA(int m, int i) {
     re.x1f(m, source_stage, ks, js, i) =
       (e.x1f(m, ks, js, i) - es.x1f(m, ks, js, i)) / a_dt;
    });
  par_for(
    "dual_ct_store_source_e23", DevExeSpace(), 0, nmb1, is, ie,
    KOKKOS_LAMBDA(int m, int i) {
     const Real r2 = (e.x2f(m, ks, js, i) - es.x2f(m, ks, js, i)) / a_dt;
     const Real r3 = (e.x3f(m, ks, js, i) - es.x3f(m, ks, js, i)) / a_dt;
     re.x2f(m, source_stage, ks, js, i)     = r2;
     re.x2f(m, source_stage, ks, js + 1, i) = r2;
     re.x3f(m, source_stage, ks, js, i)     = r3;
     re.x3f(m, source_stage, ks + 1, js, i) = r3;
    });
 } else {
  par_for(
    "dual_ct_final_sync", DevExeSpace(), 0, nmb1, is, ie,
    KOKKOS_LAMBDA(int m, int i) {
     Real ec1, ec2, ec3;
     srrmhd::ElectricFaceToCell(e, m, ks, js, i, ec1, ec2, ec3);
     u(m, srrmhd::IRE1, ks, js, i) = ec1;
     u(m, srrmhd::IRE2, ks, js, i) = ec2;
     u(m, srrmhd::IRE3, ks, js, i) = ec3;
    });
 }

 return TaskStatus::complete;
}

} // namespace mhd
