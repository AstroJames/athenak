#ifndef MHD_RSOLVERS_LLF_SRRMHD_HPP_
#define MHD_RSOLVERS_LLF_SRRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srrmhd.hpp
//! \brief LLF Riemann solver wrapper for full resistive SRMHD.

#include "eos/resistive_srmhd.hpp"
#include "mhd/rsolvers/llf_srrmhd_singlestate.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void LLF_SRR()
//! \brief Compute resistive-SRMHD LLF fluxes and face electric fields.

KOKKOS_INLINE_FUNCTION
void LLF_SRR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
     const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> emf_z, DvceArray4D<Real> emf_y) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX)+1)%3;
  int ibz = ((ivx-IVX)+2)%3;
  int iex = srrmhd::IRE1 + (ivx-IVX);
  int iey = srrmhd::IRE1 + ((ivx-IVX)+1)%3;
  int iez = srrmhd::IRE1 + ((ivx-IVX)+2)%3;

  par_for_inner(member, il, iu, [&](const int i) {
    srrmhd::SRRMHDPrim1D wli, wri;
    wli.d = wl(IDN, i);
    wli.vx = wl(ivx, i);
    wli.vy = wl(ivy, i);
    wli.vz = wl(ivz, i);
    wli.e = wl(IEN, i);
    wli.ex = wl(iex, i);
    wli.ey = wl(iey, i);
    wli.ez = wl(iez, i);
    wli.bx = bx(m, k, j, i);
    wli.by = bl(iby, i);
    wli.bz = bl(ibz, i);

    wri.d = wr(IDN, i);
    wri.vx = wr(ivx, i);
    wri.vy = wr(ivy, i);
    wri.vz = wr(ivz, i);
    wri.e = wr(IEN, i);
    wri.ex = wr(iex, i);
    wri.ey = wr(iey, i);
    wri.ez = wr(iez, i);
    wri.bx = bx(m, k, j, i);
    wri.by = br(iby, i);
    wri.bz = br(ibz, i);

    srrmhd::SRRMHDCons1D flux;
    SingleStateLLF_SRRMHD(wli, wri, eos, flux);

    flx(m, IDN, k, j, i) = flux.d;
    flx(m, ivx, k, j, i) = flux.mx;
    flx(m, ivy, k, j, i) = flux.my;
    flx(m, ivz, k, j, i) = flux.mz;
    flx(m, IEN, k, j, i) = flux.e;
    flx(m, iex, k, j, i) = flux.ex;
    flx(m, iey, k, j, i) = flux.ey;
    flx(m, iez, k, j, i) = flux.ez;

    // AthenaK's CT interface stores the physical electric field, while flux.by/bz are
    // the conservative magnetic fluxes F(By)=-Ez and F(Bz)=Ey in the local basis.
    emf_z(m, k, j, i) = -flux.by;
    emf_y(m, k, j, i) = flux.bz;
  });

  (void)indcs;
  (void)size;
  (void)coord;
}

} // namespace mhd

#endif // MHD_RSOLVERS_LLF_SRRMHD_HPP_
