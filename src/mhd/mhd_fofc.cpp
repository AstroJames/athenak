//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fofc.cpp
//! \brief Implements functions for first-order flux correction (FOFC) algorithm.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/rsolvers/llf_mhd_singlestate.hpp"
#include "mhd/rsolvers/llf_srrmhd_singlestate.hpp"
#include "mhd/relativistic_viscosity.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::FOFC
//! \brief Implements first-order flux-correction (FOFC) algorithm for MHD.  First an
//! estimate of the updated conserved variables is made. This estimate is then used to
//! flag any cell where floors will be required during the conversion to primitives. Then
//! the fluxes on the faces of flagged cells are replaced with first-order LLF fluxes.
//! Often this is enough to prevent floors from being needed.  The FOFC infrastructure is
//! also exploited for BH excision.  If a cell is about the horizon, FOFC is automatically
//! triggered (without estimating updated conserved variables).

void MHD::FOFC(Driver *pdriver, int stage) {
  if (is_resistive_rel) {
    FOFCResistive(pdriver, stage);
    return;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb = pmy_pack->nmb_thispack;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &size = pmy_pack->pmb->mb_size;

  auto &bcc0_ = bcc0;
  auto &e3x1_ = e3x1;
  auto &e2x1_ = e2x1;
  auto &e1x2_ = e1x2;
  auto &e3x2_ = e3x2;
  auto &e2x3_ = e2x3;
  auto &e1x3_ = e1x3;

  if (use_fofc) {
    Real &gam0 = pdriver->gam0[stage-1];
    Real &gam1 = pdriver->gam1[stage-1];
    Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

    int &nmhd_ = nmhd;
    auto &u0_ = u0;
    auto &u1_ = u1;
    auto &utest_ = utest;
    auto &bcctest_ = bcctest;
    auto &b1_ = b1;

    // Index bounds
    int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) { jl = js-1, ju = je+1; }
    if (three_d) { kl = ks-1, ku = ke+1; }

    // Estimate updated conserved variables and cell-centered fields
    par_for("FOFC-newu", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size.d_view(m).dx1;
      Real dtodx2 = beta_dt/size.d_view(m).dx2;
      Real dtodx3 = beta_dt/size.d_view(m).dx3;

      // Estimate conserved variables
      for (int n=0; n<nmhd_; ++n) {
        Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
        if (multi_d) {
          divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
        }
        if (three_d) {
          divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
        }
        utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
      }

      // Estimate updated cell-centered fields
      Real b1old = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      Real b2old = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      Real b3old = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));

      bcctest_(m,IBX,k,j,i) = gam0*bcc0_(m,IBX,k,j,i) + gam1*b1old;
      bcctest_(m,IBY,k,j,i) = gam0*bcc0_(m,IBY,k,j,i) + gam1*b2old;
      bcctest_(m,IBZ,k,j,i) = gam0*bcc0_(m,IBZ,k,j,i) + gam1*b3old;

      bcctest_(m,IBY,k,j,i) += dtodx1*(e3x1_(m,k,j,i+1) - e3x1_(m,k,j,i));
      bcctest_(m,IBZ,k,j,i) -= dtodx1*(e2x1_(m,k,j,i+1) - e2x1_(m,k,j,i));
      if (multi_d) {
        bcctest_(m,IBX,k,j,i) -= dtodx2*(e3x2_(m,k,j+1,i) - e3x2_(m,k,j,i));
        bcctest_(m,IBZ,k,j,i) += dtodx2*(e1x2_(m,k,j+1,i) - e1x2_(m,k,j,i));
      }
      if (three_d) {
        bcctest_(m,IBX,k,j,i) += dtodx3*(e2x3_(m,k+1,j,i) - e2x3_(m,k,j,i));
        bcctest_(m,IBY,k,j,i) -= dtodx3*(e1x3_(m,k+1,j,i) - e1x3_(m,k,j,i));
      }
    });

    // Test whether conversion to primitives requires floors
    // Note b0 and w0 passed to function, but not used/changed.
    peos->ConsToPrim(utest_, b0, w0, bcctest_, true, il, iu, jl, ju, kl, ku);
  }

  auto &coord = pmy_pack->pcoord->coord_data;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = peos->eos_data;
  auto &use_fofc_ = use_fofc;
  auto fofc_ = fofc;
  auto &use_excise_ = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &w0_ = w0;
  auto &b0_ = b0;

  // Index bounds
  int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-1, ju = je+1; }
  if (three_d) { kl = ks-1, ku = ke+1; }

  // Replace fluxes with first-order LLF fluxes at i,j,k faces for any cell where FOFC
  // and/or excision is used (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Check for GR + excision
    bool fofc_excision = false;
    if (is_gr) {
      if (use_excise_) { fofc_excision = excision_flux_(m,k,j,i); }
    }

    // Apply FOFC
    if (fofc_flag || fofc_excision) {
      // load W_{i-1} state
      MHDPrim1D wim1;
      wim1.d  = w0_(m,IDN,k,j,i-1);
      wim1.vx = w0_(m,IVX,k,j,i-1);
      wim1.vy = w0_(m,IVY,k,j,i-1);
      wim1.vz = w0_(m,IVZ,k,j,i-1);
      if (eos.is_ideal) {wim1.e  = w0_(m,IEN,k,j,i-1);}
      wim1.by = bcc0_(m,IBY,k,j,i-1);
      wim1.bz = bcc0_(m,IBZ,k,j,i-1);

      // load W_{i} state
      MHDPrim1D wi;
      wi.d  = w0_(m,IDN,k,j,i);
      wi.vx = w0_(m,IVX,k,j,i);
      wi.vy = w0_(m,IVY,k,j,i);
      wi.vz = w0_(m,IVZ,k,j,i);
      if (eos.is_ideal) {wi.e = w0_(m,IEN,k,j,i);}
      wi.by = bcc0_(m,IBY,k,j,i);
      wi.bz = bcc0_(m,IBZ,k,j,i);

      // compute new 1st-order LLF flux at i-face
      {
        Real bxi = b0_.x1f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = LeftEdgeX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wim1, wi, bxi, x1v, x2v, x3v, IVX, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wim1, wi, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wim1, wi, bxi, eos, flux);
        }

        // store 1st-order fluxes.
        flx1(m,IDN,k,j,i) = flux.d;
        flx1(m,IM1,k,j,i) = flux.mx;
        flx1(m,IM2,k,j,i) = flux.my;
        flx1(m,IM3,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx1(m,IEN,k,j,i) = flux.e;}
        e3x1_(m,k,j,i) = flux.by;
        e2x1_(m,k,j,i) = flux.bz;
      }

      if (multi_d) {
        // load W_{j-1} state, permutting components of vectors
        MHDPrim1D wjm1;
        wjm1.d  = w0_(m,IDN,k,j-1,i);
        wjm1.vx = w0_(m,IVY,k,j-1,i);
        wjm1.vy = w0_(m,IVZ,k,j-1,i);
        wjm1.vz = w0_(m,IVX,k,j-1,i);
        if (eos.is_ideal) {wjm1.e = w0_(m,IEN,k,j-1,i);}
        wjm1.by = bcc0_(m,IBZ,k,j-1,i);
        wjm1.bz = bcc0_(m,IBX,k,j-1,i);

        // load W_{j} state, permutting components of vectors
        MHDPrim1D wj;
        wj.d  = w0_(m,IDN,k,j,i);
        wj.vx = w0_(m,IVY,k,j,i);
        wj.vy = w0_(m,IVZ,k,j,i);
        wj.vz = w0_(m,IVX,k,j,i);
        if (eos.is_ideal) {wj.e = w0_(m,IEN,k,j,i);}
        wj.by = bcc0_(m,IBZ,k,j,i);
        wj.bz = bcc0_(m,IBX,k,j,i);

        // compute new first-order flux at j-face
        Real bxi = b0_.x2f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wjm1, wj, bxi, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wjm1, wj, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wjm1, wj, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx2(m,IDN,k,j,i) = flux.d;
        flx2(m,IM2,k,j,i) = flux.mx;
        flx2(m,IM3,k,j,i) = flux.my;
        flx2(m,IM1,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j,i) = flux.e;}
        e1x2_(m,k,j,i) = flux.by;
        e3x2_(m,k,j,i) = flux.bz;
      }

      if (three_d) {
        // load W_{k-1} state, permutting components of vectors
        MHDPrim1D wkm1;
        wkm1.d  = w0_(m,IDN,k-1,j,i);
        wkm1.vx = w0_(m,IVZ,k-1,j,i);
        wkm1.vy = w0_(m,IVX,k-1,j,i);
        wkm1.vz = w0_(m,IVY,k-1,j,i);
        if (eos.is_ideal) {wkm1.e = w0_(m,IEN,k-1,j,i);}
        wkm1.by = bcc0_(m,IBX,k-1,j,i);
        wkm1.bz = bcc0_(m,IBY,k-1,j,i);

        // load W_{k} state, permutting components of vectors
        MHDPrim1D wk;
        wk.d  = w0_(m,IDN,k,j,i);
        wk.vx = w0_(m,IVZ,k,j,i);
        wk.vy = w0_(m,IVX,k,j,i);
        wk.vz = w0_(m,IVY,k,j,i);
        if (eos.is_ideal) {wk.e = w0_(m,IEN,k,j,i);}
        wk.by = bcc0_(m,IBX,k,j,i);
        wk.bz = bcc0_(m,IBY,k,j,i);

        // compute new first-order flux at k-face
        Real bxi = b0_.x3f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wkm1, wk, bxi, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wkm1, wk, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wkm1, wk, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx3(m,IDN,k,j,i) = flux.d;
        flx3(m,IM3,k,j,i) = flux.mx;
        flx3(m,IM1,k,j,i) = flux.my;
        flx3(m,IM2,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k,j,i) = flux.e;}
        e2x3_(m,k,j,i) = flux.by;
        e1x3_(m,k,j,i) = flux.bz;
      }
    }
  });

  // Replace fluxes with first-order LLF fluxes at i+1,j+1,k+1 faces for any cell where
  // FOFC and/or excision is used (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Check for GR + excision
    bool fofc_excision = false;
    if (is_gr) {
      if (use_excise_) { fofc_excision = excision_flux_(m,k,j,i); }
    }

    // Apply FOFC
    if (fofc_flag || fofc_excision) {
      // load W_{i} state
      MHDPrim1D wi;
      wi.d  = w0_(m,IDN,k,j,i);
      wi.vx = w0_(m,IVX,k,j,i);
      wi.vy = w0_(m,IVY,k,j,i);
      wi.vz = w0_(m,IVZ,k,j,i);
      if (eos.is_ideal) {wi.e = w0_(m,IEN,k,j,i);}
      wi.by = bcc0_(m,IBY,k,j,i);
      wi.bz = bcc0_(m,IBZ,k,j,i);

      // load W_{i+1} state
      MHDPrim1D wip1;
      wip1.d  = w0_(m,IDN,k,j,i+1);
      wip1.vx = w0_(m,IVX,k,j,i+1);
      wip1.vy = w0_(m,IVY,k,j,i+1);
      wip1.vz = w0_(m,IVZ,k,j,i+1);
      if (eos.is_ideal) {wip1.e = w0_(m,IEN,k,j,i+1);}
      wip1.by = bcc0_(m,IBY,k,j,i+1);
      wip1.bz = bcc0_(m,IBZ,k,j,i+1);

      // compute new 1st-order LLF flux at (i+1)-face
      {
        Real bxi = b0_.x1f(m,k,j,i+1);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = LeftEdgeX(i+1-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wi, wip1, bxi, x1v, x2v, x3v, IVX, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wi, wip1, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wi, wip1, bxi, eos, flux);
        }

        // store 1st-order fluxes.
        flx1(m,IDN,k,j,i+1) = flux.d;
        flx1(m,IM1,k,j,i+1) = flux.mx;
        flx1(m,IM2,k,j,i+1) = flux.my;
        flx1(m,IM3,k,j,i+1) = flux.mz;
        if (eos.is_ideal) {flx1(m,IEN,k,j,i+1) = flux.e;}
        e3x1_(m,k,j,i+1) = flux.by;
        e2x1_(m,k,j,i+1) = flux.bz;
      }

      if (multi_d) {
        // load W_{j} state, permutting components of vectors
        MHDPrim1D wj;
        wj.d  = w0_(m,IDN,k,j,i);
        wj.vx = w0_(m,IVY,k,j,i);
        wj.vy = w0_(m,IVZ,k,j,i);
        wj.vz = w0_(m,IVX,k,j,i);
        if (eos.is_ideal) {wj.e = w0_(m,IEN,k,j,i);}
        wj.by = bcc0_(m,IBZ,k,j,i);
        wj.bz = bcc0_(m,IBX,k,j,i);

        // load W_{j+1} state, permutting components of vectors
        MHDPrim1D wjp1;
        wjp1.d  = w0_(m,IDN,k,j+1,i);
        wjp1.vx = w0_(m,IVY,k,j+1,i);
        wjp1.vy = w0_(m,IVZ,k,j+1,i);
        wjp1.vz = w0_(m,IVX,k,j+1,i);
        if (eos.is_ideal) {wjp1.e = w0_(m,IEN,k,j+1,i);}
        wjp1.by = bcc0_(m,IBZ,k,j+1,i);
        wjp1.bz = bcc0_(m,IBX,k,j+1,i);

        // compute new first-order flux at (j+1)-face
        Real bxi = b0_.x2f(m,k,j+1,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j+1-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wj, wjp1, bxi, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wj, wjp1, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wj, wjp1, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx2(m,IDN,k,j+1,i) = flux.d;
        flx2(m,IM2,k,j+1,i) = flux.mx;
        flx2(m,IM3,k,j+1,i) = flux.my;
        flx2(m,IM1,k,j+1,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j+1,i) = flux.e;}
        e1x2_(m,k,j+1,i) = flux.by;
        e3x2_(m,k,j+1,i) = flux.bz;
      }

      if (three_d) {
        // load W_{k} state, permutting components of vectors
        MHDPrim1D wk;
        wk.d  = w0_(m,IDN,k,j,i);
        wk.vx = w0_(m,IVZ,k,j,i);
        wk.vy = w0_(m,IVX,k,j,i);
        wk.vz = w0_(m,IVY,k,j,i);
        if (eos.is_ideal) {wk.e = w0_(m,IEN,k,j,i);}
        wk.by = bcc0_(m,IBX,k,j,i);
        wk.bz = bcc0_(m,IBY,k,j,i);

        // load W_{k+1} state, permutting components of vectors
        MHDPrim1D wkp1;
        wkp1.d  = w0_(m,IDN,k+1,j,i);
        wkp1.vx = w0_(m,IVZ,k+1,j,i);
        wkp1.vy = w0_(m,IVX,k+1,j,i);
        wkp1.vz = w0_(m,IVY,k+1,j,i);
        if (eos.is_ideal) {wkp1.e = w0_(m,IEN,k+1,j,i);}
        wkp1.by = bcc0_(m,IBX,k+1,j,i);
        wkp1.bz = bcc0_(m,IBY,k+1,j,i);

        // compute new first-order flux at (k+1)-face
        Real bxi = b0_.x3f(m,k+1,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wk, wkp1, bxi, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wk, wkp1, bxi, eos, flux);
        } else {
          SingleStateLLF_MHD(wk, wkp1, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx3(m,IDN,k+1,j,i) = flux.d;
        flx3(m,IM3,k+1,j,i) = flux.mx;
        flx3(m,IM1,k+1,j,i) = flux.my;
        flx3(m,IM2,k+1,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k+1,j,i) = flux.e;}
        e2x3_(m,k+1,j,i) = flux.by;
        e1x3_(m,k+1,j,i) = flux.bz;
      }
    }
  });

  // reset FOFC flag (do not reset excision flag)
  if (use_fofc_) {
    Kokkos::deep_copy(fofc, false);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \brief FOFC for dynamic resistive SRMHD, including Israel--Stewart shear.
//!
//! The resistive system evolves E as part of the cell-centered conserved state and uses
//! face electric fields for constrained transport of B.  A corrected face must therefore
//! replace the fluid, electric, magnetic-CT, and (when enabled) conservative-shear fluxes
//! as one consistent first-order LLF state.

void MHD::FOFCResistive(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int nmb = pmy_pack->nmb_thispack;

  const Real gam0 = pdriver->gam0[stage-1];
  const Real gam1 = pdriver->gam1[stage-1];
  const Real beta_dt = pdriver->beta[stage-1]*pmy_pack->pmesh->dt;
  auto u0_ = u0;
  auto u1_ = u1;
  auto utest_ = utest;
  auto bcc0_ = bcc0;
  auto bcctest_ = bcctest;
  auto b1_ = b1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto e3x1_ = e3x1;
  auto e2x1_ = e2x1;
  auto e1x2_ = e1x2;
  auto e3x2_ = e3x2;
  auto e2x3_ = e2x3;
  auto e1x3_ = e1x3;
  auto &size = pmy_pack->pmb->mb_size;

  int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) {
    jl = js-1;
    ju = je+1;
  }
  if (three_d) {
    kl = ks-1;
    ku = ke+1;
  }

  // Provisional explicit RK update used only to decide where fallback is required.
  par_for("FOFC-SRRMHD-newu", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dtodx1 = beta_dt/size.d_view(m).dx1;
    const Real dtodx2 = beta_dt/size.d_view(m).dx2;
    const Real dtodx3 = beta_dt/size.d_view(m).dx3;
    for (int n = 0; n < srrmhd::NSRRMHD; ++n) {
      Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
      if (multi_d) {
        divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
      }
      if (three_d) {
        divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
      }
      utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
    }

    const Real b1old = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
    const Real b2old = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
    const Real b3old = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));
    bcctest_(m,IBX,k,j,i) = gam0*bcc0_(m,IBX,k,j,i) + gam1*b1old;
    bcctest_(m,IBY,k,j,i) = gam0*bcc0_(m,IBY,k,j,i) + gam1*b2old;
    bcctest_(m,IBZ,k,j,i) = gam0*bcc0_(m,IBZ,k,j,i) + gam1*b3old;
    bcctest_(m,IBY,k,j,i) += dtodx1*(e3x1_(m,k,j,i+1) - e3x1_(m,k,j,i));
    bcctest_(m,IBZ,k,j,i) -= dtodx1*(e2x1_(m,k,j,i+1) - e2x1_(m,k,j,i));
    if (multi_d) {
      bcctest_(m,IBX,k,j,i) -= dtodx2*(e3x2_(m,k,j+1,i) - e3x2_(m,k,j,i));
      bcctest_(m,IBZ,k,j,i) += dtodx2*(e1x2_(m,k,j+1,i) - e1x2_(m,k,j,i));
    }
    if (three_d) {
      bcctest_(m,IBX,k,j,i) += dtodx3*(e2x3_(m,k+1,j,i) - e2x3_(m,k,j,i));
      bcctest_(m,IBY,k,j,i) -= dtodx3*(e1x3_(m,k+1,j,i) - e1x3_(m,k,j,i));
    }
  });

  if (relativistic_viscosity_data.enabled) {
    auto visc_u0_ = visc_u0;
    auto visc_u1_ = visc_u1;
    auto visc_utest_ = visc_utest;
    auto visc_flx1 = visc_flx.x1f;
    auto visc_flx2 = visc_flx.x2f;
    auto visc_flx3 = visc_flx.x3f;
    par_for("FOFC-visc-newu", DevExeSpace(), 0, nmb-1, 0, srrmhd::NVISC-1,
            kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      Real divf = beta_dt*(visc_flx1(m,n,k,j,i+1) - visc_flx1(m,n,k,j,i))
                  /size.d_view(m).dx1;
      if (multi_d) {
        divf += beta_dt*(visc_flx2(m,n,k,j+1,i) - visc_flx2(m,n,k,j,i))
                    /size.d_view(m).dx2;
      }
      if (three_d) {
        divf += beta_dt*(visc_flx3(m,n,k+1,j,i) - visc_flx3(m,n,k,j,i))
                    /size.d_view(m).dx3;
      }
      visc_utest_(m,n,k,j,i) = gam0*visc_u0_(m,n,k,j,i)
                                + gam1*visc_u1_(m,n,k,j,i) - divf;
    });

    // Recover with the provisional spatial shear held fixed.  This removes its temporal
    // energy-momentum share before testing the fluid floors and Lorentz-factor ceiling.
    auto eos = peos->eos_data;
    auto w0_ = w0;
    auto fofc_ = fofc;
    int newly_flagged = 0;
    Kokkos::parallel_reduce("FOFC-visc-c2p", Kokkos::MDRangePolicy<DevExeSpace,
        Kokkos::Rank<4>>({0,kl,jl,il}, {nmb,ku+1,ju+1,iu+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, int &sum) {
      srrmhd::SRRMHDCons1D state;
      state.d = utest_(m,IDN,k,j,i);
      state.mx = utest_(m,IM1,k,j,i);
      state.my = utest_(m,IM2,k,j,i);
      state.mz = utest_(m,IM3,k,j,i);
      state.e = utest_(m,IEN,k,j,i);
      state.ex = utest_(m,srrmhd::IRE1,k,j,i);
      state.ey = utest_(m,srrmhd::IRE2,k,j,i);
      state.ez = utest_(m,srrmhd::IRE3,k,j,i);
      state.bx = bcctest_(m,IBX,k,j,i);
      state.by = bcctest_(m,IBY,k,j,i);
      state.bz = bcctest_(m,IBZ,k,j,i);

      bool bad = !(state.d > 0.0) || !isfinite(state.d);
      srrmhd::ShearStress pi;
      if (!bad) {
        pi.p11 = visc_utest_(m,srrmhd::IVP11,k,j,i)/state.d;
        pi.p22 = visc_utest_(m,srrmhd::IVP22,k,j,i)/state.d;
        pi.p33 = visc_utest_(m,srrmhd::IVP33,k,j,i)/state.d;
        pi.p12 = visc_utest_(m,srrmhd::IVP12,k,j,i)/state.d;
        pi.p13 = visc_utest_(m,srrmhd::IVP13,k,j,i)/state.d;
        pi.p23 = visc_utest_(m,srrmhd::IVP23,k,j,i)/state.d;
        srrmhd::SRRMHDPrim1D guess;
        guess.d = w0_(m,IDN,k,j,i);
        guess.vx = w0_(m,IVX,k,j,i);
        guess.vy = w0_(m,IVY,k,j,i);
        guess.vz = w0_(m,IVZ,k,j,i);
        guess.e = w0_(m,IEN,k,j,i);
        guess.ex = state.ex;
        guess.ey = state.ey;
        guess.ez = state.ez;
        guess.bx = state.bx;
        guess.by = state.by;
        guess.bz = state.bz;
        srrmhd::SRRMHDPrim1D recovered = guess;
        srrmhd::ShearStress recovered_pi;
        int iterations = 0;
        const bool success = srrmhd::SingleC2P_IdealSRRMHDImplicitViscous(
            state, eos, state.ex, state.ey, state.ez, 0.0, pi, pi, 0.0, 1.0e30,
            true, guess, recovered, recovered_pi, iterations);
        const Real lor = sqrt(1.0 + SQR(recovered.vx) + SQR(recovered.vy)
                              + SQR(recovered.vz));
        bad = !success || !isfinite(recovered.d) || !isfinite(recovered.e)
              || recovered.d <= eos.dfloor
              || eos.IdealGasPressure(recovered.e) <= eos.pfloor
              || !isfinite(lor) || lor > eos.gamma_max;
      }
      if (bad) {
        if (!fofc_(m,k,j,i)) ++sum;
        fofc_(m,k,j,i) = true;
      }
    }, newly_flagged);
    pmy_pack->pmesh->ecounter.nfofc += newly_flagged;
  } else {
    peos->ConsToPrim(utest, b0, w0, bcctest, true, il, iu, jl, ju, kl, ku);
  }

  if (force_fofc) {
    Kokkos::deep_copy(fofc, true);
  }

  auto fofc_ = fofc;
  auto w0_ = w0;
  auto b0_ = b0;
  const auto eos = peos->eos_data;
  const bool viscosity = relativistic_viscosity_data.enabled;
  auto visc_w0_ = visc_w0;
  auto visc_flx1 = visc_flx.x1f;
  auto visc_flx2 = visc_flx.x2f;
  auto visc_flx3 = visc_flx.x3f;

  // Correct each face once if either adjacent cell was flagged.
  par_for("FOFC-SRRMHD-x1", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (!(fofc_(m,k,j,i-1) || fofc_(m,k,j,i))) return;
    srrmhd::SRRMHDPrim1D wl, wr;
    wl.d=w0_(m,IDN,k,j,i-1); wl.vx=w0_(m,IVX,k,j,i-1);
    wl.vy=w0_(m,IVY,k,j,i-1); wl.vz=w0_(m,IVZ,k,j,i-1);
    wl.e=w0_(m,IEN,k,j,i-1); wl.ex=w0_(m,srrmhd::IRE1,k,j,i-1);
    wl.ey=w0_(m,srrmhd::IRE2,k,j,i-1); wl.ez=w0_(m,srrmhd::IRE3,k,j,i-1);
    wl.bx=b0_.x1f(m,k,j,i); wl.by=bcc0_(m,IBY,k,j,i-1);
    wl.bz=bcc0_(m,IBZ,k,j,i-1);
    wr.d=w0_(m,IDN,k,j,i); wr.vx=w0_(m,IVX,k,j,i);
    wr.vy=w0_(m,IVY,k,j,i); wr.vz=w0_(m,IVZ,k,j,i);
    wr.e=w0_(m,IEN,k,j,i); wr.ex=w0_(m,srrmhd::IRE1,k,j,i);
    wr.ey=w0_(m,srrmhd::IRE2,k,j,i); wr.ez=w0_(m,srrmhd::IRE3,k,j,i);
    wr.bx=b0_.x1f(m,k,j,i); wr.by=bcc0_(m,IBY,k,j,i);
    wr.bz=bcc0_(m,IBZ,k,j,i);
    srrmhd::SRRMHDCons1D flux;
    SingleStateLLF_SRRMHD(wl,wr,eos,flux);
    flx1(m,IDN,k,j,i)=flux.d; flx1(m,IM1,k,j,i)=flux.mx;
    flx1(m,IM2,k,j,i)=flux.my; flx1(m,IM3,k,j,i)=flux.mz;
    flx1(m,IEN,k,j,i)=flux.e; flx1(m,srrmhd::IRE1,k,j,i)=flux.ex;
    flx1(m,srrmhd::IRE2,k,j,i)=flux.ey; flx1(m,srrmhd::IRE3,k,j,i)=flux.ez;
    e3x1_(m,k,j,i)=-flux.by; e2x1_(m,k,j,i)=flux.bz;
    if (viscosity) {
      srrmhd::ShearStress pl, pr;
      pl.p11=visc_w0_(m,srrmhd::IVP11,k,j,i-1);
      pl.p22=visc_w0_(m,srrmhd::IVP22,k,j,i-1);
      pl.p33=visc_w0_(m,srrmhd::IVP33,k,j,i-1);
      pl.p12=visc_w0_(m,srrmhd::IVP12,k,j,i-1);
      pl.p13=visc_w0_(m,srrmhd::IVP13,k,j,i-1);
      pl.p23=visc_w0_(m,srrmhd::IVP23,k,j,i-1);
      pr.p11=visc_w0_(m,srrmhd::IVP11,k,j,i);
      pr.p22=visc_w0_(m,srrmhd::IVP22,k,j,i);
      pr.p33=visc_w0_(m,srrmhd::IVP33,k,j,i);
      pr.p12=visc_w0_(m,srrmhd::IVP12,k,j,i);
      pr.p13=visc_w0_(m,srrmhd::IVP13,k,j,i);
      pr.p23=visc_w0_(m,srrmhd::IVP23,k,j,i);
      Real sf[srrmhd::NVISC], mf[3], ef;
      srrmhd::ViscousLLFContributions(0,wl.d,wl.vx,wl.vy,wl.vz,pl,
          wr.d,wr.vx,wr.vy,wr.vz,pr,sf,mf,ef);
      for (int n=0; n<srrmhd::NVISC; ++n) visc_flx1(m,n,k,j,i)=sf[n];
      flx1(m,IM1,k,j,i)+=mf[0]; flx1(m,IM2,k,j,i)+=mf[1];
      flx1(m,IM3,k,j,i)+=mf[2]; flx1(m,IEN,k,j,i)+=ef;
    }
  });

  if (multi_d) {
    par_for("FOFC-SRRMHD-x2", DevExeSpace(), 0, nmb-1, kl, ku, js, je+1, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (!(fofc_(m,k,j-1,i) || fofc_(m,k,j,i))) return;
      srrmhd::SRRMHDPrim1D wl, wr;
      wl.d=w0_(m,IDN,k,j-1,i); wl.vx=w0_(m,IVY,k,j-1,i);
      wl.vy=w0_(m,IVZ,k,j-1,i); wl.vz=w0_(m,IVX,k,j-1,i);
      wl.e=w0_(m,IEN,k,j-1,i); wl.ex=w0_(m,srrmhd::IRE2,k,j-1,i);
      wl.ey=w0_(m,srrmhd::IRE3,k,j-1,i); wl.ez=w0_(m,srrmhd::IRE1,k,j-1,i);
      wl.bx=b0_.x2f(m,k,j,i); wl.by=bcc0_(m,IBZ,k,j-1,i);
      wl.bz=bcc0_(m,IBX,k,j-1,i);
      wr.d=w0_(m,IDN,k,j,i); wr.vx=w0_(m,IVY,k,j,i);
      wr.vy=w0_(m,IVZ,k,j,i); wr.vz=w0_(m,IVX,k,j,i);
      wr.e=w0_(m,IEN,k,j,i); wr.ex=w0_(m,srrmhd::IRE2,k,j,i);
      wr.ey=w0_(m,srrmhd::IRE3,k,j,i); wr.ez=w0_(m,srrmhd::IRE1,k,j,i);
      wr.bx=b0_.x2f(m,k,j,i); wr.by=bcc0_(m,IBZ,k,j,i);
      wr.bz=bcc0_(m,IBX,k,j,i);
      srrmhd::SRRMHDCons1D flux;
      SingleStateLLF_SRRMHD(wl,wr,eos,flux);
      flx2(m,IDN,k,j,i)=flux.d; flx2(m,IM2,k,j,i)=flux.mx;
      flx2(m,IM3,k,j,i)=flux.my; flx2(m,IM1,k,j,i)=flux.mz;
      flx2(m,IEN,k,j,i)=flux.e; flx2(m,srrmhd::IRE2,k,j,i)=flux.ex;
      flx2(m,srrmhd::IRE3,k,j,i)=flux.ey; flx2(m,srrmhd::IRE1,k,j,i)=flux.ez;
      e1x2_(m,k,j,i)=-flux.by; e3x2_(m,k,j,i)=flux.bz;
      if (viscosity) {
        srrmhd::ShearStress pl, pr;
        pl.p11=visc_w0_(m,srrmhd::IVP11,k,j-1,i);
        pl.p22=visc_w0_(m,srrmhd::IVP22,k,j-1,i);
        pl.p33=visc_w0_(m,srrmhd::IVP33,k,j-1,i);
        pl.p12=visc_w0_(m,srrmhd::IVP12,k,j-1,i);
        pl.p13=visc_w0_(m,srrmhd::IVP13,k,j-1,i);
        pl.p23=visc_w0_(m,srrmhd::IVP23,k,j-1,i);
        pr.p11=visc_w0_(m,srrmhd::IVP11,k,j,i);
        pr.p22=visc_w0_(m,srrmhd::IVP22,k,j,i);
        pr.p33=visc_w0_(m,srrmhd::IVP33,k,j,i);
        pr.p12=visc_w0_(m,srrmhd::IVP12,k,j,i);
        pr.p13=visc_w0_(m,srrmhd::IVP13,k,j,i);
        pr.p23=visc_w0_(m,srrmhd::IVP23,k,j,i);
        Real sf[srrmhd::NVISC], mf[3], ef;
        srrmhd::ViscousLLFContributions(1,wl.d,wl.vz,wl.vx,wl.vy,pl,
            wr.d,wr.vz,wr.vx,wr.vy,pr,sf,mf,ef);
        for (int n=0; n<srrmhd::NVISC; ++n) visc_flx2(m,n,k,j,i)=sf[n];
        flx2(m,IM1,k,j,i)+=mf[0]; flx2(m,IM2,k,j,i)+=mf[1];
        flx2(m,IM3,k,j,i)+=mf[2]; flx2(m,IEN,k,j,i)+=ef;
      }
    });
  }

  if (three_d) {
    par_for("FOFC-SRRMHD-x3", DevExeSpace(), 0, nmb-1, ks, ke+1, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (!(fofc_(m,k-1,j,i) || fofc_(m,k,j,i))) return;
      srrmhd::SRRMHDPrim1D wl, wr;
      wl.d=w0_(m,IDN,k-1,j,i); wl.vx=w0_(m,IVZ,k-1,j,i);
      wl.vy=w0_(m,IVX,k-1,j,i); wl.vz=w0_(m,IVY,k-1,j,i);
      wl.e=w0_(m,IEN,k-1,j,i); wl.ex=w0_(m,srrmhd::IRE3,k-1,j,i);
      wl.ey=w0_(m,srrmhd::IRE1,k-1,j,i); wl.ez=w0_(m,srrmhd::IRE2,k-1,j,i);
      wl.bx=b0_.x3f(m,k,j,i); wl.by=bcc0_(m,IBX,k-1,j,i);
      wl.bz=bcc0_(m,IBY,k-1,j,i);
      wr.d=w0_(m,IDN,k,j,i); wr.vx=w0_(m,IVZ,k,j,i);
      wr.vy=w0_(m,IVX,k,j,i); wr.vz=w0_(m,IVY,k,j,i);
      wr.e=w0_(m,IEN,k,j,i); wr.ex=w0_(m,srrmhd::IRE3,k,j,i);
      wr.ey=w0_(m,srrmhd::IRE1,k,j,i); wr.ez=w0_(m,srrmhd::IRE2,k,j,i);
      wr.bx=b0_.x3f(m,k,j,i); wr.by=bcc0_(m,IBX,k,j,i);
      wr.bz=bcc0_(m,IBY,k,j,i);
      srrmhd::SRRMHDCons1D flux;
      SingleStateLLF_SRRMHD(wl,wr,eos,flux);
      flx3(m,IDN,k,j,i)=flux.d; flx3(m,IM3,k,j,i)=flux.mx;
      flx3(m,IM1,k,j,i)=flux.my; flx3(m,IM2,k,j,i)=flux.mz;
      flx3(m,IEN,k,j,i)=flux.e; flx3(m,srrmhd::IRE3,k,j,i)=flux.ex;
      flx3(m,srrmhd::IRE1,k,j,i)=flux.ey; flx3(m,srrmhd::IRE2,k,j,i)=flux.ez;
      e2x3_(m,k,j,i)=-flux.by; e1x3_(m,k,j,i)=flux.bz;
      if (viscosity) {
        srrmhd::ShearStress pl, pr;
        pl.p11=visc_w0_(m,srrmhd::IVP11,k-1,j,i);
        pl.p22=visc_w0_(m,srrmhd::IVP22,k-1,j,i);
        pl.p33=visc_w0_(m,srrmhd::IVP33,k-1,j,i);
        pl.p12=visc_w0_(m,srrmhd::IVP12,k-1,j,i);
        pl.p13=visc_w0_(m,srrmhd::IVP13,k-1,j,i);
        pl.p23=visc_w0_(m,srrmhd::IVP23,k-1,j,i);
        pr.p11=visc_w0_(m,srrmhd::IVP11,k,j,i);
        pr.p22=visc_w0_(m,srrmhd::IVP22,k,j,i);
        pr.p33=visc_w0_(m,srrmhd::IVP33,k,j,i);
        pr.p12=visc_w0_(m,srrmhd::IVP12,k,j,i);
        pr.p13=visc_w0_(m,srrmhd::IVP13,k,j,i);
        pr.p23=visc_w0_(m,srrmhd::IVP23,k,j,i);
        Real sf[srrmhd::NVISC], mf[3], ef;
        srrmhd::ViscousLLFContributions(2,wl.d,wl.vy,wl.vz,wl.vx,pl,
            wr.d,wr.vy,wr.vz,wr.vx,pr,sf,mf,ef);
        for (int n=0; n<srrmhd::NVISC; ++n) visc_flx3(m,n,k,j,i)=sf[n];
        flx3(m,IM1,k,j,i)+=mf[0]; flx3(m,IM2,k,j,i)+=mf[1];
        flx3(m,IM3,k,j,i)+=mf[2]; flx3(m,IEN,k,j,i)+=ef;
      }
    });
  }

  Kokkos::deep_copy(fofc, false);
}

} // namespace mhd
