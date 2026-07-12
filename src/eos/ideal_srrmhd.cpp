//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_srrmhd.cpp
//! \brief Ideal-gas EOS and known-E conversion kernels for resistive SRMHD.

#include <float.h>

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/resistive_srmhd.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealSRRMHD::IdealSRRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd", "gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("mhd", "gamma_max", (FLT_MAX));
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Convert total resistive-SRMHD conserved variables to primitives for known E.

void IdealSRRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                             DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                             const bool only_testfloors,
                             const int il, const int iu, const int jl, const int ju,
                             const int kl, const int ku) {
  int &nmhd = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto eos = eos_data;
  auto &fofc_ = pmy_pack->pmhd->fofc;

  const int ni = iu - il + 1;
  const int nji = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord = 0, nfloore = 0, nceilv = 0, nfail = 0, maxit = 0;
  Kokkos::parallel_reduce("srrmhd_c2p", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumv, int &sumf,
                int &max_it) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    srrmhd::SRRMHDCons1D u;
    u.d = cons(m, IDN, k, j, i);
    u.mx = cons(m, IM1, k, j, i);
    u.my = cons(m, IM2, k, j, i);
    u.mz = cons(m, IM3, k, j, i);
    u.e = cons(m, IEN, k, j, i);
    u.ex = cons(m, srrmhd::IRE1, k, j, i);
    u.ey = cons(m, srrmhd::IRE2, k, j, i);
    u.ez = cons(m, srrmhd::IRE3, k, j, i);

    if (only_testfloors) {
      u.bx = bcc(m, IBX, k, j, i);
      u.by = bcc(m, IBY, k, j, i);
      u.bz = bcc(m, IBZ, k, j, i);
    } else {
      u.bx = 0.5*(b.x1f(m, k, j, i) + b.x1f(m, k, j, i+1));
      u.by = 0.5*(b.x2f(m, k, j, i) + b.x2f(m, k, j+1, i));
      u.bz = 0.5*(b.x3f(m, k, j, i) + b.x3f(m, k+1, j, i));
    }

    srrmhd::SRRMHDPrim1D w;
    bool dfloor_used = false, efloor_used = false;
    bool vceiling_used = false, c2p_failure = false;
    int iter_used = 0;
    srrmhd::SingleC2P_IdealSRRMHDKnownE(u, eos, w, dfloor_used, efloor_used,
                                        c2p_failure, iter_used);

    Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
    if (lor > eos.gamma_max) {
      vceiling_used = true;
      Real factor = sqrt((SQR(eos.gamma_max) - 1.0)/(SQR(lor) - 1.0));
      w.vx *= factor;
      w.vy *= factor;
      w.vz *= factor;
    }

    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        fofc_(m, k, j, i) = true;
        sumd++;
      }
    } else {
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (vceiling_used) {sumv++;}
      if (c2p_failure) {sumf++;}
      max_it = (iter_used > max_it) ? iter_used : max_it;

      prim(m, IDN, k, j, i) = w.d;
      prim(m, IVX, k, j, i) = w.vx;
      prim(m, IVY, k, j, i) = w.vy;
      prim(m, IVZ, k, j, i) = w.vz;
      prim(m, IEN, k, j, i) = w.e;
      prim(m, srrmhd::IRE1, k, j, i) = w.ex;
      prim(m, srrmhd::IRE2, k, j, i) = w.ey;
      prim(m, srrmhd::IRE3, k, j, i) = w.ez;

      bcc(m, IBX, k, j, i) = u.bx;
      bcc(m, IBY, k, j, i) = u.by;
      bcc(m, IBZ, k, j, i) = u.bz;

      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        srrmhd::SRRMHDCons1D uout;
        srrmhd::SingleP2C_IdealSRRMHD(w, eos.gamma, uout);
        cons(m, IDN, k, j, i) = uout.d;
        cons(m, IM1, k, j, i) = uout.mx;
        cons(m, IM2, k, j, i) = uout.my;
        cons(m, IM3, k, j, i) = uout.mz;
        cons(m, IEN, k, j, i) = uout.e;
        cons(m, srrmhd::IRE1, k, j, i) = uout.ex;
        cons(m, srrmhd::IRE2, k, j, i) = uout.ey;
        cons(m, srrmhd::IRE3, k, j, i) = uout.ez;
        u.d = uout.d;
      }

      for (int n = nmhd; n < (nmhd+nscal); ++n) {
        prim(m, n, k, j, i) = cons(m, n, k, j, i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord), Kokkos::Sum<int>(nfloore), Kokkos::Sum<int>(nceilv),
     Kokkos::Sum<int>(nfail), Kokkos::Max<int>(maxit));

  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore;
    pmy_pack->pmesh->ecounter.neos_vceil += nceilv;
    pmy_pack->pmesh->ecounter.neos_fail += nfail;
    pmy_pack->pmesh->ecounter.maxit_c2p = maxit;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Convert resistive-SRMHD primitives to total conserved variables.

void IdealSRRMHD::PrimToCons(const DvceArray5D<Real> &prim,
                             const DvceArray5D<Real> &bcc,
                             DvceArray5D<Real> &cons, const int il, const int iu,
                             const int jl, const int ju, const int kl, const int ku) {
  int &nmhd = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;

  par_for("srrmhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    srrmhd::SRRMHDPrim1D w;
    w.d = prim(m, IDN, k, j, i);
    w.vx = prim(m, IVX, k, j, i);
    w.vy = prim(m, IVY, k, j, i);
    w.vz = prim(m, IVZ, k, j, i);
    w.e = prim(m, IEN, k, j, i);
    w.ex = prim(m, srrmhd::IRE1, k, j, i);
    w.ey = prim(m, srrmhd::IRE2, k, j, i);
    w.ez = prim(m, srrmhd::IRE3, k, j, i);
    w.bx = bcc(m, IBX, k, j, i);
    w.by = bcc(m, IBY, k, j, i);
    w.bz = bcc(m, IBZ, k, j, i);

    srrmhd::SRRMHDCons1D u;
    srrmhd::SingleP2C_IdealSRRMHD(w, gamma, u);

    cons(m, IDN, k, j, i) = u.d;
    cons(m, IM1, k, j, i) = u.mx;
    cons(m, IM2, k, j, i) = u.my;
    cons(m, IM3, k, j, i) = u.mz;
    cons(m, IEN, k, j, i) = u.e;
    cons(m, srrmhd::IRE1, k, j, i) = u.ex;
    cons(m, srrmhd::IRE2, k, j, i) = u.ey;
    cons(m, srrmhd::IRE3, k, j, i) = u.ez;

    for (int n = nmhd; n < (nmhd+nscal); ++n) {
      cons(m, n, k, j, i) = u.d*prim(m, n, k, j, i);
    }
  });
}
