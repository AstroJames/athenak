#ifndef MHD_RSOLVERS_LLF_SRRMHD_SINGLESTATE_HPP_
#define MHD_RSOLVERS_LLF_SRRMHD_SINGLESTATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srrmhd_singlestate.hpp
//! \brief Physical and LLF fluxes for full resistive special-relativistic MHD.

#include "athena.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void SingleStateFlux_SRRMHD()
//! \brief Compute conserved variables and the physical x-flux for one SRRMHD state.
//!
//! The input components are in a local cyclic basis: x is normal to the face and y,z
//! are tangential.  The electric field is evolved independently.  Conductive and
//! advective currents are source terms and are therefore absent from this flux.

KOKKOS_INLINE_FUNCTION
void SingleStateFlux_SRRMHD(const srrmhd::SRRMHDPrim1D &w, const EOS_Data &eos,
                            srrmhd::SRRMHDCons1D &u, srrmhd::SRRMHDCons1D &flux) {
  srrmhd::SingleP2C_IdealSRRMHD(w, eos.gamma, u);

  Real pgas = eos.IdealGasPressure(w.e);
  Real wgas = w.d + eos.gamma*w.e;
  Real em_pressure = 0.5*(SQR(w.ex) + SQR(w.ey) + SQR(w.ez)
                          + SQR(w.bx) + SQR(w.by) + SQR(w.bz));

  flux.d = w.d*w.vx;
  flux.mx = wgas*SQR(w.vx) + pgas + em_pressure - SQR(w.ex) - SQR(w.bx);
  flux.my = wgas*w.vx*w.vy - w.ex*w.ey - w.bx*w.by;
  flux.mz = wgas*w.vx*w.vz - w.ex*w.ez - w.bx*w.bz;

  // Total energy flux is total momentum.  AthenaK evolves tau=U-D.
  flux.e = u.mx - flux.d;

  // Ampere fluxes: d_t E - curl(B) = -J.
  flux.ex = 0.0;
  flux.ey = w.bz;
  flux.ez = -w.by;

  // Faraday fluxes: d_t B + curl(E) = 0.
  flux.bx = 0.0;
  flux.by = -w.ez;
  flux.bz = w.ey;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_SRRMHD()
//! \brief Compute the local Lax-Friedrichs flux with light-speed signal bounds.

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_SRRMHD(const srrmhd::SRRMHDPrim1D &wl,
                           const srrmhd::SRRMHDPrim1D &wr, const EOS_Data &eos,
                           srrmhd::SRRMHDCons1D &flux) {
  srrmhd::SRRMHDCons1D ul, ur, fl, fr;
  SingleStateFlux_SRRMHD(wl, eos, ul, fl);
  SingleStateFlux_SRRMHD(wr, eos, ur, fr);

  // The full Maxwell-fluid system contains light waves.  A unit signal speed is a safe
  // flat-spacetime bound for every component and avoids ideal-MHD characteristic solves.
  constexpr Real lambda = 1.0;
  flux.d = 0.5*(fl.d + fr.d - lambda*(ur.d - ul.d));
  flux.mx = 0.5*(fl.mx + fr.mx - lambda*(ur.mx - ul.mx));
  flux.my = 0.5*(fl.my + fr.my - lambda*(ur.my - ul.my));
  flux.mz = 0.5*(fl.mz + fr.mz - lambda*(ur.mz - ul.mz));
  flux.e = 0.5*(fl.e + fr.e - lambda*(ur.e - ul.e));
  flux.ex = 0.5*(fl.ex + fr.ex - lambda*(ur.ex - ul.ex));
  flux.ey = 0.5*(fl.ey + fr.ey - lambda*(ur.ey - ul.ey));
  flux.ez = 0.5*(fl.ez + fr.ez - lambda*(ur.ez - ul.ez));
  flux.bx = 0.0;
  flux.by = 0.5*(fl.by + fr.by - lambda*(ur.by - ul.by));
  flux.bz = 0.5*(fl.bz + fr.bz - lambda*(ur.bz - ul.bz));
}

} // namespace mhd

#endif // MHD_RSOLVERS_LLF_SRRMHD_SINGLESTATE_HPP_
