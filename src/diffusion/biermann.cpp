//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file biermann.cpp
//  \brief Implements functions for BiermannBattery class.

#include <cmath>
#include <iostream>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "units/units.hpp"
#include "biermann.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
Real GasPressure(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                 const int m, const int k, const int j, const int i) {
  if (eos.is_ideal) {
    return (eos.gamma - 1.0) * w0(m, IEN, k, j, i);
  }
  return SQR(eos.iso_cs) * w0(m, IDN, k, j, i);
}

KOKKOS_INLINE_FUNCTION
void BiermannCellE(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                   const RegionSize &size, const Real coeff,
                   const bool multi_d, const bool three_d,
                   const int m, const int k, const int j, const int i,
                   Real &e1, Real &e2, Real &e3) {
  Real dpdx1 = (GasPressure(w0, eos, m, k, j, i+1) - GasPressure(w0, eos, m, k, j, i-1))
               /(2.0*size.dx1);
  Real dpdx2 = 0.0;
  Real dpdx3 = 0.0;
  if (multi_d) {
    dpdx2 = (GasPressure(w0, eos, m, k, j+1, i) - GasPressure(w0, eos, m, k, j-1, i))
            /(2.0*size.dx2);
  }
  if (three_d) {
    dpdx3 = (GasPressure(w0, eos, m, k+1, j, i) - GasPressure(w0, eos, m, k-1, j, i))
            /(2.0*size.dx3);
  }

  Real rho = fmax(w0(m, IDN, k, j, i), eos.dfloor);
  e1 = -coeff*dpdx1/rho;
  e2 = -coeff*dpdx2/rho;
  e3 = -coeff*dpdx3/rho;
}
} // namespace

//----------------------------------------------------------------------------------------
// ctor

BiermannBattery::BiermannBattery(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp) {
  coeff = pin->GetOrAddReal("mhd", "biermann_coeff", 0.0);
  coeff_from_closure = pin->GetOrAddBoolean("mhd", "biermann_from_cgs", false);
  pe_fraction = pin->GetOrAddReal("mhd", "biermann_pe_fraction", 1.0);
  mu_e = pin->GetOrAddReal("mhd", "biermann_mu_e", 1.0);

  if (coeff_from_closure) {
    if (pmy_pack->punit == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Biermann closure conversion requires a <units> block"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    constexpr Real four_pi = 12.56637061435917295385;
    Real closure_cgs = units::Units::speed_of_light_cgs*pe_fraction*mu_e
                     * units::Units::atomic_mass_unit_cgs
                     / units::Units::elementary_charge_cgs;
    Real conv = pmy_pack->punit->length_cgs()
              * std::sqrt(four_pi*pmy_pack->punit->density_cgs());
    coeff = closure_cgs/conv;
  }

  enabled = (coeff > 0.0);
  if (!enabled) {
    return;
  }

  if (pmy_pack->pcoord->is_special_relativistic ||
      pmy_pack->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Biermann battery is currently implemented only for"
              << " Newtonian MHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// destructor

BiermannBattery::~BiermannBattery() {
}

//----------------------------------------------------------------------------------------
//! \fn BiermannEField()
//  \brief Adds Biermann battery electric field to edge-centered electric fields
//  The term is implemented as:
//    E_batt = - C_batt * (grad(p_gas) / rho)

void BiermannBattery::BiermannEField(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                                     DvceEdgeFld4D<Real> &efld) {
  if (!enabled) {
    return;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto w0_ = w0;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto eos_ = eos;
  Real coeff_ = coeff;

  if (pmy_pack->pmesh->one_d) {
    return;
  }

  if (pmy_pack->pmesh->two_d) {
    par_for("biermann_e1_2d", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      Real e1c0, e2c0, e3c0;
      Real e1c1, e2c1, e3c1;
      BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, ks, j, i,
                    e1c0, e2c0, e3c0);
      BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, ks, j-1, i,
                    e1c1, e2c1, e3c1);
      Real e1edge = 0.5*(e1c0 + e1c1);
      e1(m,ks,  j,i) += e1edge;
      e1(m,ke+1,j,i) += e1edge;
    });

    par_for("biermann_e2_2d", DevExeSpace(), 0, nmb1, js, je, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      Real e1c0, e2c0, e3c0;
      Real e1c1, e2c1, e3c1;
      BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, ks, j, i,
                    e1c0, e2c0, e3c0);
      BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, ks, j, i-1,
                    e1c1, e2c1, e3c1);
      Real e2edge = 0.5*(e2c0 + e2c1);
      e2(m,ks,  j,i) += e2edge;
      e2(m,ke+1,j,i) += e2edge;
    });
    return;
  }

  par_for("biermann_e1_3d", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real e1c00, e2tmp, e3tmp;
    Real e1c10, e1c01, e1c11;
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k,   j,   i,
                  e1c00, e2tmp, e3tmp);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k-1, j,   i,
                  e1c10, e2tmp, e3tmp);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k,   j-1, i,
                  e1c01, e2tmp, e3tmp);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k-1, j-1, i,
                  e1c11, e2tmp, e3tmp);
    e1(m,k,j,i) += 0.25*(e1c00 + e1c10 + e1c01 + e1c11);
  });

  par_for("biermann_e2_3d", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real e2c00, e1tmp, e3tmp;
    Real e2c10, e2c01, e2c11;
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k,   j, i,
                  e1tmp, e2c00, e3tmp);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k-1, j, i,
                  e1tmp, e2c10, e3tmp);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k,   j, i-1,
                  e1tmp, e2c01, e3tmp);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k-1, j, i-1,
                  e1tmp, e2c11, e3tmp);
    e2(m,k,j,i) += 0.25*(e2c00 + e2c10 + e2c01 + e2c11);
  });

  par_for("biermann_e3_3d", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real e3c00, e1tmp, e2tmp;
    Real e3c10, e3c01, e3c11;
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j,   i,
                  e1tmp, e2tmp, e3c00);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j-1, i,
                  e1tmp, e2tmp, e3c10);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j,   i-1,
                  e1tmp, e2tmp, e3c01);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j-1, i-1,
                  e1tmp, e2tmp, e3c11);
    e3(m,k,j,i) += 0.25*(e3c00 + e3c10 + e3c01 + e3c11);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn BiermannEnergyFlux()
//! \brief Adds Biermann contribution to total-energy flux:
//!        F_E += E_batt x B, where E_batt = -C_batt * grad(P)/rho

void BiermannBattery::BiermannEnergyFlux(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                                         const DvceFaceFld4D<Real> &b,
                                         DvceFaceFld5D<Real> &flx) {
  if (!enabled) {
    return;
  }
  if (pmy_pack->pmesh->one_d) {
    return;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto w0_ = w0;
  auto b_ = b;
  auto eos_ = eos;
  auto flx1 = flx.x1f;
  auto flx2 = flx.x2f;
  auto flx3 = flx.x3f;
  Real coeff_ = coeff;

  //------------------------------
  // energy fluxes in x1-direction
  par_for("biermann_heat1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real e1l, e2l, e3l;
    Real e1r, e2r, e3r;
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j, i-1,
                  e1l, e2l, e3l);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j, i,
                  e1r, e2r, e3r);
    Real e2f = 0.5*(e2l + e2r);
    Real e3f = 0.5*(e3l + e3r);

    Real b2l = 0.5*(b_.x2f(m,k,j  ,i-1) + b_.x2f(m,k,j+1,i-1));
    Real b2r = 0.5*(b_.x2f(m,k,j  ,i  ) + b_.x2f(m,k,j+1,i  ));
    Real b3l = 0.5*(b_.x3f(m,k  ,j,i-1) + b_.x3f(m,k+1,j,i-1));
    Real b3r = 0.5*(b_.x3f(m,k  ,j,i  ) + b_.x3f(m,k+1,j,i  ));
    Real b2f = 0.5*(b2l + b2r);
    Real b3f = 0.5*(b3l + b3r);

    flx1(m,IEN,k,j,i) += (e2f*b3f - e3f*b2f);
  });

  //------------------------------
  // energy fluxes in x2-direction
  par_for("biermann_heat2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real e1l, e2l, e3l;
    Real e1r, e2r, e3r;
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j-1, i,
                  e1l, e2l, e3l);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j, i,
                  e1r, e2r, e3r);
    Real e1f = 0.5*(e1l + e1r);
    Real e3f = 0.5*(e3l + e3r);

    Real b1l = 0.5*(b_.x1f(m,k,j-1,i) + b_.x1f(m,k,j-1,i+1));
    Real b1r = 0.5*(b_.x1f(m,k,j  ,i) + b_.x1f(m,k,j  ,i+1));
    Real b3l = 0.5*(b_.x3f(m,k  ,j-1,i) + b_.x3f(m,k+1,j-1,i));
    Real b3r = 0.5*(b_.x3f(m,k  ,j  ,i) + b_.x3f(m,k+1,j  ,i));
    Real b1f = 0.5*(b1l + b1r);
    Real b3f = 0.5*(b3l + b3r);

    flx2(m,IEN,k,j,i) += (e3f*b1f - e1f*b3f);
  });
  if (pmy_pack->pmesh->two_d) {
    return;
  }

  //------------------------------
  // energy fluxes in x3-direction
  par_for("biermann_heat3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real e1l, e2l, e3l;
    Real e1r, e2r, e3r;
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k-1, j, i,
                  e1l, e2l, e3l);
    BiermannCellE(w0_, eos_, mbsize.d_view(m), coeff_, multi_d, three_d, m, k, j, i,
                  e1r, e2r, e3r);
    Real e1f = 0.5*(e1l + e1r);
    Real e2f = 0.5*(e2l + e2r);

    Real b1l = 0.5*(b_.x1f(m,k-1,j,i  ) + b_.x1f(m,k-1,j,i+1));
    Real b1r = 0.5*(b_.x1f(m,k  ,j,i  ) + b_.x1f(m,k  ,j,i+1));
    Real b2l = 0.5*(b_.x2f(m,k-1,j  ,i) + b_.x2f(m,k-1,j+1,i));
    Real b2r = 0.5*(b_.x2f(m,k  ,j  ,i) + b_.x2f(m,k  ,j+1,i));
    Real b1f = 0.5*(b1l + b1r);
    Real b2f = 0.5*(b2l + b2r);

    flx3(m,IEN,k,j,i) += (e1f*b2f - e2f*b1f);
  });

  return;
}
