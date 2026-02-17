//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sn_drive_box.cpp
//  \brief Problem generator for a uniform periodic hydro box used to test SN driving

#include <iostream>  // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Initialize a uniform hydro state for periodic stochastic SN-driving tests.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sn_drive_box requires a <hydro> block in the input file." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (pmbp->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sn_drive_box is intended for pure hydro tests (no <mhd> block)."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &u0 = pmbp->phydro->u0;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sn_drive_box requires an ideal-gas EOS in <hydro>." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real pgas0 = pin->GetOrAddReal("problem", "pgas0", 1.0);
  Real vx0 = pin->GetOrAddReal("problem", "vx0", 0.0);
  Real vy0 = pin->GetOrAddReal("problem", "vy0", 0.0);
  Real vz0 = pin->GetOrAddReal("problem", "vz0", 0.0);

  if (rho0 <= 0.0 || pgas0 <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sn_drive_box requires rho0 > 0 and pgas0 > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real m1 = rho0*vx0;
  Real m2 = rho0*vy0;
  Real m3 = rho0*vz0;
  Real eint = pgas0/(eos.gamma - 1.0);
  Real ekin = 0.5*rho0*(SQR(vx0) + SQR(vy0) + SQR(vz0));
  Real etot = eint + ekin;

  par_for("pgen_sn_drive_box", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je,
  is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = m1;
    u0(m, IM2, k, j, i) = m2;
    u0(m, IM3, k, j, i) = m3;
    u0(m, IEN, k, j, i) = etot;
  });

  int nscalars = pmbp->phydro->nscalars;
  int nhydro = pmbp->phydro->nhydro;
  if (nscalars > 0) {
    par_for("pgen_sn_drive_box_scalars", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), 0,
    (nscalars - 1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      u0(m, nhydro + n, k, j, i) = 0.0;
    });
  }

  return;
}
