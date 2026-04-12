//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sn_drive_box_mhd.cpp
//  \brief Problem generator for a uniform periodic MHD box used for SN-driving tests.

#include <iostream>  // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Initialize a uniform MHD state with zero magnetic field for SN-driving tests.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sn_drive_box_mhd requires a <mhd> block in the input file."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (pmbp->phydro != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sn_drive_box_mhd is intended for pure MHD tests (no <hydro> block)."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sn_drive_box_mhd requires an ideal-gas EOS in <mhd>." << std::endl;
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
              << "sn_drive_box_mhd requires rho0 > 0 and pgas0 > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real gm1 = eos.gamma - 1.0;
  int nscalars = pmbp->pmhd->nscalars;

  Kokkos::deep_copy(b0.x1f, 0.0);
  Kokkos::deep_copy(b0.x2f, 0.0);
  Kokkos::deep_copy(b0.x3f, 0.0);

  par_for("pgen_sn_drive_box_mhd", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
          js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m, IDN, k, j, i) = rho0;
    w0(m, IVX, k, j, i) = vx0;
    w0(m, IVY, k, j, i) = vy0;
    w0(m, IVZ, k, j, i) = vz0;
    w0(m, IEN, k, j, i) = pgas0/gm1;

    for (int n = 0; n < nscalars; ++n) {
      w0(m, IYF + n, k, j, i) = 0.0;
    }

    bcc0(m, IBX, k, j, i) = 0.0;
    bcc0(m, IBY, k, j, i) = 0.0;
    bcc0(m, IBZ, k, j, i) = 0.0;
  });

  pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);

  return;
}
