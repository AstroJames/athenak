//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectrum_modes.cpp
//! \brief Problem generator for power spectrum regression testing.
//!
//! Initializes a superposition of sinusoidal velocity modes along x1:
//!   v_x(x) = sum_{k=1}^{nmode} (1/k) * sin(2*pi*k*(x - x1min)/L)
//!   v_y = v_z = 0,  rho = 1,  p = 1
//!
//! For this field the analytical velocity power spectrum is:
//!   P(k) = 1 / (2 * k^2)   for k = 1 ... nmode
//!   P(k) = 0               otherwise
//!
//! This result holds to machine precision and can be used to verify both
//! the bin assignments and the normalization of the power spectrum output.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

void ProblemGenerator::SpectrumModes(ParameterInput *pin, const bool restart) {
  if (restart) return;

  int nmode = pin->GetOrAddInteger("problem", "nmode", 4);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << "\n"
              << "spectrum_modes pgen requires <hydro> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  Real p0 = 1.0;
  Real rho0 = 1.0;

  Real x1min = pmy_mesh_->mesh_size.x1min;
  Real x1max = pmy_mesh_->mesh_size.x1max;
  Real L = x1max - x1min;

  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;

  par_for("pgen_spectrum_modes", DevExeSpace(),
          0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1bmin = size.d_view(m).x1min;
    Real &x1bmax = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real xv = CellCenterX(i-is, nx1, x1bmin, x1bmax);

    // Sum of sinusoidal modes: v_x = sum_{n=1}^{nmode} (1/n)*sin(2*pi*n*(x-x1min)/L)
    Real vx = 0.0;
    for (int n = 1; n <= nmode; ++n) {
      vx += (1.0/n) * std::sin(2.0*M_PI*n*(xv - x1min)/L);
    }

    Real ekin = 0.5 * rho0 * vx * vx;

    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = rho0 * vx;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    if (eos.is_ideal) {
      u0(m, IEN, k, j, i) = p0/gm1 + ekin;
    }
  });
}
