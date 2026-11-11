//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file text_shock.cpp
//! \brief Problem generator: a planar hydrodynamic shock sweeping through text written
//! in density. A float32 mask m(x1,x2) in [0,1] (1 inside a letter) covering the whole
//! domain sets rho = rho_amb*(1 + (drat-1)*m) at uniform pressure (v = 0).
//! A shock of Mach number `Mach` (relative to the ambient sound speed) starts at
//! x1 = xshock and is maintained by the inflow boundary at inner_x1 (as in
//! shock_cloud.cpp).
//!
//! Input parameters (block <problem>):
//!  - mask_file : raw float32 little-endian, row-major, x1 fastest, row 0 at x2min
//!  - mask_nx1, mask_nx2 : mask dimensions (may differ from the grid resolution)
//!  - Mach : incident shock Mach number
//!  - drat : letter-to-ambient density ratio
//!  - xshock : initial shock position (default x1min + 0.02*(x1max - x1min))
//! The mask is sampled at cell centres by bilinear interpolation. The run is 2D
//! (nx3 = 1); in 3D the mask is applied independently of x3.
//========================================================================================

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "text_shock requires a <hydro> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Parameters
  const std::string fname = pin->GetString("problem", "mask_file");
  const int mnx1 = pin->GetInteger("problem", "mask_nx1");
  const int mnx2 = pin->GetInteger("problem", "mask_nx2");
  const Real mach = pin->GetReal("problem", "Mach");
  const Real drat = pin->GetReal("problem", "drat");
  auto &msize = pmy_mesh_->mesh_size;
  const Real gx1min = msize.x1min, gx1max = msize.x1max;
  const Real gx2min = msize.x2min, gx2max = msize.x2max;
  const Real xshock = pin->GetOrAddReal("problem", "xshock",
                                        gx1min + 0.02*(gx1max - gx1min));

  // Read the mask on the host (every rank reads the small file), then copy to device
  const std::size_t nmask = static_cast<std::size_t>(mnx1)*static_cast<std::size_t>(mnx2);
  DvceArray1D<float> mask("text_mask", nmask);
  auto mask_h = Kokkos::create_mirror_view(mask);
  {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "cannot open mask_file '" << fname << "'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    in.read(reinterpret_cast<char*>(mask_h.data()), nmask*sizeof(float));
    if (static_cast<std::size_t>(in.gcount()) != nmask*sizeof(float)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "mask_file '" << fname
                << "' has fewer than mask_nx1*mask_nx2 floats" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  Kokkos::deep_copy(mask, mask_h);

  // Ambient (pre-shock, "R") state and Rankine-Hugoniot post-shock ("L") state
  const Real gm = pmbp->phydro->peos->eos_data.gamma;
  const Real gm1 = gm - 1.0;
  const Real dr = 1.0, pr = 1.0/gm, ur = 0.0;   // ambient sound speed = 1
  const Real jump1 = (gm + 1.0)/(gm1 + 2.0/(mach*mach));
  const Real jump2 = (2.0*gm*mach*mach - gm1)/(gm + 1.0);
  const Real jump3 = 2.0*(1.0 - 1.0/(mach*mach))/(gm + 1.0);
  const Real dl = dr*jump1;
  const Real pl = pr*jump2;
  const Real ul = ur + jump3*mach*std::sqrt(gm*pr/dr);

  // Inflow state at inner_x1 (requires ix1_bc = inflow)
  auto &u_in = pmbp->phydro->pbval_u->u_in;
  u_in.h_view(IDN, BoundaryFace::inner_x1) = dl;
  u_in.h_view(IM1, BoundaryFace::inner_x1) = dl*ul;
  u_in.h_view(IM2, BoundaryFace::inner_x1) = 0.0;
  u_in.h_view(IM3, BoundaryFace::inner_x1) = 0.0;
  u_in.h_view(IEN, BoundaryFace::inner_x1) = pl/gm1 + 0.5*dl*ul*ul;
  u_in.template modify<HostMemSpace>();
  u_in.template sync<DevExeSpace>();

  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  auto &u0 = pmbp->phydro->u0;

  par_for("pgen_text_shock", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);

    if (x1v < xshock) {  // post-shock inflow region
      u0(m,IDN,k,j,i) = dl;
      u0(m,IM1,k,j,i) = dl*ul;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = pl/gm1 + 0.5*dl*ul*ul;
      return;
    }

    // Bilinear sample of the mask at the cell centre (mask pixel centres at (p+1/2)/n)
    Real s1 = (x1v - gx1min)/(gx1max - gx1min)*mnx1 - 0.5;
    Real s2 = (x2v - gx2min)/(gx2max - gx2min)*mnx2 - 0.5;
    s1 = fmin(fmax(s1, 0.0), static_cast<Real>(mnx1 - 1));
    s2 = fmin(fmax(s2, 0.0), static_cast<Real>(mnx2 - 1));
    const int p1 = static_cast<int>(s1), p2 = static_cast<int>(s2);
    const int q1 = (p1 + 1 < mnx1) ? p1 + 1 : p1;
    const int q2 = (p2 + 1 < mnx2) ? p2 + 1 : p2;
    const Real f1 = s1 - p1, f2 = s2 - p2;
    auto at = [&](int a, int b) {
      return static_cast<Real>(mask(static_cast<std::size_t>(b)*mnx1 + a));
    };
    Real mval = (1.0-f1)*(1.0-f2)*at(p1,p2) + f1*(1.0-f2)*at(q1,p2)
              + (1.0-f1)*f2*at(p1,q2) + f1*f2*at(q1,q2);
    mval = fmin(fmax(mval, 0.0), 1.0);

    const Real rho = dr*(1.0 + (drat - 1.0)*mval);
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = pr/gm1;   // uniform pressure: letters start in pressure equilibrium
  });

  return;
}
