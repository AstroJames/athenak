//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scale_separated_ssd.cpp
//! \brief Gaussian spectral seed, optionally force-free, for Newtonian dynamos.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "utils/spectral_ic_gen.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->pmhd == nullptr || !pack->pmhd->peos->eos_data.is_ideal ||
      pack->pcoord->is_special_relativistic || pack->pcoord->is_general_relativistic ||
      pack->pcoord->is_dynamical_relativistic) {
    std::cerr << "scale_separated_ssd requires Newtonian ideal-gas MHD.\n";
    std::exit(EXIT_FAILURE);
  }
  // Restart restores the evolved field, never regenerate or reproject it.
  if (restart) return;
  Real rho = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real pressure = pin->GetOrAddReal("problem", "p0", 1.0);
  Real rms = pin->GetOrAddReal("spectral_ic", "rms_b", 0.01);
  if (!(rho > 0.0) || !std::isfinite(rho) || !(pressure > 0.0) ||
      !std::isfinite(pressure) || !(rms > 0.0) || !std::isfinite(rms)) {
    std::cerr << "Invalid spectral seed: require positive finite rho0/p0/rms_b.\n";
    std::exit(EXIT_FAILURE);
  }
  auto &b = pack->pmhd->b0;
  SpectralICGenerator generator(pack, pin);
  auto &idx = pmy_mesh_->mb_indcs;
  int n1 = idx.nx1+2*idx.ng, n2 = idx.nx2+2*idx.ng, n3 = idx.nx3+2*idx.ng;
  DvceArray4D<Real> ax("ssd_ax", pack->nmb_thispack, n3, n2, n1);
  DvceArray4D<Real> ay("ssd_ay", pack->nmb_thispack, n3, n2, n1);
  DvceArray4D<Real> az("ssd_az", pack->nmb_thispack, n3, n2, n1);
  generator.GenerateVectorPotentialFFT(ax, ay, az);
  auto size = pack->pmb->mb_size;
  // Standard edge-to-face discrete curl. Shared edge circulations cancel in div B.
  for (int axis = 0; axis < 3; ++axis) {
    auto field = (axis == 0) ? b.x1f : ((axis == 1) ? b.x2f : b.x3f);
    par_for("ssd_curl_a", DevExeSpace(), 0, pack->nmb_thispack-1,
            idx.ks, idx.ke+(axis == 2), idx.js, idx.je+(axis == 1),
            idx.is, idx.ie+(axis == 0),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx = size.d_view(m).dx1, dy = size.d_view(m).dx2, dz = size.d_view(m).dx3;
      if (axis == 0) {
        field(m,k,j,i) = (az(m,k,j+1,i)-az(m,k,j,i))/dy
                       -(ay(m,k+1,j,i)-ay(m,k,j,i))/dz;
      } else if (axis == 1) {
        field(m,k,j,i) = (ax(m,k+1,j,i)-ax(m,k,j,i))/dz
                       -(az(m,k,j,i+1)-az(m,k,j,i))/dx;
      } else {
        field(m,k,j,i) = (ay(m,k,j,i+1)-ay(m,k,j,i))/dx
                       -(ax(m,k,j+1,i)-ax(m,k,j,i))/dy;
      }
    });
  }
  NormalizeRmsB(pack, b, rms);

  auto u = pack->pmhd->u0;
  auto bcc = pack->pmhd->bcc0;
  Real gm1 = pack->pmhd->peos->eos_data.gamma-1.0;
  int nscalars = pack->pmhd->nscalars, nmhd = pack->pmhd->nmhd;
  par_for("ssd_seed_fluid", DevExeSpace(), 0, pack->nmb_thispack-1,
          idx.ks, idx.ke, idx.js, idx.je, idx.is, idx.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real bx = 0.5*(b.x1f(m,k,j,i)+b.x1f(m,k,j,i+1));
    Real by = 0.5*(b.x2f(m,k,j,i)+b.x2f(m,k,j+1,i));
    Real bz = 0.5*(b.x3f(m,k,j,i)+b.x3f(m,k+1,j,i));
    bcc(m,IBX,k,j,i) = bx;
    bcc(m,IBY,k,j,i) = by;
    bcc(m,IBZ,k,j,i) = bz;
    u(m,IDN,k,j,i) = rho;
    u(m,IM1,k,j,i) = 0.0;
    u(m,IM2,k,j,i) = 0.0;
    u(m,IM3,k,j,i) = 0.0;
    u(m,IEN,k,j,i) = pressure/gm1+0.5*(bx*bx+by*by+bz*bz);
    for (int n = 0; n < nscalars; ++n) u(m,nmhd+n,k,j,i) = 0.0;
  });
}
