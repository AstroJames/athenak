//========================================================================================
// AthenaK: astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coalescing_ft.cpp
//! \brief Problem generator for the coalescing flux-tube setup of Vicentin et al. (2025)
//!
//! Section 3 of the paper defines a 2D visco-resistive, isothermal MHD setup with
//! vector potential
//!   psi = (B0/2pi) tanh(y/delta) cos(pi x) sin(2pi y),
//! in-plane magnetic field B = zhat x grad(psi), a uniform guide field Bz, and density
//! chosen from total-pressure balance with p = rho cs^2.
//========================================================================================

// C++ headers
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include <Kokkos_Random.hpp>

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "utils/spectral_2d_field_gen.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "coalescing_ft requires an <mhd> block." << std::endl;
    exit(EXIT_FAILURE);
  }

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  if (eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "coalescing_ft is implemented for the isothermal MHD setup used in "
              << "Vicentin et al. (2025)." << std::endl;
    exit(EXIT_FAILURE);
  }

  constexpr Real pi = 3.14159265358979323846264338327950288;

  const Real b0_amp = pin->GetOrAddReal("problem", "b0", 1.0);
  const Real bg = pin->GetOrAddReal("problem", "bg", 0.5);
  const Real delta = pin->GetOrAddReal("problem", "a0", 1.0/std::sqrt(1.0e3));
  const Real beta = pin->GetOrAddReal("problem", "beta", 2.0);
  const Real epsv = pin->GetOrAddReal("problem", "epsv", 1.0e-2);
  const std::string perturbation = pin->GetOrAddString("problem", "perturbation", "rn");
  const int rng_seed = pin->GetOrAddInteger("problem", "rng_seed", 42);

  const Real cs = eos.iso_cs;
  const Real cs2 = cs*cs;
  const Real pmag_max = 0.5*(b0_amp*b0_amp + bg*bg);
  const Real p0 = beta*pmag_max;
  const Real ptot = pmag_max + p0;
  const Real dfloor = eos.dfloor;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;

  const int nmb = pmbp->nmb_thispack;
  auto &b0 = pmbp->pmhd->b0;
  par_for("coalescing_ft_b", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;

    Real x1f = LeftEdgeX(i - is, nx1, x1min, x1max);
    Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
    Real x2f = LeftEdgeX(j - js, nx2, x2min, x2max);
    Real x2v = CellCenterX(j - js, nx2, x2min, x2max);

    Real tanh_y = tanh(x2v/delta);
    Real sech2_y = 1.0/SQR(cosh(x2v/delta));
    Real sin_2piy = sin(2.0*pi*x2v);
    Real cos_2piy = cos(2.0*pi*x2v);
    Real dpsi_dy = (b0_amp/(2.0*pi))*cos(pi*x1f)*
                   ((sech2_y/delta)*sin_2piy + 2.0*pi*tanh_y*cos_2piy);

    b0.x1f(m,k,j,i) = -dpsi_dy;
    b0.x2f(m,k,j,i) = -0.5*b0_amp*sin(pi*x1v)*tanh(x2f/delta)*sin(2.0*pi*x2f);
    b0.x3f(m,k,j,i) = bg;

    if (i == ie) {
      Real x1fp1 = LeftEdgeX(i + 1 - is, nx1, x1min, x1max);
      b0.x1f(m,k,j,i + 1) = -(b0_amp/(2.0*pi))*cos(pi*x1fp1)*
                            ((sech2_y/delta)*sin_2piy + 2.0*pi*tanh_y*cos_2piy);
    }
    if (j == je) {
      Real x2fp1 = LeftEdgeX(j + 1 - js, nx2, x2min, x2max);
      b0.x2f(m,k,j + 1,i) = -0.5*b0_amp*sin(pi*x1v)*tanh(x2fp1/delta)*
                            sin(2.0*pi*x2fp1);
    }
    if (k == ke) {
      b0.x3f(m,k + 1,j,i) = bg;
    }
  });

  auto &u0 = pmbp->pmhd->u0;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  DvceArray4D<Real> dvx("coalescing_ft_dvx", nmb, ncells3, ncells2, ncells1);
  DvceArray4D<Real> dvy("coalescing_ft_dvy", nmb, ncells3, ncells2, ncells1);
  DvceArray4D<Real> dvz("coalescing_ft_dvz", nmb, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(dvx, 0.0);
  Kokkos::deep_copy(dvy, 0.0);
  Kokkos::deep_copy(dvz, 0.0);

  if (epsv > 0.0) {
    if (perturbation.compare("rn") == 0) {
      Kokkos::Random_XorShift64_Pool<> rand_pool64(static_cast<uint64_t>(rng_seed));
      par_for("coalescing_ft_rn", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto rand_gen = rand_pool64.get_state();
        Real u1 = fmax(static_cast<Real>(rand_gen.frand()), 1.0e-12);
        Real u2 = static_cast<Real>(rand_gen.frand());
        Real amp = sqrt(-2.0*log(u1));
        dvx(m,k,j,i) = amp*cos(2.0*pi*u2);
        dvy(m,k,j,i) = amp*sin(2.0*pi*u2);
        rand_pool64.free_state(rand_gen);
      });
    } else if (perturbation.compare("mm") == 0) {
      Spectral2DFieldGenerator gen(pmbp, pin);
      gen.GenerateCurlField(dvx, dvy);
      if (global_variable::my_rank == 0) {
        std::cout << "coalescing_ft: initialized spectral MM perturbation with "
                  << gen.mode_count << " modes" << std::endl;
      }
    } else if (perturbation.compare("none") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "problem/perturbation must be one of: none, rn, mm." << std::endl;
      exit(EXIT_FAILURE);
    }

    RemoveVectorFieldMean(pmbp, dvx, dvy, dvz);
    NormalizeVectorFieldRms(pmbp, dvx, dvy, dvz, epsv);
  }

  par_for("coalescing_ft_u", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i + 1));
    Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j + 1,i));
    Real bz = bg;
    Real b2 = bx*bx + by*by + bz*bz;

    Real dens = (ptot - 0.5*b2)/cs2;
    dens = fmax(dens, dfloor);

    u0(m,IDN,k,j,i) = dens;
    u0(m,IM1,k,j,i) = dens*dvx(m,k,j,i);
    u0(m,IM2,k,j,i) = dens*dvy(m,k,j,i);
    u0(m,IM3,k,j,i) = dens*dvz(m,k,j,i);
  });
}
