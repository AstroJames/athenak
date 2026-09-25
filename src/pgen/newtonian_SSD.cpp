//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file newtonian_SSD.cpp
//! \brief Isothermal turbulence box with a Gaussian spectral magnetic seed.

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

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->pmhd == nullptr || pack->phydro != nullptr || pack->pionn != nullptr ||
      pack->pmhd->peos->eos_data.is_ideal ||
      pack->pcoord->is_special_relativistic || pack->pcoord->is_general_relativistic ||
      pack->pcoord->is_dynamical_relativistic) {
    std::cerr << "newtonian_SSD requires Newtonian isothermal MHD.\n";
    std::exit(EXIT_FAILURE);
  }
  Real cs = pack->pmhd->peos->eos_data.iso_cs;
  if (!(cs > 0.0) || !std::isfinite(cs)) {
    std::cerr << "newtonian_SSD requires a positive finite iso_sound_speed.\n";
    std::exit(EXIT_FAILURE);
  }
  bool seed_on_restart = pin->GetOrAddBoolean("spectral_ic", "seed_on_restart", false);
  // Ordinary restarts preserve the evolved field, irrespective of the seed inputs.
  if (restart && !seed_on_restart) return;
  if (seed_on_restart && !restart) {
    std::cerr << "Invalid spectral seed: seed_on_restart requires a restart file.\n";
    std::exit(EXIT_FAILURE);
  }
  Real rms = pin->GetOrAddReal("spectral_ic", "rms_b", 0.01);
  if (!(rms >= 0.0) || !std::isfinite(rms) || (seed_on_restart && rms == 0.0)) {
    std::cerr << "Invalid spectral seed: require finite nonnegative rms_b "
                 "(positive for seed_on_restart).\n";
    std::exit(EXIT_FAILURE);
  }

  auto &b = pack->pmhd->b0;
  auto &idx = pmy_mesh_->mb_indcs;
  if (seed_on_restart) {
    // Inspect individual active faces, including upper boundaries: averaging can
    // hide a nonzero checkerboard field. NaNs also fail this exact-zero check.
    int nx = idx.nx1, ny = idx.nx2, nz = idx.nx3;
    int is = idx.is, js = idx.js, ks = idx.ks;
    int nonzero = 0;
    Kokkos::parallel_reduce("ssd_check_unmagnetized",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, pack->nmb_thispack*nx*ny*nz),
      KOKKOS_LAMBDA(const int q, int &found) {
        int i = q%nx+is, j = (q/nx)%ny+js, k = (q/(nx*ny))%nz+ks;
        int m = q/(nx*ny*nz);
        int occupied = (b.x1f(m,k,j,i) != 0.0 || b.x1f(m,k,j,i+1) != 0.0 ||
                        b.x2f(m,k,j,i) != 0.0 || b.x2f(m,k,j+1,i) != 0.0 ||
                        b.x3f(m,k,j,i) != 0.0 || b.x3f(m,k+1,j,i) != 0.0);
        if (occupied > found) found = occupied;
      }, Kokkos::Max<int>(nonzero));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &nonzero, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (nonzero > 0) {
      std::cerr << "Invalid spectral seed: seed_on_restart requires an exactly "
                   "zero magnetic field; refusing to overwrite the checkpoint field.\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if (rms == 0.0) {
    // Spin up using MHD with B=0 so its checkpoint already has the MHD layout.
    Kokkos::deep_copy(b.x1f, 0.0);
    Kokkos::deep_copy(b.x2f, 0.0);
    Kokkos::deep_copy(b.x3f, 0.0);
  } else {
    SpectralICGenerator generator(pack, pin);
    int n1 = idx.nx1+2*idx.ng, n2 = idx.nx2+2*idx.ng, n3 = idx.nx3+2*idx.ng;
    DvceArray4D<Real> ax("ssd_ax", pack->nmb_thispack, n3, n2, n1);
    DvceArray4D<Real> ay("ssd_ay", pack->nmb_thispack, n3, n2, n1);
    DvceArray4D<Real> az("ssd_az", pack->nmb_thispack, n3, n2, n1);
    generator.GenerateVectorPotentialFFT(ax, ay, az);
    auto size = pack->pmb->mb_size;
    // Discrete edge-to-face curl preserves the constrained-transport divergence.
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
  }

  auto u = pack->pmhd->u0;
  auto bcc = pack->pmhd->bcc0;
  int nscalars = pack->pmhd->nscalars, nmhd = pack->pmhd->nmhd;
  par_for("ssd_seed_fluid", DevExeSpace(), 0, pack->nmb_thispack-1,
          idx.ks, idx.ke, idx.js, idx.je, idx.is, idx.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc(m,IBX,k,j,i) = 0.5*(b.x1f(m,k,j,i)+b.x1f(m,k,j,i+1));
    bcc(m,IBY,k,j,i) = 0.5*(b.x2f(m,k,j,i)+b.x2f(m,k,j+1,i));
    bcc(m,IBZ,k,j,i) = 0.5*(b.x3f(m,k,j,i)+b.x3f(m,k+1,j,i));
    if (!restart) {
      // Match turb.cpp's single-fluid MHD box: rho=1, v=0, p=rho*cs^2.
      // Isothermal MHD has no energy variable to initialize.
      u(m,IDN,k,j,i) = 1.0;
      u(m,IM1,k,j,i) = 0.0;
      u(m,IM2,k,j,i) = 0.0;
      u(m,IM3,k,j,i) = 0.0;
      for (int n = 0; n < nscalars; ++n) u(m,nmhd+n,k,j,i) = 0.0;
    }
  });
  // Subsequent checkpoints restart normally, without reinjecting the seed.
  if (seed_on_restart) pin->SetBoolean("spectral_ic", "seed_on_restart", false);
}
