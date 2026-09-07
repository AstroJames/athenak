#ifndef TST_UNIT_FOFC_CFL_HPP_
#define TST_UNIT_FOFC_CFL_HPP_
// Copyright (C) 2026 James Beattie and the Athena code team.
// Licensed under the 3-clause BSD License (the "LICENSE").

#include <cmath>
#include <cstdio>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"

// Admissible saved RK3 stage states, one density pulse on a block/rank corner,
// zero velocity/B, and uniform sound speed 1. The high-order test flux merely
// requests correction; the final center update depends only on production LLF.
inline int TestLLFCFL(Mesh &mesh, Driver &driver, bool safe, int stage) {
  auto pack = mesh.pmb_pack;
  auto pmhd = pack->pmhd;
  const auto &ind = mesh.mb_indcs;
  mesh.dt = safe ? 0.3 : 0.6;
  const Real bdt = driver.beta[stage-1]*mesh.dt;
  Kokkos::deep_copy(pmhd->u0, 0.0);
  Kokkos::deep_copy(pmhd->u1, 0.0);
  Kokkos::deep_copy(pmhd->w0, 0.0);
  Kokkos::deep_copy(pmhd->bcc0, 0.0);
  Kokkos::deep_copy(pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x3f, 0.0);
  Kokkos::deep_copy(pmhd->b1.x1f, 0.0);
  Kokkos::deep_copy(pmhd->b1.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b1.x3f, 0.0);
  Kokkos::deep_copy(pmhd->uflx.x1f, 0.0);
  Kokkos::deep_copy(pmhd->uflx.x2f, 0.0);
  Kokkos::deep_copy(pmhd->uflx.x3f, 0.0);
  Kokkos::deep_copy(pmhd->e3x1, 0.0);
  Kokkos::deep_copy(pmhd->e2x1, 0.0);
  Kokkos::deep_copy(pmhd->e1x2, 0.0);
  Kokkos::deep_copy(pmhd->e3x2, 0.0);
  Kokkos::deep_copy(pmhd->e2x3, 0.0);
  Kokkos::deep_copy(pmhd->e1x3, 0.0);
  auto u = Kokkos::create_mirror(pmhd->u0);
  auto previous = Kokkos::create_mirror(pmhd->u1);
  auto w = Kokkos::create_mirror(pmhd->w0);
  auto f = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->uflx.x1f);
  Kokkos::deep_copy(u, pmhd->u0);
  Kokkos::deep_copy(previous, pmhd->u1);
  Kokkos::deep_copy(w, pmhd->w0);
  for (int m=0; m<pack->nmb_thispack; ++m) {
    const auto &size = pack->pmb->mb_size.h_view(m);
    const int x = std::lround(size.x1min);
    const int y = std::lround(size.x2min);
    const int z = std::lround(size.x3min);
    for (int k=0; k<u.extent_int(2); ++k) {
      for (int j=0; j<u.extent_int(3); ++j) {
        for (int i=0; i<u.extent_int(4); ++i) {
          const int gx = (x+i-ind.is+32)%32;
          const int gy = (y+j-ind.js+32)%32;
          const int gz = (z+k-ind.ks+32)%32;
          const Real rho = gx==15 && gy==15 && gz==15 ? 1.0 : 0.01;
          u(m,IDN,k,j,i) = w(m,IDN,k,j,i) = rho;
          u(m,IEN,k,j,i) = w(m,IEN,k,j,i) = 0.9*rho;
          previous(m,IDN,k,j,i) = 0.01;
          previous(m,IEN,k,j,i) = 0.009;
        }
        for (int i=0; i<f.extent_int(4); ++i) {
          if (x+i-ind.is==16 && y+j-ind.js==15 && z+k-ind.ks==15) {
            f(m,IDN,k,j,i) = 2.0/bdt;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(pmhd->u0, u);
  Kokkos::deep_copy(pmhd->u1, previous);
  Kokkos::deep_copy(pmhd->w0, w);
  Kokkos::deep_copy(pmhd->uflx.x1f, f);
  pmhd->NewTimeStep(&driver, driver.nexp_stages);
  int errors = std::abs(pmhd->dtnew-1.0)>1e-12 ? 1 : 0;
  if (errors) return errors;
  if (global_variable::my_rank==0) {
    std::printf("FOFC_CFL_FIXTURE dt=%.17e directional_dt=%.17e "
                "expected_center_rho=%.17e\n", mesh.dt, pmhd->dtnew,
                0.2575-bdt*3.0*0.99);
    std::fflush(stdout);
  }
  pmhd->FOFC(&driver, stage);  // unsafe case must abort here, after all faces use LLF
  const auto fx = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->uflx.x1f);
  const auto fy = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->uflx.x2f);
  const auto fz = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->uflx.x3f);
  Real mass = 0.0;
  for (int m=0; m<pack->nmb_thispack; ++m) {
    const auto &size = pack->pmb->mb_size.h_view(m);
    for (int k=ind.ks; k<=ind.ke; ++k) {
      for (int j=ind.js; j<=ind.je; ++j) {
        for (int i=ind.is; i<=ind.ie; ++i) {
          const Real rho = 0.25*u(m,IDN,k,j,i)+0.75*previous(m,IDN,k,j,i)-bdt*(
              fx(m,IDN,k,j,i+1)-fx(m,IDN,k,j,i) +
              fy(m,IDN,k,j+1,i)-fy(m,IDN,k,j,i) +
              fz(m,IDN,k+1,j,i)-fz(m,IDN,k,j,i));
          if (!std::isfinite(rho) || !(rho>0.0)) ++errors;
          if (std::lround(size.x1min)+i-ind.is==15 &&
              std::lround(size.x2min)+j-ind.js==15 &&
              std::lround(size.x3min)+k-ind.ks==15 &&
              std::abs(rho-(0.2575-bdt*3.0*0.99))>1e-12) ++errors;
          mass += rho;
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (std::abs(mass-(32768*0.01+0.25*0.99))>1e-8) ++errors;
  if (!safe) ++errors;
  return errors;
}
#endif // TST_UNIT_FOFC_CFL_HPP_
