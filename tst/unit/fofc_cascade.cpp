// Copyright (C) 2026 James Beattie and the Athena code team.
// Licensed under the 3-clause BSD License (the "LICENSE").
// Deterministic tests of the production FOFC kernels; run only under Slurm on Trillium.
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "mhd/fofc_boundary.hpp"
#include "driver/driver.hpp"

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
#endif
  Kokkos::initialize(argc, argv);
  int errors = 0;
  {
    const std::string mode = argc > 1 ? argv[1] : "cascade";
    const bool mask_test = mode == "mask3d";
    const bool rk3 = mode == "rk3";
    const bool deep = mode == "deep" || mode == "limit";
    const bool legacy = mode == "legacy";
    const bool energy_test = mode == "energy";
    const bool soft_floor = mode == "soft_floor";
    const bool invalid = mode == "invalid";
    const int nx = mask_test ? 32 : 128;
    ParameterInput pin;
    std::ostringstream input;
    input << "<mesh>\nnghost=4\nnx1=" << nx
          << "\nnx2=" << (mask_test ? 32 : 1)
          << "\nnx3=" << (mask_test ? 32 : 1)
          << "\nx1min=0\nx1max=" << nx
          << "\nx2min=0\nx2max=32\nx3min=0\nx3max=32\n"
          << "ix1_bc=periodic\nox1_bc=periodic\nix2_bc=periodic\nox2_bc=periodic\n"
          << "ix3_bc=periodic\nox3_bc=periodic\n"
          << "<meshblock>\nnx1=16\nnx2=" << (mask_test ? 16 : 1)
          << "\nnx3=" << (mask_test ? 16 : 1)
          << "\n<time>\nevolution=dynamic\nintegrator=" << (rk3 ? "rk3" : "rk1")
          << "\ncfl_number=0.6\ntlim=1\n"
          << "<mhd>\neos=ideal\ngamma=1.6666666666666667\n"
          << "reconstruct=wenoz\nrsolver=hlld\nfofc=true\n"
          << "dfloor=1e-12\npfloor=1e-14\ntfloor=" << (soft_floor ? 10 : 0)
          << "\nfofc_max_iterations=" << (legacy ? 1 : (mode == "limit" ? 2 : 8))
          << "\nfofc_diagnostics=false\n";
    std::istringstream stream(input.str());
    pin.LoadFromStream(stream);
    Mesh mesh(&pin);
    mesh.BuildTreeFromScratch(&pin);
    mesh.AddCoordinatesAndPhysics(&pin);
    mesh.time = 0.0;
    mesh.ncycle = 0;
    mesh.dt = 0.1;
    auto pack = mesh.pmb_pack;
    auto pmhd = pack->pmhd;
    const auto &ind = mesh.mb_indcs;
    const int nmb = pack->nmb_thispack;
    const auto &sizes = pack->pmb->mb_size.h_view;
    Kokkos::Timer timer;
    Driver driver(&pin, &mesh, 0.0, &timer);
    const int stage = rk3 ? 3 : 1;
    const Real beta_dt = driver.beta[stage-1]*mesh.dt;

    if (mask_test) {
      auto flags = Kokkos::create_mirror_view(pmhd->fofc);
      Kokkos::deep_copy(flags, false);
      for (int m=0; m<nmb; ++m) {
        int x = std::lround(sizes(m).x1min);
        int y = std::lround(sizes(m).x2min);
        int z = std::lround(sizes(m).x3min);
        for (int k=ind.ks; k<=ind.ke; ++k) {
          for (int j=ind.js; j<=ind.je; ++j) {
            for (int i=ind.is; i<=ind.ie; ++i) {
              flags(m,k,j,i) = ((x+i-ind.is)+3*(y+j-ind.js)+5*(z+k-ind.ks))%7 == 0;
            }
          }
        }
      }
      Kokkos::deep_copy(pmhd->fofc, flags);
      mhd::FOFCBoundary boundary(pack, pmhd->pbval_u);
      boundary.Exchange(pmhd->fofc);
      Kokkos::deep_copy(flags, pmhd->fofc);
      for (int m=0; m<nmb; ++m) {
        int x = std::lround(sizes(m).x1min);
        int y = std::lround(sizes(m).x2min);
        int z = std::lround(sizes(m).x3min);
        for (int k=ind.ks-1; k<=ind.ke+1; ++k) {
          for (int j=ind.js-1; j<=ind.je+1; ++j) {
            for (int i=ind.is-1; i<=ind.ie+1; ++i) {
              int gx = (x+i-ind.is+32)%32;
              int gy = (y+j-ind.js+32)%32;
              int gz = (z+k-ind.ks+32)%32;
              if (flags(m,k,j,i) != ((gx+3*gy+5*gz)%7 == 0)) ++errors;
            }
          }
        }
      }
    } else {
      Kokkos::deep_copy(pmhd->u0, 0.0);
      Kokkos::deep_copy(pmhd->w0, 0.0);
      Kokkos::deep_copy(pmhd->bcc0, 0.0);
      Kokkos::deep_copy(pmhd->b0.x1f, 0.5);
      Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
      Kokkos::deep_copy(pmhd->b0.x3f, 0.0);
      Kokkos::deep_copy(pmhd->b1.x1f, 0.5);
      Kokkos::deep_copy(pmhd->b1.x2f, 0.0);
      Kokkos::deep_copy(pmhd->b1.x3f, 0.0);
      Kokkos::deep_copy(pmhd->e3x1, 0.0);
      Kokkos::deep_copy(pmhd->e2x1, 0.0);
      auto u = Kokkos::create_mirror(pmhd->u0);
      auto w = Kokkos::create_mirror(pmhd->w0);
      Kokkos::deep_copy(u, pmhd->u0);
      Kokkos::deep_copy(w, pmhd->w0);
      auto b = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bcc0);
      auto f = Kokkos::create_mirror_view(pmhd->uflx.x1f);
      Kokkos::deep_copy(f, 0.0);
      // For >1 rank the initially flagged cell sits immediately left of a rank boundary.
      const int center = global_variable::nranks > 1 ? nx/global_variable::nranks-1 : 15;
      const int width = deep ? 4 : 2;
      for (int m=0; m<nmb; ++m) {
        const int start = std::lround(sizes(m).x1min);
        for (int i=0; i<u.extent_int(4); ++i) {
          const int g = start+i-ind.is;
          u(m,IDN,0,0,i) = invalid && g == center ? -1.0 : 1.0;
          u(m,IEN,0,0,i) = 1.625;
          w(m,IDN,0,0,i) = 1.0;
          w(m,IEN,0,0,i) = 1.5;
          b(m,IBX,0,0,i) = 0.5;
        }
        for (int i=0; i<f.extent_int(4); ++i) {
          const int g = start+i-ind.is;
          f(m,IM1,0,0,i) = 0.875;
          if (!soft_floor && !invalid) {
            const Real amplitude = (energy_test ? 3.0 : 2.0)/beta_dt;
            Real flux = g > center-width && g <= center ? -amplitude : 0.0;
            if (g > center && g <= center+width) flux = amplitude;
            f(m,energy_test ? IEN : IDN,0,0,i) = flux;
          }
          if (mode == "nan" && g == center) {
            f(m,IDN,0,0,i) = std::numeric_limits<Real>::quiet_NaN();
          }
        }
      }
      auto previous = Kokkos::create_mirror(pmhd->u1);
      Kokkos::deep_copy(previous, u);
      if (rk3) {
        for (int m=0; m<nmb; ++m) {
          for (int i=0; i<previous.extent_int(4); ++i) {
            previous(m,IDN,0,0,i) = 2.0;
            previous(m,IEN,0,0,i) = 2.0;
          }
        }
      }
      Kokkos::deep_copy(pmhd->u0, u);
      Kokkos::deep_copy(pmhd->u1, previous);
      Kokkos::deep_copy(pmhd->w0, w);
      Kokkos::deep_copy(pmhd->bcc0, b);
      Kokkos::deep_copy(pmhd->uflx.x1f, f);
      pmhd->FOFC(&driver, stage);
      Kokkos::deep_copy(f, pmhd->uflx.x1f);
      const Real expected_rho = driver.gam0[stage-1] +
                                driver.gam1[stage-1]*(rk3 ? 2.0 : 1.0);
      const Real expected_energy = driver.gam0[stage-1]*1.625 +
                                   driver.gam1[stage-1]*(rk3 ? 2.0 : 1.625);
      double mass = 0.0;
      int negative = 0;
      for (int m=0; m<nmb; ++m) {
        for (int i=ind.is; i<=ind.ie; ++i) {
          const Real rho = driver.gam0[stage-1]*u(m,IDN,0,0,i) +
              driver.gam1[stage-1]*previous(m,IDN,0,0,i) -
              beta_dt*(f(m,IDN,0,0,i+1)-f(m,IDN,0,0,i));
          const Real energy = driver.gam0[stage-1]*u(m,IEN,0,0,i) +
              driver.gam1[stage-1]*previous(m,IEN,0,0,i) -
              beta_dt*(f(m,IEN,0,0,i+1)-f(m,IEN,0,0,i));
          if (!std::isfinite(rho) || !std::isfinite(energy)) ++errors;
          if (rho < 0.0) ++negative;
          if (!legacy && (std::abs(rho-expected_rho)>1e-12 ||
                          std::abs(energy-expected_energy)>1e-12)) {
            ++errors;
          }
          mass += rho;
        }
      }
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &negative, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
      if (std::abs(mass-nx*expected_rho)>1e-10 ||
          negative != (legacy ? 2 : 0)) ++errors;
      // The repair must not mutate either stage's saved state or the primitives.
      auto saved = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u0);
      auto saved1 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u1);
      auto prim = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
      for (std::size_t q=0; q<u.size(); ++q) {
        if (saved.data()[q] != u.data()[q] || saved1.data()[q] != previous.data()[q] ||
            prim.data()[q] != w.data()[q]) ++errors;
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::printf("FOFC_TEST mode=%s ranks=%d errors=%d\n",
                  mode.c_str(), global_variable::nranks, errors);
    }
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return errors == 0 ? 0 : 1;
}
