//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2026 James Beattie and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fofc_diagnostics.cpp
//! \brief Bounded, failure-only snapshots of an inadmissible iterative FOFC stage.

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "athena.hpp"
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "mhd/fofc_diagnostics.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "mhd/rsolvers/llf_mhd_singlestate.hpp"

namespace mhd {

void DumpFOFCFailure(MHD *pmhd, MeshBlockPack *pack, Driver *driver, int stage,
                     int last_recheck) {
  // Copies and file I/O happen only after the existing fatal condition is reached.
  // Avoid MPI: other ranks may already be unwinding or waiting in a later task.
  Kokkos::fence();
  const auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u0);
  const auto u1 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u1);
  const auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  const auto b = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bcc0);
  const auto ho = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->utest);
  const auto mask = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->fofc);
  const auto f1 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->uflx.x1f);
  const std::array<decltype(f1), 3> f = {f1,
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->uflx.x2f),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->uflx.x3f)};
  const auto bx = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x1f);
  const std::array<decltype(bx), 3> bf = {bx,
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x2f),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x3f)};
  const std::array<decltype(bx), 3> oldb = {
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b1.x1f),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b1.x2f),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b1.x3f)};
  // Each pair is in the same by/bz order as SingleStateLLF_MHD's outputs.
  const std::array<decltype(bx), 6> emf = {
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e3x1),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e2x1),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e1x2),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e3x2),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e2x3),
    Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e1x3)};
  const auto &ind = pack->pmesh->mb_indcs;
  const auto &eos = pmhd->peos->eos_data;
  const int ndim = pack->pmesh->three_d ? 3 : (pack->pmesh->multi_d ? 2 : 1);
  const Real dt = pack->pmesh->dt;
  const Real g0 = driver->gam0[stage-1], g1 = driver->gam1[stage-1];
  const Real bdt = driver->beta[stage-1]*dt;
  char filename[128];
  std::snprintf(filename, sizeof(filename),
                "fofc_failure_rank%06d_cycle%08d_stage%d.txt",
                global_variable::my_rank, pack->pmesh->ncycle, stage);
  std::ofstream out(filename);
  if (!out) {
    std::fprintf(stderr, "FOFC_SNAPSHOT_WRITE_FAILED %s\n", filename);
    return;
  }
  out << std::scientific << std::setprecision(17);
  out << "FOFC_SNAPSHOT 1\nMETA " << global_variable::my_rank << ' '
      << pack->pmesh->ncycle << ' ' << pack->pmesh->time << ' ' << dt << ' '
      << stage << ' ' << last_recheck << ' ' << pmhd->fofc_max_iterations << ' '
      << g0 << ' ' << g1 << ' ' << driver->beta[stage-1] << ' ' << ndim << '\n';
  out << "EOS " << eos.gamma << ' ' << eos.dfloor << ' ' << eos.pfloor << ' '
      << eos.tfloor << ' ' << eos.sfloor << '\n';
  out << "# META rank cycle time dt stage last_recheck cap gam0 gam1 beta ndim\n"
      << "# FACE axis side b0_normal b1_normal speed stored_flux[5] "
      << "stored_emf[2] llf_flux[5] llf_emf[2]\n"
      << "# STENCIL di dj dk flag U0[5] U1[5] W0[5] Bcc0[3]\n"
      << "# DENSITY rho_weighted stored_div llf_div drain_courant "
      << "central_coefficient neighbor_inflow llf_rho coefficient_rho\n";
  int dumped = 0, hard_total = 0;
  for (int m=0; m<pack->nmb_thispack; ++m) {
    const auto &size = pack->pmb->mb_size.h_view(m);
    const Real dx[3] = {size.dx1, size.dx2, size.dx3};
    for (int k=ind.ks; k<=ind.ke; ++k) {
      for (int j=ind.js; j<=ind.je; ++j) {
        for (int i=ind.is; i<=ind.ie; ++i) {
          // Matching the last production recheck, but on host, after all work ends.
          bool affected = mask(m,k,j,i) || mask(m,k,j,i-1) || mask(m,k,j,i+1);
          if (ndim>1) affected |= mask(m,k,j-1,i) || mask(m,k,j+1,i);
          if (ndim>2) affected |= mask(m,k-1,j,i) || mask(m,k+1,j,i);
          if (!affected) continue;
          Real c[5], div[5] = {};
          for (int n=0; n<5; ++n) {
            for (int a=0; a<ndim; ++a) {
              div[n] += bdt/dx[a]*(f[a](m,n,k+(a==2),j+(a==1),i+(a==0))
                                     - f[a](m,n,k,j,i));
            }
            c[n] = g0*u(m,n,k,j,i) + g1*u1(m,n,k,j,i) - div[n];
          }
          Real bc[3];
          for (int a=0; a<3; ++a) {
            bc[a] = g0*b(m,a,k,j,i) + g1*0.5*(oldb[a](m,k,j,i)
                        + oldb[a](m,k+(a==2),j+(a==1),i+(a==0)));
          }
          for (int a=0; a<ndim; ++a) {
            bc[(a+1)%3] += bdt/dx[a]*(emf[2*a](m,k+(a==2),j+(a==1),i+(a==0))
                                       - emf[2*a](m,k,j,i));
            bc[(a+2)%3] -= bdt/dx[a]*(emf[2*a+1](m,k+(a==2),j+(a==1),i+(a==0))
                                       - emf[2*a+1](m,k,j,i));
          }
          MHDCons1D trial;
          trial.d = c[IDN]; trial.mx = c[IM1]; trial.my = c[IM2];
          trial.mz = c[IM3]; trial.e = c[IEN];
          trial.bx = bc[0]; trial.by = bc[1]; trial.bz = bc[2];
          bool finite = true;
          for (const auto value : c) finite &= std::isfinite(value);
          for (const auto value : bc) finite &= std::isfinite(value);
          bool df = false, ef = false, tf = false;
          if (finite) {
            HydPrim1D prim;
            SingleC2P_IdealMHD(trial, eos, prim, df, ef, tf);
            finite = std::isfinite(prim.d) && std::isfinite(prim.vx) &&
                std::isfinite(prim.vy) && std::isfinite(prim.vz) &&
                std::isfinite(prim.e) && std::isfinite(trial.e);
          }
          if (finite && c[IDN]>0.0) continue;
          ++hard_total;
          if (dumped>=32) continue;
          ++dumped;
          out << "CELL " << m << ' ' << pack->pmb->mb_gid.h_view(m) << ' '
              << i << ' ' << j << ' ' << k << ' ' << mask(m,k,j,i) << ' '
              << finite << ' ' << df << ' ' << ef << ' ' << tf << '\n';
          out << "DX " << dx[0] << ' ' << dx[1] << ' ' << dx[2] << '\n';
          out << "HIGH_ORDER";
          for (int n=0; n<5; ++n) out << ' ' << ho(m,n,k,j,i);
          out << "\nTRIAL";
          for (const auto value : c) out << ' ' << value;
          for (const auto value : bc) out << ' ' << value;
          out << '\n';
          for (int dk=(ndim>2 ? -1 : 0); dk<=(ndim>2 ? 1 : 0); ++dk) {
            for (int dj=(ndim>1 ? -1 : 0); dj<=(ndim>1 ? 1 : 0); ++dj) {
              for (int di=-1; di<=1; ++di) {
                out << "STENCIL " << di << ' ' << dj << ' ' << dk << ' '
                    << mask(m,k+dk,j+dj,i+di);
                for (int n=0; n<5; ++n) out << ' ' << u(m,n,k+dk,j+dj,i+di);
                for (int n=0; n<5; ++n) out << ' ' << u1(m,n,k+dk,j+dj,i+di);
                for (int n=0; n<5; ++n) out << ' ' << w(m,n,k+dk,j+dj,i+di);
                for (int n=0; n<3; ++n) out << ' ' << b(m,n,k+dk,j+dj,i+di);
                out << '\n';
              }
            }
          }
          Real llf_div = 0.0, drain = 0.0, inflow = 0.0;
          for (int a=0; a<ndim; ++a) {
            const auto load_primitive = [&](int offset) {
              const int ii = i+(a==0)*offset;
              const int jj = j+(a==1)*offset;
              const int kk = k+(a==2)*offset;
              MHDPrim1D p;
              p.d = w(m,IDN,kk,jj,ii); p.e = w(m,IEN,kk,jj,ii);
              p.vx = w(m,IVX+a,kk,jj,ii);
              p.vy = w(m,IVX+(a+1)%3,kk,jj,ii);
              p.vz = w(m,IVX+(a+2)%3,kk,jj,ii);
              p.by = b(m,(a+1)%3,kk,jj,ii);
              p.bz = b(m,(a+2)%3,kk,jj,ii);
              return p;
            };
            for (int side=0; side<2; ++side) {
              const int ii=i+(a==0)*side, jj=j+(a==1)*side, kk=k+(a==2)*side;
              const auto left = load_primitive(side-1), right = load_primitive(side);
              const Real normal = bf[a](m,kk,jj,ii);
              const Real al = std::abs(left.vx) + eos.IdealMHDFastSpeed(left.d,
                  eos.IdealGasPressure(left.e), normal, left.by, left.bz);
              const Real ar = std::abs(right.vx) + eos.IdealMHDFastSpeed(right.d,
                  eos.IdealGasPressure(right.e), normal, right.by, right.bz);
              const Real speed = std::fmax(al, ar);
              MHDCons1D flux;
              SingleStateLLF_MHD(left, right, normal, eos, flux);
              const Real flux_global[5] = {flux.d,
                  a==0 ? flux.mx : (a==1 ? flux.mz : flux.my),
                  a==0 ? flux.my : (a==1 ? flux.mx : flux.mz),
                  a==0 ? flux.mz : (a==1 ? flux.my : flux.mx), flux.e};
              out << "FACE " << a << ' ' << side << ' ' << normal << ' '
                  << oldb[a](m,kk,jj,ii) << ' ' << speed;
              for (int n=0; n<5; ++n) out << ' ' << f[a](m,n,kk,jj,ii);
              out << ' ' << emf[2*a](m,kk,jj,ii) << ' ' << emf[2*a+1](m,kk,jj,ii);
              for (const auto value : flux_global) out << ' ' << value;
              out << ' ' << flux.by << ' ' << flux.bz << '\n';
              llf_div += (side==1 ? 1.0 : -1.0)*bdt/dx[a]*flux.d;
              drain += 0.5*dt/dx[a]*speed;
              inflow += 0.5*bdt/dx[a]*(side==1 ?
                  (speed-right.vx)*right.d : (speed+left.vx)*left.d);
            }
          }
          const Real weighted = g0*u(m,IDN,k,j,i) + g1*u1(m,IDN,k,j,i);
          const Real central = g0-driver->beta[stage-1]*drain;
          const Real coefficient_rho = weighted-driver->beta[stage-1]*drain*
                                      w(m,IDN,k,j,i)+inflow;
          out << "DENSITY " << weighted << ' ' << div[IDN] << ' ' << llf_div << ' '
              << drain << ' ' << central << ' ' << inflow << ' '
              << weighted-llf_div << ' ' << coefficient_rho << "\nEND_CELL\n";
          out.flush();
        }
      }
    }
  }
  out << "END_SNAPSHOT " << dumped << ' ' << hard_total << '\n';
  out.close();
  std::fprintf(stderr, "FOFC_SNAPSHOT file=%s cells=%d hard_recomputed=%d "
      "last_recheck=%d\n", filename, dumped, hard_total, last_recheck);
  std::fflush(stderr);
}

} // namespace mhd
