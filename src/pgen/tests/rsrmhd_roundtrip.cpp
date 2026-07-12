//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_roundtrip.cpp
//! \brief Manufactured known-E primitive/conserved round-trip test for resistive SRMHD.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "mhd/mhd.hpp"
#include "mhd/resistivity_model.hpp"
#include "mhd/rsolvers/llf_srrmhd_singlestate.hpp"
#include "pgen/pgen.hpp"

void SRRMHDRoundTripErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDFluxErrors(Mesh *pm);
void SRRMHDImplicitErrors(Mesh *pm);
void SRRMHDResistivityModelErrors();

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ResistiveSRMHDRoundTrip()
//! \brief Initialize smooth physical states spanning all E, B, and velocity components.

void ProblemGenerator::ResistiveSRMHDRoundTrip(ParameterInput *pin, const bool restart) {
  pgen_final_func = SRRMHDRoundTripErrors;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || !(pmbp->pmhd->is_resistive_rel)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_roundtrip requires <mhd>/resistive_rel=true"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  int nx1 = indcs.nx1;
  int nmb = pmbp->nmb_thispack;

  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;

  constexpr Real bx = 0.7;
  constexpr Real by = -0.3;
  constexpr Real bz = 1.1;
  Kokkos::deep_copy(b0.x1f, bx);
  Kokkos::deep_copy(b0.x2f, by);
  Kokkos::deep_copy(b0.x3f, bz);

  par_for("pgen_rsrmhd_roundtrip", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = (static_cast<Real>(i-is) + 0.5)/static_cast<Real>(nx1);
    Real phase = 2.0*M_PI*x;

    w0(m, IDN, k, j, i) = 0.5 + 1.5*x;
    w0(m, IVX, k, j, i) = 0.8*sin(phase);
    w0(m, IVY, k, j, i) = -0.6*cos(phase);
    w0(m, IVZ, k, j, i) = 0.4*sin(2.0*phase);
    w0(m, IEN, k, j, i) = 0.2 + 0.7*x;
    w0(m, srrmhd::IRE1, k, j, i) = 0.25*cos(phase);
    w0(m, srrmhd::IRE2, k, j, i) = -0.35*sin(phase);
    w0(m, srrmhd::IRE3, k, j, i) = 0.15*cos(2.0*phase);

    bcc0(m, IBX, k, j, i) = bx;
    bcc0(m, IBY, k, j, i) = by;
    bcc0(m, IBZ, k, j, i) = bz;
  });

  pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);

  // Driver::Initialize() will recover w0 from u0.  Preserve the manufactured primitive
  // state in the MHD-owned diagnostic registers for comparison in the finalizer.
  pmbp->pmhd->SetSaveWBcc();
  Kokkos::deep_copy(pmbp->pmhd->wsaved, w0);
  Kokkos::deep_copy(pmbp->pmhd->bccsaved, bcc0);
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDRoundTripErrors()
//! \brief Measure the known-E primitive round-trip error after Driver initialization.

void SRRMHDRoundTripErrors(ParameterInput *pin, Mesh *pm) {
  (void)pin;
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  auto *pmhd = pm->pmb_pack->pmhd;
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto wref = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->wsaved);
  auto bcc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bcc0);
  auto bccref = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bccsaved);

  Real max_abs = 0.0;
  Real max_rel = 0.0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          for (int n = 0; n < srrmhd::NSRRMHD; ++n) {
            Real abs_err = fabs(w(m, n, k, j, i) - wref(m, n, k, j, i));
            Real rel_err = abs_err/std::max(1.0, fabs(wref(m, n, k, j, i)));
            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, rel_err);
          }
          for (int n = 0; n < NMAG; ++n) {
            Real abs_err = fabs(bcc(m, n, k, j, i) - bccref(m, n, k, j, i));
            Real rel_err = abs_err/std::max(1.0, fabs(bccref(m, n, k, j, i)));
            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, rel_err);
          }
        }
      }
    }
  }

  if (global_variable::my_rank == 0) {
    Real diagnostic_dt = 0.0;
    if (pin->GetString("time", "evolution").compare("static") != 0) {
      diagnostic_dt = pmhd->dtnew;
    }
    std::ofstream file("rsrmhd_roundtrip-errs.dat");
    file << std::setprecision(17) << max_rel << " " << max_abs << " "
         << pm->ecounter.neos_dfloor << " " << pm->ecounter.neos_efloor << " "
         << pm->ecounter.neos_vceil << " " << pm->ecounter.neos_fail << " "
         << pmhd->nmhd << " " << diagnostic_dt << std::endl;
  }

  SRRMHDFluxErrors(pm);
  SRRMHDImplicitErrors(pm);
  SRRMHDResistivityModelErrors();
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDFluxErrors()
//! \brief Verify physical fluxes, LLF consistency, light-speed diffusion, and CT signs.

void SRRMHDFluxErrors(Mesh *pm) {
  auto eos = pm->pmb_pack->pmhd->peos->eos_data;
  DvceArray2D<Real> result("rsrmhd_flux_test", 3, 10);

  Kokkos::parallel_for("rsrmhd_flux_test", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
  KOKKOS_LAMBDA(const int idx) {
    srrmhd::SRRMHDPrim1D w;
    w.d = 1.3;
    w.vx = 0.4;
    w.vy = -0.2;
    w.vz = 0.3;
    w.e = 0.8;
    w.ex = 0.15;
    w.ey = -0.25;
    w.ez = 0.35;
    w.bx = 0.7;
    w.by = -0.3;
    w.bz = 1.1;

    srrmhd::SRRMHDCons1D u, physical, llf_equal;
    mhd::SingleStateFlux_SRRMHD(w, eos, u, physical);
    mhd::SingleStateLLF_SRRMHD(w, w, eos, llf_equal);

    result(0, 0) = physical.d;
    result(0, 1) = physical.mx;
    result(0, 2) = physical.my;
    result(0, 3) = physical.mz;
    result(0, 4) = physical.e;
    result(0, 5) = physical.ex;
    result(0, 6) = physical.ey;
    result(0, 7) = physical.ez;
    result(0, 8) = physical.by;
    result(0, 9) = physical.bz;

    result(1, 0) = llf_equal.d;
    result(1, 1) = llf_equal.mx;
    result(1, 2) = llf_equal.my;
    result(1, 3) = llf_equal.mz;
    result(1, 4) = llf_equal.e;
    result(1, 5) = llf_equal.ex;
    result(1, 6) = llf_equal.ey;
    result(1, 7) = llf_equal.ez;
    result(1, 8) = llf_equal.by;
    result(1, 9) = llf_equal.bz;

    srrmhd::SRRMHDPrim1D wr = w;
    wr.ex += 0.6;
    wr.by -= 0.4;
    srrmhd::SRRMHDCons1D llf_jump;
    mhd::SingleStateLLF_SRRMHD(w, wr, eos, llf_jump);
    result(2, 0) = llf_jump.ex;
    result(2, 1) = llf_jump.by;
    result(2, 2) = wr.ex - w.ex;
    result(2, 3) = wr.by - w.by;
    result(2, 4) = -physical.by;
    result(2, 5) = physical.bz;
    (void)idx;
  });

  auto r = Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);

  constexpr Real rho = 1.3;
  constexpr Real ux = 0.4;
  constexpr Real uy = -0.2;
  constexpr Real uz = 0.3;
  constexpr Real eint = 0.8;
  constexpr Real ex = 0.15;
  constexpr Real ey = -0.25;
  constexpr Real ez = 0.35;
  constexpr Real bx = 0.7;
  constexpr Real by = -0.3;
  constexpr Real bz = 1.1;

  Real lor = sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
  Real pgas = (eos.gamma - 1.0)*eint;
  Real wgas = rho + eos.gamma*eint;
  Real em_pressure = 0.5*(SQR(ex) + SQR(ey) + SQR(ez)
                          + SQR(bx) + SQR(by) + SQR(bz));
  Real sx = wgas*lor*ux + ey*bz - ez*by;

  Real expected[10];
  expected[0] = rho*ux;
  expected[1] = wgas*SQR(ux) + pgas + em_pressure - SQR(ex) - SQR(bx);
  expected[2] = wgas*ux*uy - ex*ey - bx*by;
  expected[3] = wgas*ux*uz - ex*ez - bx*bz;
  expected[4] = sx - rho*ux;
  expected[5] = 0.0;
  expected[6] = bz;
  expected[7] = -by;
  expected[8] = -ez;
  expected[9] = ey;

  Real physical_err = 0.0;
  Real equal_state_err = 0.0;
  for (int n = 0; n < 10; ++n) {
    physical_err = std::max(physical_err, fabs(r(0, n) - expected[n]));
    equal_state_err = std::max(equal_state_err, fabs(r(1, n) - r(0, n)));
  }

  // With lambda=1, a jump only in normal E has LLF flux -Delta(E_normal)/2.
  Real light_speed_err = fabs(r(2, 0) + 0.5*r(2, 2));
  // F(By)=-Ez plus LLF dissipation; CT stores -F(By).  F(Bz)=Ey.
  Real expected_fby_jump = -ez - 0.5*r(2, 3);
  Real ct_sign_err = std::max(fabs(r(2, 1) - expected_fby_jump),
                              std::max(fabs(r(2, 4) - ez), fabs(r(2, 5) - ey)));

  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_flux-errs.dat");
    file << std::setprecision(17) << physical_err << " " << equal_state_err << " "
         << light_speed_err << " " << ct_sign_err << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDImplicitErrors()
//! \brief Verify the closed-form E update and coupled three-variable Newton recovery.

void SRRMHDImplicitErrors(Mesh *pm) {
  auto eos = pm->pmb_pack->pmhd->peos->eos_data;
  constexpr int ncases = 4;
  DvceArray2D<Real> result("rsrmhd_implicit_test", ncases, 5);

  Kokkos::parallel_for("rsrmhd_implicit_test",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncases), KOKKOS_LAMBDA(const int n) {
    const Real kappa_values[ncases] = {1.0e-4, 0.1, 10.0, 1.0e4};
    const Real kappa = kappa_values[n];

    srrmhd::SRRMHDPrim1D target;
    target.d = 0.8 + 0.2*n;
    target.vx = 0.25 + 0.08*n;
    target.vy = -0.35 + 0.04*n;
    target.vz = 0.18 - 0.03*n;
    target.e = 0.45 + 0.15*n;
    target.ex = 0.12 - 0.025*n;
    target.ey = -0.27 + 0.03*n;
    target.ez = 0.31 - 0.02*n;
    target.bx = 0.7;
    target.by = -0.4;
    target.bz = 0.9;

    srrmhd::SRRMHDCons1D conserved;
    srrmhd::SingleP2C_IdealSRRMHD(target, eos.gamma, conserved);

    const Real lor = sqrt(1.0 + SQR(target.vx) + SQR(target.vy)
                           + SQR(target.vz));
    const Real vx = target.vx/lor;
    const Real vy = target.vy/lor;
    const Real vz = target.vz/lor;
    const Real vxb_x = vy*target.bz - vz*target.by;
    const Real vxb_y = vz*target.bx - vx*target.bz;
    const Real vxb_z = vx*target.by - vy*target.bx;
    const Real v_dot_e = vx*target.ex + vy*target.ey + vz*target.ez;
    const Real scale = kappa*lor;
    const Real ex_star = target.ex
        + scale*(target.ex + vxb_x - v_dot_e*vx);
    const Real ey_star = target.ey
        + scale*(target.ey + vxb_y - v_dot_e*vy);
    const Real ez_star = target.ez
        + scale*(target.ez + vxb_z - v_dot_e*vz);

    srrmhd::SRRMHDPrim1D guess = target;
    guess.vx += 0.12;
    guess.vy -= 0.09;
    guess.vz += 0.07;
    srrmhd::SRRMHDPrim1D recovered = guess;
    int iterations = 0;
    const bool success = srrmhd::SingleC2P_IdealSRRMHDImplicit(
        conserved, eos, ex_star, ey_star, ez_star, kappa, guess, recovered,
        iterations);

    Real primitive_error = 0.0;
    if (success) {
      primitive_error = fmax(primitive_error, fabs(recovered.d - target.d));
      primitive_error = fmax(primitive_error, fabs(recovered.vx - target.vx));
      primitive_error = fmax(primitive_error, fabs(recovered.vy - target.vy));
      primitive_error = fmax(primitive_error, fabs(recovered.vz - target.vz));
      primitive_error = fmax(primitive_error, fabs(recovered.e - target.e));
      primitive_error = fmax(primitive_error, fabs(recovered.ex - target.ex));
      primitive_error = fmax(primitive_error, fabs(recovered.ey - target.ey));
      primitive_error = fmax(primitive_error, fabs(recovered.ez - target.ez));
    } else {
      primitive_error = 1.0;
    }

    Real ex_check, ey_check, ez_check;
    srrmhd::ImplicitElectricField(target.vx, target.vy, target.vz, ex_star, ey_star,
                                  ez_star, target.bx, target.by, target.bz, kappa,
                                  ex_check, ey_check, ez_check);
    const Real electric_error = fmax(fabs(ex_check - target.ex),
                                     fmax(fabs(ey_check - target.ey),
                                          fabs(ez_check - target.ez)));

    Real fx, fy, fz;
    srrmhd::SRRMHDPrim1D residual_state;
    const bool residual_ok = srrmhd::ImplicitPrimitiveResidual(
        conserved, eos, ex_star, ey_star, ez_star, kappa, recovered.vx,
        recovered.vy, recovered.vz, fx, fy, fz, residual_state);
    Real residual_error = 1.0;
    if (success && residual_ok) {
      residual_error = fmax(fabs(fx), fmax(fabs(fy), fabs(fz)));
    }

    result(n, 0) = success ? 0.0 : 1.0;
    result(n, 1) = primitive_error;
    result(n, 2) = electric_error;
    result(n, 3) = residual_error;
    result(n, 4) = iterations;
  });

  auto r = Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
  Real failures = 0.0;
  Real primitive_error = 0.0;
  Real electric_error = 0.0;
  Real residual_error = 0.0;
  Real max_iterations = 0.0;
  for (int n = 0; n < ncases; ++n) {
    failures += r(n, 0);
    primitive_error = std::max(primitive_error, r(n, 1));
    electric_error = std::max(electric_error, r(n, 2));
    residual_error = std::max(residual_error, r(n, 3));
    max_iterations = std::max(max_iterations, r(n, 4));
  }

  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_implicit-errs.dat");
    file << std::setprecision(17) << failures << " " << primitive_error << " "
         << electric_error << " " << residual_error << " " << max_iterations
         << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDResistivityModelErrors()
//! \brief Verify the device charge-starvation evaluator and eta floor.

void SRRMHDResistivityModelErrors() {
  DvceArray2D<Real> result("rsrmhd_resistivity_model_test", 1, 4);

  Kokkos::parallel_for("rsrmhd_resistivity_model_test",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, 1), KOKKOS_LAMBDA(const int n) {
    srrmhd::ResistivityData uniform;
    uniform.model = srrmhd::ResistivityModel::uniform;
    uniform.eta_uniform = 0.037;
    result(0, 0) = srrmhd::EvaluateResistivity(
        uniform, 2.0, 0.3, -0.2, 0.1, 1.0, 2.0, 3.0, -0.4, 0.5, 0.7);

    srrmhd::ResistivityData effective;
    effective.model = srrmhd::ResistivityModel::charge_starvation;
    effective.eta_floor = 1.0e-8;
    effective.eta_scale = 2.0;
    effective.number_per_mass = 0.5;
    result(0, 1) = srrmhd::EvaluateResistivity(
        effective, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.7, -0.4, 0.9);

    // u=(0.75,0,0) gives Gamma=1.25 and v=(0.6,0,0).  E=(0,1.2,0)
    // exactly cancels v cross B for B=(0,0,2), so only the floor remains.
    result(0, 2) = srrmhd::EvaluateResistivity(
        effective, 1.0, 0.75, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 2.0);

    // For E parallel to v, |E_*|=1.6.  With n=2*rho=1.6 and scale=0.5,
    // eta is exactly 0.5.
    effective.eta_scale = 0.5;
    effective.number_per_mass = 2.0;
    result(0, 3) = srrmhd::EvaluateResistivity(
        effective, 0.8, 0.75, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    (void)n;
  });

  auto r = Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
  const Real expected[4] = {0.037, 10.0, 1.0e-8, 0.5};
  Real max_error = 0.0;
  for (int n = 0; n < 4; ++n) {
    max_error = std::max(max_error, fabs(r(0, n) - expected[n]));
  }
  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_resistivity_model-errs.dat");
    file << std::setprecision(17) << max_error << " " << r(0, 2) << std::endl;
  }
}
