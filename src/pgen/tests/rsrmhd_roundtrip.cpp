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
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "mhd/mhd.hpp"
#include "mhd/relativistic_viscosity.hpp"
#include "mhd/resistivity_model.hpp"
#include "mhd/rsolvers/llf_srrmhd_singlestate.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/relativistic_forcing.hpp"

void SRRMHDRoundTripErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDFluxErrors(Mesh *pm);
void SRRMHDImplicitErrors(Mesh *pm);
void SRRMHDResistivityModelErrors();
void SRRMHDForcingAlgebraErrors();
void SRRMHDViscosityAlgebraErrors(Mesh *pm);
void SRRMHDViscosityKinematicsErrors(Mesh *pm);
void SRRMHDViscousImplicitErrors(Mesh *pm);
void SRRMHDViscousTransportErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDViscousRelaxationErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDViscousTelegraphErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDViscousBoostedErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDViscousLongitudinalErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDViscousShearLayerErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDViscousKHErrors(ParameterInput *pin, Mesh *pm);
void SRRMHDViscousKHHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ResistiveSRMHDRoundTrip()
//! \brief Initialize smooth physical states spanning all E, B, and velocity components.

void ProblemGenerator::ResistiveSRMHDRoundTrip(ParameterInput *pin, const bool restart) {
  const bool transport_test = pin->GetOrAddBoolean("problem", "viscous_transport",
                                                    false);
  const bool relaxation_test = pin->GetOrAddBoolean(
      "problem", "viscous_relaxation", false);
  const bool telegraph_test = pin->GetOrAddBoolean(
      "problem", "viscous_telegraph", false);
  const bool boosted_test = pin->GetOrAddBoolean(
      "problem", "viscous_boosted", false);
  const bool longitudinal_test = pin->GetOrAddBoolean(
      "problem", "viscous_longitudinal", false);
  const bool shear_layer_test = pin->GetOrAddBoolean(
      "problem", "viscous_shear_layer", false);
  const bool kh_test = pin->GetOrAddBoolean(
      "problem", "viscous_kh", false);
  if (transport_test || relaxation_test || telegraph_test || boosted_test
      || longitudinal_test || shear_layer_test || kh_test) {
    if (relaxation_test) {
      pgen_final_func = SRRMHDViscousRelaxationErrors;
    } else if (telegraph_test) {
      pgen_final_func = SRRMHDViscousTelegraphErrors;
    } else if (boosted_test) {
      pgen_final_func = SRRMHDViscousBoostedErrors;
    } else if (longitudinal_test) {
      pgen_final_func = SRRMHDViscousLongitudinalErrors;
    } else if (shear_layer_test) {
      pgen_final_func = SRRMHDViscousShearLayerErrors;
    } else if (kh_test) {
      pgen_final_func = SRRMHDViscousKHErrors;
      user_hist_func = SRRMHDViscousKHHistory;
    } else {
      pgen_final_func = SRRMHDViscousTransportErrors;
    }
    if (restart) return;

    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
    auto &indcs = pmy_mesh_->mb_indcs;
    const int is = indcs.is, ie = indcs.ie;
    const int js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    const int nmb = pmbp->nmb_thispack;
    auto &w0 = pmbp->pmhd->w0;
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    auto &e0 = pmbp->pmhd->e0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    auto &visc_u0 = pmbp->pmhd->visc_u0;
    auto &size = pmbp->pmb->mb_size;
    const Real boost_velocity = boosted_test
        ? pin->GetOrAddReal("problem", "boost_velocity", 0.4) : 0.0;
    const Real v1 = transport_test ? 0.4 : boost_velocity;
    const Real lor = 1.0/sqrt(1.0 - SQR(v1));
    const Real u1 = lor*v1;
    const Real amplitude = (telegraph_test || boosted_test || longitudinal_test
                            || shear_layer_test)
        ? pin->GetOrAddReal("problem", "amplitude", 1.0e-4) : 0.01;
    const Real shear_velocity = kh_test
        ? pin->GetOrAddReal("problem", "shear_velocity", 0.4) : 0.0;
    const Real shear_width = kh_test
        ? pin->GetOrAddReal("problem", "shear_width", 0.04) : 1.0;
    const Real perturbation = kh_test
        ? pin->GetOrAddReal("problem", "perturbation", 0.01) : 0.0;
    const Real perturbation_width = kh_test
        ? pin->GetOrAddReal("problem", "perturbation_width", 0.05) : 1.0;
    const Real background_b1 = pin->GetOrAddReal("problem", "background_b1", 0.0);
    const Real background_b2 = pin->GetOrAddReal("problem", "background_b2", 0.0);
    const Real background_b3 = pin->GetOrAddReal("problem", "background_b3", 0.0);
    const Real background_e1 = pin->GetOrAddReal("problem", "background_e1", 0.0);
    const Real background_e2 = pin->GetOrAddReal("problem", "background_e2", 0.0);
    const Real background_e3 = pin->GetOrAddReal("problem", "background_e3", 0.0);
    const int wave_direction = telegraph_test
        ? pin->GetOrAddInteger("problem", "wave_direction", 1) : 1;
    if (wave_direction < 1 || wave_direction > 3) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "problem/wave_direction must be 1, 2, or 3"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Kokkos::deep_copy(b0.x1f, background_b1);
    Kokkos::deep_copy(b0.x2f, background_b2);
    Kokkos::deep_copy(b0.x3f, background_b3);
    if (pmbp->pmhd->use_electric_ct) {
      Kokkos::deep_copy(e0.x1f, background_e1);
      Kokkos::deep_copy(e0.x2f, background_e2);
      Kokkos::deep_copy(e0.x3f, background_e3);
    }
    par_for("pgen_viscous_transport", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x = size.d_view(m).x1min
                   + (static_cast<Real>(i-is) + 0.5)*size.d_view(m).dx1;
      const Real y = size.d_view(m).x2min
                   + (static_cast<Real>(j-js) + 0.5)*size.d_view(m).dx2;
      const Real z = size.d_view(m).x3min
                   + (static_cast<Real>(k-ks) + 0.5)*size.d_view(m).dx3;
      const Real coordinate = (wave_direction == 1) ? x
          : ((wave_direction == 2) ? y : z);
      const Real transverse_wave = amplitude*sin(2.0*M_PI*coordinate);
      Real initial_u1 = longitudinal_test ? amplitude*sin(2.0*M_PI*x)
          : ((telegraph_test && wave_direction == 3) ? transverse_wave : u1);
      Real initial_u2 = (telegraph_test || boosted_test || shear_layer_test)
          && wave_direction == 1 ? transverse_wave : 0.0;
      Real initial_u3 = telegraph_test && wave_direction == 2
          ? transverse_wave : 0.0;
      if (kh_test) {
        const Real vx = shear_velocity*(tanh((y-0.25)/shear_width)
                                      - tanh((y-0.75)/shear_width) - 1.0);
        const Real layer1 = exp(-SQR((y-0.25)/perturbation_width));
        const Real layer2 = exp(-SQR((y-0.75)/perturbation_width));
        const Real vy = perturbation*sin(2.0*M_PI*x)*(layer1 + layer2);
        const Real gamma_lorentz = 1.0/sqrt(1.0 - SQR(vx) - SQR(vy));
        initial_u1 = gamma_lorentz*vx;
        initial_u2 = gamma_lorentz*vy;
        initial_u3 = 0.0;
      }
      w0(m, IDN, k, j, i) = 1.0;
      w0(m, IVX, k, j, i) = initial_u1;
      w0(m, IVY, k, j, i) = initial_u2;
      w0(m, IVZ, k, j, i) = initial_u3;
      w0(m, IEN, k, j, i) = 1.0;
      w0(m, srrmhd::IRE1, k, j, i) = background_e1;
      w0(m, srrmhd::IRE2, k, j, i) = background_e2;
      w0(m, srrmhd::IRE3, k, j, i) = background_e3;
      bcc0(m, IBX, k, j, i) = background_b1;
      bcc0(m, IBY, k, j, i) = background_b2;
      bcc0(m, IBZ, k, j, i) = background_b3;
      for (int n = 0; n < srrmhd::NVISC; ++n) visc_u0(m, n, k, j, i) = 0.0;
      visc_u0(m, srrmhd::IVP12, k, j, i) =
          transport_test ? lor*0.002 : 0.0;
      if (relaxation_test) {
        visc_u0(m, srrmhd::IVP11, k, j, i) = 0.004;
        visc_u0(m, srrmhd::IVP22, k, j, i) = -0.001;
        visc_u0(m, srrmhd::IVP33, k, j, i) = -0.003;
        visc_u0(m, srrmhd::IVP12, k, j, i) = 0.002;
        visc_u0(m, srrmhd::IVP13, k, j, i) = -0.0015;
        visc_u0(m, srrmhd::IVP23, k, j, i) = amplitude;
      } else if (transport_test) {
        visc_u0(m, srrmhd::IVP23, k, j, i) =
            lor*amplitude*sin(2.0*M_PI*x);
      }
    });
    pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
    par_for("pgen_viscous_transport_total", DevExeSpace(), 0, nmb-1, ks, ke,
            js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      srrmhd::ShearStress pi;
      const Real d = u0(m, IDN, k, j, i);
      pi.p11 = visc_u0(m, srrmhd::IVP11, k, j, i)/d;
      pi.p22 = visc_u0(m, srrmhd::IVP22, k, j, i)/d;
      pi.p33 = visc_u0(m, srrmhd::IVP33, k, j, i)/d;
      pi.p12 = visc_u0(m, srrmhd::IVP12, k, j, i)/d;
      pi.p13 = visc_u0(m, srrmhd::IVP13, k, j, i)/d;
      pi.p23 = visc_u0(m, srrmhd::IVP23, k, j, i)/d;
      srrmhd::SRRMHDCons1D state;
      state.mx = u0(m, IM1, k, j, i);
      state.my = u0(m, IM2, k, j, i);
      state.mz = u0(m, IM3, k, j, i);
      state.e = u0(m, IEN, k, j, i);
      srrmhd::AddShearToConserved(
          w0(m, IVX, k, j, i), w0(m, IVY, k, j, i),
          w0(m, IVZ, k, j, i), pi, state);
      u0(m, IM1, k, j, i) = state.mx;
      u0(m, IM2, k, j, i) = state.my;
      u0(m, IM3, k, j, i) = state.mz;
      u0(m, IEN, k, j, i) = state.e;
    });
    return;
  }

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
  auto &visc_u0 = pmbp->pmhd->visc_u0;

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
  par_for("pgen_rsrmhd_viscous_state", DevExeSpace(), 0, (nmb-1), ks, ke, js, je,
          is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = (static_cast<Real>(i-is) + 0.5)/static_cast<Real>(nx1);
    Real phase = 2.0*M_PI*x;
    srrmhd::ShearStress raw_pi;
    raw_pi.p11 = 0.004*cos(phase);
    raw_pi.p22 = -0.003*cos(phase);
    raw_pi.p33 = -0.001*cos(phase);
    raw_pi.p12 = 0.002*sin(phase);
    raw_pi.p13 = -0.0015*sin(phase);
    raw_pi.p23 = 0.001*cos(2.0*phase);
    const srrmhd::ShearStress pi = srrmhd::ProjectShearStress(
        w0(m, IVX, k, j, i), w0(m, IVY, k, j, i), w0(m, IVZ, k, j, i), raw_pi);
    const Real d = u0(m, IDN, k, j, i);
    visc_u0(m, srrmhd::IVP11, k, j, i) = d*pi.p11;
    visc_u0(m, srrmhd::IVP22, k, j, i) = d*pi.p22;
    visc_u0(m, srrmhd::IVP33, k, j, i) = d*pi.p33;
    visc_u0(m, srrmhd::IVP12, k, j, i) = d*pi.p12;
    visc_u0(m, srrmhd::IVP13, k, j, i) = d*pi.p13;
    visc_u0(m, srrmhd::IVP23, k, j, i) = d*pi.p23;

    srrmhd::SRRMHDCons1D state;
    state.mx = u0(m, IM1, k, j, i);
    state.my = u0(m, IM2, k, j, i);
    state.mz = u0(m, IM3, k, j, i);
    state.e = u0(m, IEN, k, j, i);
    srrmhd::AddShearToConserved(w0(m, IVX, k, j, i), w0(m, IVY, k, j, i),
                                w0(m, IVZ, k, j, i), pi, state);
    u0(m, IM1, k, j, i) = state.mx;
    u0(m, IM2, k, j, i) = state.my;
    u0(m, IM3, k, j, i) = state.mz;
    u0(m, IEN, k, j, i) = state.e;
  });

  // Driver::Initialize() will recover w0 from u0.  Preserve the manufactured primitive
  // state in the MHD-owned diagnostic registers for comparison in the finalizer.
  pmbp->pmhd->SetSaveWBcc();
  Kokkos::deep_copy(pmbp->pmhd->wsaved, w0);
  Kokkos::deep_copy(pmbp->pmhd->bccsaved, bcc0);
}

//----------------------------------------------------------------------------------------
//! \brief Compare homogeneous IMEX shear relaxation with the exact exponential.

void SRRMHDViscousRelaxationErrors(ParameterInput *pin, Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  const Real tau = pin->GetReal("mhd", "shear_relaxation_time");
  constexpr Real initial[srrmhd::NVISC] = {
      0.004, -0.001, -0.003, 0.002, -0.0015, 0.01};
  Real stress_sum[srrmhd::NVISC] = {};
  Real stress_error[srrmhd::NVISC] = {};
  Real electric_sum[3] = {};
  Real fluid_error = 0.0;
  int ncells = 0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      for (int n = 0; n < srrmhd::NVISC; ++n) {
        const Real exact = initial[n]*exp(-pm->time/tau);
        const Real stress = visc_u(m, n, indcs.ks, indcs.js, i);
        stress_sum[n] += stress;
        stress_error[n] = std::max(stress_error[n], fabs(stress - exact));
      }
      electric_sum[0] += w(m, srrmhd::IRE1, indcs.ks, indcs.js, i);
      electric_sum[1] += w(m, srrmhd::IRE2, indcs.ks, indcs.js, i);
      electric_sum[2] += w(m, srrmhd::IRE3, indcs.ks, indcs.js, i);
      fluid_error = std::max(fluid_error,
          fabs(w(m, IDN, indcs.ks, indcs.js, i) - 1.0));
      fluid_error = std::max(fluid_error,
          fabs(w(m, IEN, indcs.ks, indcs.js, i) - 1.0));
      fluid_error = std::max(fluid_error,
          fabs(w(m, IVX, indcs.ks, indcs.js, i)));
      fluid_error = std::max(fluid_error,
          fabs(w(m, IVY, indcs.ks, indcs.js, i)));
      fluid_error = std::max(fluid_error,
          fabs(w(m, IVZ, indcs.ks, indcs.js, i)));
      ++ncells;
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, stress_sum, srrmhd::NVISC, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, stress_error, srrmhd::NVISC, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, electric_sum, 3, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &fluid_error, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ncells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::string filename = "rsrmhd_viscous_relaxation-errs.dat";
    if (pin->DoesParameterExist("problem", "viscous_diagnostic_name")) {
      const std::string name = pin->GetString("problem", "viscous_diagnostic_name");
      if (name.compare("none") != 0) filename = name + "-errs.dat";
    }
    std::ofstream file(filename);
    file << "# Nx Ncycle time fluid_error mean_pi11 mean_pi22 mean_pi33 "
         << "mean_pi12 mean_pi13 mean_pi23 err_pi11 err_pi22 err_pi33 "
         << "err_pi12 err_pi13 err_pi23 mean_E1 mean_E2 mean_E3\n"
         << std::setprecision(17)
         << pm->mesh_indcs.nx1 << " " << pm->ncycle << " " << pm->time << " "
         << fluid_error;
    for (int n = 0; n < srrmhd::NVISC; ++n) {
      file << " " << stress_sum[n]/static_cast<Real>(ncells);
    }
    for (int n = 0; n < srrmhd::NVISC; ++n) file << " " << stress_error[n];
    for (int n = 0; n < 3; ++n) {
      file << " " << electric_sum[n]/static_cast<Real>(ncells);
    }
    file << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Compare the linear transverse mode with the Israel--Stewart telegraph solution.

void SRRMHDViscousTelegraphErrors(ParameterInput *pin, Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto size_h = pm->pmb_pack->pmb->mb_size.h_view;
  const Real amplitude = pin->GetReal("problem", "amplitude");
  const Real nu = pin->GetReal("mhd", "shear_viscosity");
  const Real tau = pin->GetReal("mhd", "shear_relaxation_time");
  const Real gamma = pin->GetReal("mhd", "gamma");
  const Real wave_number = 2.0*M_PI;
  const int wave_direction = pin->GetOrAddInteger("problem", "wave_direction", 1);
  const int velocity_component = (wave_direction == 1) ? IVY
      : ((wave_direction == 2) ? IVZ : IVX);
  const int stress_component = (wave_direction == 1) ? srrmhd::IVP12
      : ((wave_direction == 2) ? srrmhd::IVP23 : srrmhd::IVP13);
  const Real alpha = 0.5/tau;
  const Real discriminant = 1.0 - 4.0*tau*nu*SQR(wave_number);
  const Real decay = exp(-alpha*pm->time);
  Real exact_velocity;
  Real velocity_derivative;
  if (discriminant > 1.0e-14) {
    const Real rate = sqrt(discriminant)/(2.0*tau);
    exact_velocity = amplitude*decay*
        (cosh(rate*pm->time) + alpha*sinh(rate*pm->time)/rate);
    velocity_derivative = -amplitude*decay*nu*SQR(wave_number)
        *sinh(rate*pm->time)/(tau*rate);
  } else if (discriminant < -1.0e-14) {
    const Real omega = sqrt(-discriminant)/(2.0*tau);
    exact_velocity = amplitude*decay*
        (cos(omega*pm->time) + alpha*sin(omega*pm->time)/omega);
    velocity_derivative = -amplitude*decay*nu*SQR(wave_number)
        *sin(omega*pm->time)/(tau*omega);
  } else {
    exact_velocity = amplitude*decay*(1.0 + alpha*pm->time);
    velocity_derivative = -amplitude*decay*SQR(alpha)*pm->time;
  }
  const Real enthalpy_density = 1.0 + gamma;
  const Real exact_stress = enthalpy_density*velocity_derivative/wave_number;
  Real velocity_mode = 0.0;
  Real stress_mode = 0.0;
  Real density_sum = 0.0;
  Real internal_energy_sum = 0.0;
  Real velocity_norm_sum = 0.0;
  Real shear_norm_sum = 0.0;
  Real max_constraint_error = 0.0;
  int ncells = 0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      for (int j = indcs.js; j <= indcs.je; ++j) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const Real x = size_h(m).x1min
                       + (static_cast<Real>(i-indcs.is) + 0.5)*size_h(m).dx1;
          const Real y = size_h(m).x2min
                       + (static_cast<Real>(j-indcs.js) + 0.5)*size_h(m).dx2;
          const Real z = size_h(m).x3min
                       + (static_cast<Real>(k-indcs.ks) + 0.5)*size_h(m).dx3;
          const Real coordinate = (wave_direction == 1) ? x
              : ((wave_direction == 2) ? y : z);
          const Real u1 = w(m, IVX, k, j, i);
          const Real u2 = w(m, IVY, k, j, i);
          const Real u3 = w(m, IVZ, k, j, i);
          const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
          const Real d = w(m, IDN, k, j, i)*lor;
          srrmhd::ShearStress pi;
          pi.p11 = visc_u(m, srrmhd::IVP11, k, j, i)/d;
          pi.p22 = visc_u(m, srrmhd::IVP22, k, j, i)/d;
          pi.p33 = visc_u(m, srrmhd::IVP33, k, j, i)/d;
          pi.p12 = visc_u(m, srrmhd::IVP12, k, j, i)/d;
          pi.p13 = visc_u(m, srrmhd::IVP13, k, j, i)/d;
          pi.p23 = visc_u(m, srrmhd::IVP23, k, j, i)/d;
          velocity_mode += 2.0*w(m, velocity_component, k, j, i)
                           *sin(wave_number*coordinate);
          const Real stress_value = visc_u(m, stress_component, k, j, i)/d;
          stress_mode += 2.0*stress_value*cos(wave_number*coordinate);
          density_sum += w(m, IDN, k, j, i);
          internal_energy_sum += w(m, IEN, k, j, i);
          velocity_norm_sum += SQR(u1) + SQR(u2) + SQR(u3);
          shear_norm_sum += SQR(pi.p11) + SQR(pi.p22) + SQR(pi.p33)
                          + 2.0*(SQR(pi.p12) + SQR(pi.p13) + SQR(pi.p23));
          Real orthogonality_error, trace_error;
          srrmhd::ShearConstraintErrors(u1, u2, u3, pi,
                                        orthogonality_error, trace_error);
          max_constraint_error = std::max(max_constraint_error,
              std::max(orthogonality_error, trace_error));
          ++ncells;
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  Real sum_diagnostics[6] = {velocity_mode, stress_mode, density_sum,
                             internal_energy_sum, velocity_norm_sum,
                             shear_norm_sum};
  MPI_Allreduce(MPI_IN_PLACE, sum_diagnostics, 6, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_constraint_error, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ncells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  velocity_mode = sum_diagnostics[0];
  stress_mode = sum_diagnostics[1];
  density_sum = sum_diagnostics[2];
  internal_energy_sum = sum_diagnostics[3];
  velocity_norm_sum = sum_diagnostics[4];
  shear_norm_sum = sum_diagnostics[5];
#endif
  velocity_mode /= static_cast<Real>(ncells);
  stress_mode /= static_cast<Real>(ncells);
  const Real velocity_error = fabs(velocity_mode - exact_velocity)/amplitude;
  const Real stress_error = fabs(stress_mode - exact_stress)/
      (enthalpy_density*amplitude);
  if (global_variable::my_rank == 0) {
    std::string filename = "rsrmhd_viscous_telegraph-errs.dat";
    if (wave_direction == 2) filename = "rsrmhd_viscous_multid_x2-errs.dat";
    if (wave_direction == 3) filename = "rsrmhd_viscous_multid_x3-errs.dat";
    if (pin->DoesParameterExist("problem", "viscous_diagnostic_name")) {
      const std::string diagnostic_name =
          pin->GetString("problem", "viscous_diagnostic_name");
      if (diagnostic_name.compare("none") != 0) filename = diagnostic_name + "-errs.dat";
    }
    const Real inv_cells = 1.0/static_cast<Real>(ncells);
    std::ofstream file(filename);
    file << std::setprecision(17) << velocity_error << " " << stress_error << " "
         << velocity_mode << " " << exact_velocity << " " << stress_mode << " "
         << exact_stress << " " << pmhd->dtnew << " "
         << density_sum*inv_cells << " " << internal_energy_sum*inv_cells << " "
         << sqrt(velocity_norm_sum*inv_cells) << " "
         << sqrt(shear_norm_sum*inv_cells) << " " << max_constraint_error << " "
         << pm->time << " " << pm->ncycle << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Measure nonlinear periodic shear smoothing and global conservation.

void SRRMHDViscousShearLayerErrors(ParameterInput *pin, Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u0);
  auto size_h = pm->pmb_pack->pmb->mb_size.h_view;
  const Real gamma = pin->GetReal("mhd", "gamma");
  const Real wave_number = 2.0*M_PI;
  Real velocity_mode = 0.0;
  Real third_harmonic = 0.0;
  Real velocity_norm_sum = 0.0;
  Real shear_norm_sum = 0.0;
  Real density_sum = 0.0;
  Real internal_energy_sum = 0.0;
  Real conserved_energy_sum = 0.0;
  Real momentum1_sum = 0.0;
  Real momentum2_sum = 0.0;
  Real momentum3_sum = 0.0;
  Real max_constraint_error = 0.0;
  Real max_inverse_reynolds = 0.0;
  int ncells = 0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      for (int j = indcs.js; j <= indcs.je; ++j) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const Real x = size_h(m).x1min
                       + (static_cast<Real>(i-indcs.is) + 0.5)*size_h(m).dx1;
          const Real u1 = w(m, IVX, k, j, i);
          const Real u2 = w(m, IVY, k, j, i);
          const Real u3 = w(m, IVZ, k, j, i);
          const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
          const Real d = w(m, IDN, k, j, i)*lor;
          srrmhd::ShearStress pi;
          pi.p11 = visc_u(m, srrmhd::IVP11, k, j, i)/d;
          pi.p22 = visc_u(m, srrmhd::IVP22, k, j, i)/d;
          pi.p33 = visc_u(m, srrmhd::IVP33, k, j, i)/d;
          pi.p12 = visc_u(m, srrmhd::IVP12, k, j, i)/d;
          pi.p13 = visc_u(m, srrmhd::IVP13, k, j, i)/d;
          pi.p23 = visc_u(m, srrmhd::IVP23, k, j, i)/d;
          const Real shear_norm = srrmhd::ShearInvariantNorm(u1, u2, u3, pi);
          const Real enthalpy_density = w(m, IDN, k, j, i)
                                      + gamma*w(m, IEN, k, j, i);
          velocity_mode += 2.0*u2*sin(wave_number*x);
          third_harmonic += 2.0*u2*sin(3.0*wave_number*x);
          velocity_norm_sum += SQR(u1) + SQR(u2) + SQR(u3);
          shear_norm_sum += SQR(shear_norm);
          density_sum += w(m, IDN, k, j, i);
          internal_energy_sum += w(m, IEN, k, j, i);
          conserved_energy_sum += u(m, IEN, k, j, i);
          momentum1_sum += u(m, IM1, k, j, i);
          momentum2_sum += u(m, IM2, k, j, i);
          momentum3_sum += u(m, IM3, k, j, i);
          Real orthogonality_error, trace_error;
          srrmhd::ShearConstraintErrors(u1, u2, u3, pi,
                                        orthogonality_error, trace_error);
          max_constraint_error = std::max(max_constraint_error,
              std::max(orthogonality_error, trace_error));
          max_inverse_reynolds = std::max(max_inverse_reynolds,
              shear_norm/enthalpy_density);
          ++ncells;
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  Real sum_diagnostics[10] = {velocity_mode, third_harmonic, velocity_norm_sum,
      shear_norm_sum, density_sum, internal_energy_sum, conserved_energy_sum,
      momentum1_sum, momentum2_sum, momentum3_sum};
  Real max_diagnostics[2] = {max_constraint_error, max_inverse_reynolds};
  MPI_Allreduce(MPI_IN_PLACE, sum_diagnostics, 10, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, max_diagnostics, 2, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ncells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  velocity_mode = sum_diagnostics[0];
  third_harmonic = sum_diagnostics[1];
  velocity_norm_sum = sum_diagnostics[2];
  shear_norm_sum = sum_diagnostics[3];
  density_sum = sum_diagnostics[4];
  internal_energy_sum = sum_diagnostics[5];
  conserved_energy_sum = sum_diagnostics[6];
  momentum1_sum = sum_diagnostics[7];
  momentum2_sum = sum_diagnostics[8];
  momentum3_sum = sum_diagnostics[9];
  max_constraint_error = max_diagnostics[0];
  max_inverse_reynolds = max_diagnostics[1];
#endif
  if (global_variable::my_rank == 0) {
    const Real inv_cells = 1.0/static_cast<Real>(ncells);
    std::string filename = "rsrmhd_viscous_shear_layer-errs.dat";
    if (pin->DoesParameterExist("problem", "viscous_diagnostic_name")) {
      const std::string diagnostic_name =
          pin->GetString("problem", "viscous_diagnostic_name");
      if (diagnostic_name.compare("none") != 0) filename = diagnostic_name + "-errs.dat";
    }
    std::ofstream file(filename);
    file << std::setprecision(17) << velocity_mode*inv_cells << " "
         << third_harmonic*inv_cells << " "
         << sqrt(velocity_norm_sum*inv_cells) << " "
         << sqrt(shear_norm_sum*inv_cells) << " " << density_sum*inv_cells << " "
         << internal_energy_sum*inv_cells << " " << conserved_energy_sum*inv_cells
         << " " << momentum1_sum*inv_cells << " " << momentum2_sum*inv_cells << " "
         << momentum3_sum*inv_cells << " " << max_constraint_error << " "
         << max_inverse_reynolds << " " << pm->time << " " << pm->ncycle << std::endl;
  }
  if (pin->DoesParameterExist("problem", "viscous_profile_name")) {
    const std::string profile_name = pin->GetString("problem", "viscous_profile_name");
    if (profile_name.compare("none") != 0) {
      std::string filename = profile_name;
      if (global_variable::nranks > 1) {
        filename += "-rank" + std::to_string(global_variable::my_rank);
      }
      std::ofstream file(filename + "-profile.dat");
      file << "# x u2 pi12\n" << std::setprecision(17);
      for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const Real x = size_h(m).x1min
                       + (static_cast<Real>(i-indcs.is) + 0.5)*size_h(m).dx1;
          const Real u1 = w(m, IVX, indcs.ks, indcs.js, i);
          const Real u2 = w(m, IVY, indcs.ks, indcs.js, i);
          const Real u3 = w(m, IVZ, indcs.ks, indcs.js, i);
          const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
          const Real d = w(m, IDN, indcs.ks, indcs.js, i)*lor;
          const Real pi12 = visc_u(m, srrmhd::IVP12, indcs.ks, indcs.js, i)/d;
          file << x << " " << u2 << " " << pi12 << "\n";
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Record transverse growth and enstrophy for the two-dimensional KH problem.

void SRRMHDViscousKHHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 4;
  pdata->label[0] = "volume";
  pdata->label[1] = "vy2_vol";
  pdata->label[2] = "ens_vol";
  pdata->label[3] = "mode_vol";

  auto &w = pm->pmb_pack->pmhd->w0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int ncell = pm->pmb_pack->nmb_thispack*nkji;
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("kh_history", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &result) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k-ks)*nji)/nx1 + js;
    const int i = idx - m*nkji - (k-ks)*nji - (j-js)*nx1 + is;
    const Real x = size.d_view(m).x1min
                 + (static_cast<Real>(i-is) + 0.5)*size.d_view(m).dx1;
    const Real y = size.d_view(m).x2min
                 + (static_cast<Real>(j-js) + 0.5)*size.d_view(m).dx2;
    auto velocity = KOKKOS_LAMBDA(const int jj, const int ii, const int component) {
      const Real u1 = w(m, IVX, k, jj, ii);
      const Real u2 = w(m, IVY, k, jj, ii);
      const Real u3 = w(m, IVZ, k, jj, ii);
      const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
      return w(m, component, k, jj, ii)/lor;
    };
    const Real vy = velocity(j, i, IVY);
    const Real dvy_dx = (velocity(j, i+1, IVY) - velocity(j, i-1, IVY))
                       /(2.0*size.d_view(m).dx1);
    const Real dvx_dy = (velocity(j+1, i, IVX) - velocity(j-1, i, IVX))
                       /(2.0*size.d_view(m).dx2);
    const Real omega = dvy_dx - dvx_dy;
    const Real layer = exp(-SQR((y-0.25)/0.05)) + exp(-SQR((y-0.75)/0.05));
    const Real volume = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    array_sum::GlobalSum cell;
    cell.the_array[0] = volume;
    cell.the_array[1] = volume*SQR(vy);
    cell.the_array[2] = volume*SQR(omega);
    cell.the_array[3] = volume*vy*sin(2.0*M_PI*x)*layer;
    for (int n = 4; n < NHISTORY_VARIABLES; ++n) cell.the_array[n] = 0.0;
    result += cell;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));
  Kokkos::fence();
  for (int n = 0; n < NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = sum.the_array[n];
  }
}

//----------------------------------------------------------------------------------------
//! \brief Measure conservation and dump final fields for the two-dimensional KH problem.

void SRRMHDViscousKHErrors(ParameterInput *pin, Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u0);
  auto size_h = pm->pmb_pack->pmb->mb_size.h_view;
  const Real gamma = pin->GetReal("mhd", "gamma");
  Real volume_sum = 0.0;
  Real vy2_sum = 0.0;
  Real enstrophy_sum = 0.0;
  Real energy_sum = 0.0;
  Real momentum1_sum = 0.0;
  Real momentum2_sum = 0.0;
  Real momentum3_sum = 0.0;
  Real max_constraint_error = 0.0;
  Real max_inverse_reynolds = 0.0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    const Real volume = size_h(m).dx1*size_h(m).dx2*size_h(m).dx3;
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      for (int j = indcs.js; j <= indcs.je; ++j) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          auto velocity = [&](const int jj, const int ii, const int component) {
            const Real uu1 = w(m, IVX, k, jj, ii);
            const Real uu2 = w(m, IVY, k, jj, ii);
            const Real uu3 = w(m, IVZ, k, jj, ii);
            const Real lor = sqrt(1.0 + SQR(uu1) + SQR(uu2) + SQR(uu3));
            return w(m, component, k, jj, ii)/lor;
          };
          const Real u1 = w(m, IVX, k, j, i);
          const Real u2 = w(m, IVY, k, j, i);
          const Real u3 = w(m, IVZ, k, j, i);
          const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
          const Real d = w(m, IDN, k, j, i)*lor;
          srrmhd::ShearStress pi;
          pi.p11 = visc_u(m, srrmhd::IVP11, k, j, i)/d;
          pi.p22 = visc_u(m, srrmhd::IVP22, k, j, i)/d;
          pi.p33 = visc_u(m, srrmhd::IVP33, k, j, i)/d;
          pi.p12 = visc_u(m, srrmhd::IVP12, k, j, i)/d;
          pi.p13 = visc_u(m, srrmhd::IVP13, k, j, i)/d;
          pi.p23 = visc_u(m, srrmhd::IVP23, k, j, i)/d;
          const Real shear_norm = srrmhd::ShearInvariantNorm(u1, u2, u3, pi);
          const Real enthalpy_density = w(m, IDN, k, j, i)
                                      + gamma*w(m, IEN, k, j, i);
          const Real vy = velocity(j, i, IVY);
          const Real dvy_dx = (velocity(j, i+1, IVY) - velocity(j, i-1, IVY))
                             /(2.0*size_h(m).dx1);
          const Real dvx_dy = (velocity(j+1, i, IVX) - velocity(j-1, i, IVX))
                             /(2.0*size_h(m).dx2);
          const Real omega = dvy_dx - dvx_dy;
          volume_sum += volume;
          vy2_sum += volume*SQR(vy);
          enstrophy_sum += volume*SQR(omega);
          energy_sum += volume*u(m, IEN, k, j, i);
          momentum1_sum += volume*u(m, IM1, k, j, i);
          momentum2_sum += volume*u(m, IM2, k, j, i);
          momentum3_sum += volume*u(m, IM3, k, j, i);
          Real orthogonality_error, trace_error;
          srrmhd::ShearConstraintErrors(u1, u2, u3, pi,
                                        orthogonality_error, trace_error);
          max_constraint_error = std::max(max_constraint_error,
              std::max(orthogonality_error, trace_error));
          max_inverse_reynolds = std::max(max_inverse_reynolds,
              shear_norm/enthalpy_density);
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  Real sums[7] = {volume_sum, vy2_sum, enstrophy_sum, energy_sum,
                  momentum1_sum, momentum2_sum, momentum3_sum};
  Real maxima[2] = {max_constraint_error, max_inverse_reynolds};
  MPI_Allreduce(MPI_IN_PLACE, sums, 7, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, maxima, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  volume_sum = sums[0];
  vy2_sum = sums[1];
  enstrophy_sum = sums[2];
  energy_sum = sums[3];
  momentum1_sum = sums[4];
  momentum2_sum = sums[5];
  momentum3_sum = sums[6];
  max_constraint_error = maxima[0];
  max_inverse_reynolds = maxima[1];
#endif
  if (global_variable::my_rank == 0) {
    std::string filename = "rsrmhd_viscous_kh-errs.dat";
    if (pin->DoesParameterExist("problem", "viscous_diagnostic_name")) {
      const std::string name = pin->GetString("problem", "viscous_diagnostic_name");
      if (name.compare("none") != 0) filename = name + "-errs.dat";
    }
    const Real inv_volume = 1.0/volume_sum;
    std::ofstream file(filename);
    file << std::setprecision(17) << sqrt(vy2_sum*inv_volume) << " "
         << sqrt(enstrophy_sum*inv_volume) << " " << energy_sum*inv_volume << " "
         << momentum1_sum*inv_volume << " " << momentum2_sum*inv_volume << " "
         << momentum3_sum*inv_volume << " " << max_constraint_error << " "
         << max_inverse_reynolds << " " << pm->time << " " << pm->ncycle << "\n";
  }
  if (pin->DoesParameterExist("problem", "viscous_profile_name")) {
    const std::string name = pin->GetString("problem", "viscous_profile_name");
    if (name.compare("none") != 0) {
      std::string filename = name;
      if (global_variable::nranks > 1) {
        filename += "-rank" + std::to_string(global_variable::my_rank);
      }
      std::ofstream file(filename + "-profile.dat");
      file << "# x y vx vy rho eint pi11 pi22 pi12 omega_z\n"
           << std::setprecision(17);
      for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
        for (int j = indcs.js; j <= indcs.je; ++j) {
          for (int i = indcs.is; i <= indcs.ie; ++i) {
            const int k = indcs.ks;
            auto velocity = [&](const int jj, const int ii, const int component) {
              const Real u1 = w(m, IVX, k, jj, ii);
              const Real u2 = w(m, IVY, k, jj, ii);
              const Real u3 = w(m, IVZ, k, jj, ii);
              const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
              return w(m, component, k, jj, ii)/lor;
            };
            const Real x = size_h(m).x1min
                         + (static_cast<Real>(i-indcs.is) + 0.5)*size_h(m).dx1;
            const Real y = size_h(m).x2min
                         + (static_cast<Real>(j-indcs.js) + 0.5)*size_h(m).dx2;
            const Real u1 = w(m, IVX, k, j, i);
            const Real u2 = w(m, IVY, k, j, i);
            const Real u3 = w(m, IVZ, k, j, i);
            const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
            const Real d = w(m, IDN, k, j, i)*lor;
            const Real vx = velocity(j, i, IVX);
            const Real vy = velocity(j, i, IVY);
            const Real dvy_dx = (velocity(j, i+1, IVY) - velocity(j, i-1, IVY))
                               /(2.0*size_h(m).dx1);
            const Real dvx_dy = (velocity(j+1, i, IVX) - velocity(j-1, i, IVX))
                               /(2.0*size_h(m).dx2);
            file << x << " " << y << " " << vx << " " << vy << " "
                 << w(m, IDN, k, j, i) << " " << w(m, IEN, k, j, i) << " "
                 << visc_u(m, srrmhd::IVP11, k, j, i)/d << " "
                 << visc_u(m, srrmhd::IVP22, k, j, i)/d << " "
                 << visc_u(m, srrmhd::IVP12, k, j, i)/d << " "
                 << dvy_dx-dvx_dy << "\n";
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Measure phase-resolved transverse Fourier amplitudes on a boosted background.

void SRRMHDViscousBoostedErrors(ParameterInput *pin, Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto size_h = pm->pmb_pack->pmb->mb_size.h_view;
  const Real wave_number = 2.0*M_PI;
  Real velocity_cos = 0.0;
  Real velocity_sin = 0.0;
  Real stress_cos = 0.0;
  Real stress_sin = 0.0;
  Real mean_u1 = 0.0;
  int ncells = 0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      const Real x = size_h(m).x1min
                   + (static_cast<Real>(i-indcs.is) + 0.5)*size_h(m).dx1;
      const Real cosine = cos(wave_number*x);
      const Real sine = sin(wave_number*x);
      const Real u1 = w(m, IVX, indcs.ks, indcs.js, i);
      const Real u2 = w(m, IVY, indcs.ks, indcs.js, i);
      const Real u3 = w(m, IVZ, indcs.ks, indcs.js, i);
      const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
      const Real d = w(m, IDN, indcs.ks, indcs.js, i)*lor;
      const Real pi12 = visc_u(m, srrmhd::IVP12, indcs.ks, indcs.js, i)/d;
      velocity_cos += 2.0*u2*cosine;
      velocity_sin += 2.0*u2*sine;
      stress_cos += 2.0*pi12*cosine;
      stress_sin += 2.0*pi12*sine;
      mean_u1 += u1;
      ++ncells;
    }
  }
  const Real inv_cells = 1.0/static_cast<Real>(ncells);
  velocity_cos *= inv_cells;
  velocity_sin *= inv_cells;
  stress_cos *= inv_cells;
  stress_sin *= inv_cells;
  mean_u1 *= inv_cells;
  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_viscous_boosted-amps.dat");
    file << std::setprecision(17) << velocity_cos << " " << velocity_sin << " "
         << stress_cos << " " << stress_sin << " " << mean_u1 << " "
         << pin->GetReal("problem", "boost_velocity") << " " << pm->time
         << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Measure the Fourier amplitudes of a linear longitudinal viscous-sound mode.

void SRRMHDViscousLongitudinalErrors(ParameterInput *pin, Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto size_h = pm->pmb_pack->pmb->mb_size.h_view;
  const Real gamma = pin->GetReal("mhd", "gamma");
  const Real pressure0 = gamma - 1.0;
  const Real wave_number = 2.0*M_PI;
  Real velocity_mode = 0.0;
  Real pressure_mode = 0.0;
  Real stress_mode = 0.0;
  int ncells = 0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      const Real x = size_h(m).x1min
                   + (static_cast<Real>(i-indcs.is) + 0.5)*size_h(m).dx1;
      const Real u1 = w(m, IVX, indcs.ks, indcs.js, i);
      const Real u2 = w(m, IVY, indcs.ks, indcs.js, i);
      const Real u3 = w(m, IVZ, indcs.ks, indcs.js, i);
      const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
      const Real d = w(m, IDN, indcs.ks, indcs.js, i)*lor;
      const Real pressure = (gamma - 1.0)*w(m, IEN, indcs.ks, indcs.js, i);
      velocity_mode += 2.0*u1*sin(wave_number*x);
      pressure_mode += 2.0*(pressure - pressure0)*cos(wave_number*x);
      stress_mode += 2.0*visc_u(m, srrmhd::IVP11, indcs.ks, indcs.js, i)
                     *cos(wave_number*x)/d;
      ++ncells;
    }
  }
  velocity_mode /= static_cast<Real>(ncells);
  pressure_mode /= static_cast<Real>(ncells);
  stress_mode /= static_cast<Real>(ncells);
  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_viscous_longitudinal-amps.dat");
    file << std::setprecision(17) << velocity_mode << " " << pressure_mode << " "
         << stress_mode << " " << pm->time << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Measure one-period conservative advection of a transverse shear component.

void SRRMHDViscousTransportErrors(ParameterInput *pin, Mesh *pm) {
  (void)pin;
  auto &indcs = pm->mb_indcs;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
  auto flux = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->uflx.x1f);
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto size_h = size.h_view;
  constexpr Real v1 = 0.4;
  const Real lor = 1.0/sqrt(1.0 - SQR(v1));
  constexpr Real amplitude = 0.01;
  Real l1_error = 0.0;
  Real max_error = 0.0;
  Real flux_error = 0.0;
  int ncells = 0;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      const Real x = size_h(m).x1min
                   + (static_cast<Real>(i-indcs.is) + 0.5)*size_h(m).dx1;
      const Real exact = lor*amplitude*sin(2.0*M_PI*(x - v1*pm->time));
      const Real error = fabs(visc_u(m, srrmhd::IVP23, indcs.ks, indcs.js, i) - exact);
      l1_error += error;
      max_error = std::max(max_error, error);
      flux_error = std::max(flux_error,
          fabs(flux(m, IM2, indcs.ks, indcs.js, i) - 0.002));
      ++ncells;
    }
  }
  l1_error /= static_cast<Real>(ncells);
  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_viscous_transport-errs.dat");
    file << std::setprecision(17) << l1_error << " " << max_error << " "
         << flux_error << std::endl;
  }
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
  SRRMHDForcingAlgebraErrors();
  SRRMHDViscosityAlgebraErrors(pm);
  SRRMHDViscosityKinematicsErrors(pm);
  SRRMHDViscousImplicitErrors(pm);
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDForcingAlgebraErrors()
//! \brief Verify mechanical four-source orthogonality and the cold Newtonian limit.

void SRRMHDForcingAlgebraErrors() {
  constexpr int ncases = 6;
  constexpr int nerrors = 7;
  DvceArray2D<Real> result("rsrmhd_forcing_algebra_test", ncases, nerrors);

  Kokkos::parallel_for("rsrmhd_forcing_algebra_test",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncases), KOKKOS_LAMBDA(const int n) {
    const Real u1_values[ncases] = {0.0, 0.2, -0.7, 1.4, -2.1, 6.0};
    const Real u2_values[ncases] = {0.0, -0.4, 0.5, -1.1, 0.8, -3.0};
    const Real u3_values[ncases] = {0.0, 0.3, 1.2, 0.6, -1.7, 4.0};
    const Real a1_values[ncases] = {1.0, -0.3, 0.7, 2.0, -1.2, 0.05};
    const Real a2_values[ncases] = {-0.4, 0.8, -1.1, 0.2, 0.6, -0.9};
    const Real a3_values[ncases] = {0.2, 1.1, 0.3, -0.7, 1.4, 0.5};
    const Real enthalpy_values[ncases] = {1.0, 1.3, 0.8, 4.0, 5.0, 0.01};
    const Real amplitude_values[ncases] = {0.0, 1.0e-4, 0.2, 2.0, 10.0, 0.7};
    const Real u1 = u1_values[n];
    const Real u2 = u2_values[n];
    const Real u3 = u3_values[n];
    const Real a1 = a1_values[n];
    const Real a2 = a2_values[n];
    const Real a3 = a3_values[n];
    const Real enthalpy = enthalpy_values[n];
    const Real amplitude = amplitude_values[n];
    const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));

    const srrmhd::ProperFourAcceleration acceleration =
        srrmhd::CompleteProperAcceleration(u1, u2, u3, a1, a2, a3);
    const srrmhd::MechanicalFourSource source = srrmhd::MechanicalForcingSource(
        u1, u2, u3, enthalpy, amplitude, a1, a2, a3);
    const Real expected_a0 = (u1*a1 + u2*a2 + u3*a3)/lor;
    const Real force_scale = amplitude*enthalpy;

    result(n, 0) = fabs(-lor*acceleration.a0 + u1*acceleration.a1
                        + u2*acceleration.a2 + u3*acceleration.a3);
    result(n, 1) = fabs(-lor*source.g0 + u1*source.g1 + u2*source.g2
                        + u3*source.g3);
    result(n, 2) = fabs(acceleration.a0 - expected_a0);
    result(n, 3) = fmax(fabs(acceleration.a1 - a1),
                        fmax(fabs(acceleration.a2 - a2),
                             fabs(acceleration.a3 - a3)));
    result(n, 4) = fmax(fabs(source.g1 - force_scale*a1),
                        fmax(fabs(source.g2 - force_scale*a2),
                             fabs(source.g3 - force_scale*a3)));
    result(n, 5) = fabs(source.g0 - force_scale*expected_a0);

    const Real rho = 0.7 + 0.2*n;
    const Real cold_amplitude = 0.3 + 0.1*n;
    const Real epsilon = 1.0e-7*(n + 1);
    const Real cold_u1 = epsilon;
    const Real cold_u2 = -0.5*epsilon;
    const Real cold_u3 = 0.25*epsilon;
    const srrmhd::MechanicalFourSource cold_source =
        srrmhd::MechanicalForcingSource(cold_u1, cold_u2, cold_u3, rho,
                                        cold_amplitude, a1, a2, a3);
    const Real newtonian_power = cold_amplitude*rho*
        (cold_u1*a1 + cold_u2*a2 + cold_u3*a3);
    Real newtonian_error = fabs(cold_source.g0 - newtonian_power);
    newtonian_error = fmax(newtonian_error,
                           fabs(cold_source.g1 - cold_amplitude*rho*a1));
    newtonian_error = fmax(newtonian_error,
                           fabs(cold_source.g2 - cold_amplitude*rho*a2));
    newtonian_error = fmax(newtonian_error,
                           fabs(cold_source.g3 - cold_amplitude*rho*a3));
    result(n, 6) = newtonian_error;
  });

  auto r = Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
  Real errors[nerrors] = {};
  for (int n = 0; n < ncases; ++n) {
    for (int e = 0; e < nerrors; ++e) {
      errors[e] = std::max(errors[e], r(n, e));
    }
  }

  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_forcing_algebra-errs.dat");
    file << std::setprecision(17);
    for (int e = 0; e < nerrors; ++e) file << errors[e] << " ";
    file << std::endl;
  }
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

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDViscosityAlgebraErrors()
//! \brief Verify projected shear constraints, conserved contributions, and relaxation.

void SRRMHDViscosityAlgebraErrors(Mesh *pm) {
  constexpr int ncases = 5;
  DvceArray2D<Real> result("rsrmhd_viscosity_algebra_test", ncases, 6);

  Kokkos::parallel_for("rsrmhd_viscosity_algebra_test",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncases), KOKKOS_LAMBDA(const int n) {
    const Real u1_values[ncases] = {0.0, 0.2, -0.7, 1.4, -2.1};
    const Real u2_values[ncases] = {0.0, -0.4, 0.5, -1.1, 0.8};
    const Real u3_values[ncases] = {0.0, 0.3, 1.2, 0.6, -1.7};
    const Real kappa_values[ncases] = {0.0, 1.0e-4, 0.2, 10.0, 1.0e4};
    const Real u1 = u1_values[n];
    const Real u2 = u2_values[n];
    const Real u3 = u3_values[n];

    srrmhd::ShearStress raw;
    raw.p11 = 0.31 + 0.07*n;
    raw.p22 = -0.18 + 0.04*n;
    raw.p33 = 0.27 - 0.03*n;
    raw.p12 = -0.11 + 0.02*n;
    raw.p13 = 0.09 - 0.015*n;
    raw.p23 = 0.14 + 0.01*n;
    const srrmhd::ShearStress pi =
        srrmhd::ProjectShearStress(u1, u2, u3, raw);

    Real orthogonality_error, trace_error;
    srrmhd::ShearConstraintErrors(u1, u2, u3, pi, orthogonality_error,
                                  trace_error);
    result(n, 0) = orthogonality_error;
    result(n, 1) = trace_error;

    const srrmhd::ShearStress projected_twice =
        srrmhd::ProjectShearStress(u1, u2, u3, pi);
    Real idempotence_error = 0.0;
    idempotence_error = fmax(idempotence_error,
                             fabs(projected_twice.p11 - pi.p11));
    idempotence_error = fmax(idempotence_error,
                             fabs(projected_twice.p22 - pi.p22));
    idempotence_error = fmax(idempotence_error,
                             fabs(projected_twice.p33 - pi.p33));
    idempotence_error = fmax(idempotence_error,
                             fabs(projected_twice.p12 - pi.p12));
    idempotence_error = fmax(idempotence_error,
                             fabs(projected_twice.p13 - pi.p13));
    idempotence_error = fmax(idempotence_error,
                             fabs(projected_twice.p23 - pi.p23));
    result(n, 2) = idempotence_error;

    srrmhd::SRRMHDPrim1D w;
    w.d = 0.8 + 0.1*n;
    w.vx = u1;
    w.vy = u2;
    w.vz = u3;
    w.e = 0.5 + 0.08*n;
    w.ex = 0.1;
    w.ey = -0.2;
    w.ez = 0.3;
    w.bx = 0.7;
    w.by = -0.4;
    w.bz = 0.9;
    srrmhd::SRRMHDCons1D baseline;
    srrmhd::SingleP2C_IdealSRRMHD(w, 5.0/3.0, baseline);
    srrmhd::SRRMHDCons1D roundtrip = baseline;
    srrmhd::AddShearToConserved(u1, u2, u3, pi, roundtrip);
    srrmhd::RemoveShearFromConserved(u1, u2, u3, pi, roundtrip);
    Real conserved_error = 0.0;
    conserved_error = fmax(conserved_error, fabs(roundtrip.mx - baseline.mx));
    conserved_error = fmax(conserved_error, fabs(roundtrip.my - baseline.my));
    conserved_error = fmax(conserved_error, fabs(roundtrip.mz - baseline.mz));
    conserved_error = fmax(conserved_error, fabs(roundtrip.e - baseline.e));
    result(n, 3) = conserved_error;

    srrmhd::ShearStress raw_ns;
    raw_ns.p11 = -0.17 + 0.01*n;
    raw_ns.p22 = 0.23 - 0.02*n;
    raw_ns.p33 = -0.06 + 0.015*n;
    raw_ns.p12 = 0.12;
    raw_ns.p13 = -0.08;
    raw_ns.p23 = 0.05;
    const srrmhd::ShearStress pi_ns =
        srrmhd::ProjectShearStress(u1, u2, u3, raw_ns);
    const Real kappa = kappa_values[n];
    const Real inv = 1.0/(1.0 + kappa);
    const srrmhd::ShearStress relaxed = srrmhd::ImplicitShearRelaxation(
        u1, u2, u3, pi, pi_ns, kappa);
    Real relaxation_error = 0.0;
    relaxation_error = fmax(relaxation_error,
        fabs(relaxed.p11 - (pi.p11 + kappa*pi_ns.p11)*inv));
    relaxation_error = fmax(relaxation_error,
        fabs(relaxed.p22 - (pi.p22 + kappa*pi_ns.p22)*inv));
    relaxation_error = fmax(relaxation_error,
        fabs(relaxed.p33 - (pi.p33 + kappa*pi_ns.p33)*inv));
    relaxation_error = fmax(relaxation_error,
        fabs(relaxed.p12 - (pi.p12 + kappa*pi_ns.p12)*inv));
    relaxation_error = fmax(relaxation_error,
        fabs(relaxed.p13 - (pi.p13 + kappa*pi_ns.p13)*inv));
    relaxation_error = fmax(relaxation_error,
        fabs(relaxed.p23 - (pi.p23 + kappa*pi_ns.p23)*inv));
    result(n, 4) = relaxation_error;

    Real relaxed_orthogonality, relaxed_trace;
    srrmhd::ShearConstraintErrors(u1, u2, u3, relaxed,
                                  relaxed_orthogonality, relaxed_trace);
    result(n, 5) = fmax(relaxed_orthogonality, relaxed_trace);
  });

  auto r = Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
  Real errors[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int n = 0; n < ncases; ++n) {
    for (int e = 0; e < 6; ++e) errors[e] = std::max(errors[e], r(n, e));
  }

  auto data = pm->pmb_pack->pmhd->relativistic_viscosity_data;
  const Real causal_margin =
      srrmhd::LinearViscosityCausalityMargin(5.0/3.0, data);
  if (global_variable::my_rank == 0) {
    std::ofstream file("rsrmhd_viscosity_algebra-errs.dat");
    file << std::setprecision(17);
    for (int e = 0; e < 6; ++e) file << errors[e] << " ";
    file << (data.enabled ? 1 : 0) << " " << data.nu << " " << data.tau << " "
         << data.chi_max << " " << causal_margin << std::endl;
  }
}
