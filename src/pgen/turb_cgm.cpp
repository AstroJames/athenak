//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_cgm.cpp
//! \brief Stratified CGM turbulence problem generator based on Wibking, Voit & O'Shea
//!        (2024), arXiv:2410.03886.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "srcterms/turb_driver.hpp"
#include "units/units.hpp"
#include "utils/spectral_ic_gen.hpp"

namespace {

struct TcgParams {
  Real temp0 = 1.0e7;
  Real temp_floor = 1.0e6;
  Real theta0 = 1.0;
  Real theta_floor = 0.1;
  Real z0 = 50.0;
  Real h = 5.0;
  Real xmu = 0.6;
  Real hydrogen_mass_fraction = 0.75;
  Real lambda0_cgs = 1.0e-22;
  Real tcool_tff_z0 = 4.0;
  Real rho_z0 = 1.0;
  Real cooling_coef = 0.0;
  Real cooling_cfl = 0.1;
  Real thermal_transition = 2.5;
  Real disk_balance_width = 5.0;
  Real disk_balance_transition = 2.5;
  Real beta_z0 = 100.0;
  Real perturb_amp = 0.0;
  int mesh_nx3 = 0;
  int thermal_mask = 0;
  int magnetic_field = 0;
  bool gravity = true;
  bool cooling = false;
  bool thermostat = false;
  bool enforce_temp_floor = true;
  bool initialized = false;
};

TcgParams tcg;
Real tcg_global_balance_target = 0.0;
Real tcg_last_turb_power = 0.0;
Real tcg_disk_turb_power = 0.0;
Real tcg_disk_balance_target = 0.0;
Real tcg_disk_balance_rate = 0.0;
Real tcg_applied_cool_src = 0.0;
Real tcg_applied_heat_src = 0.0;
Real tcg_applied_disk_src = 0.0;
Real tcg_applied_net_src = 0.0;
DualArray1D<Real> *tcg_plane_sums = nullptr;
DualArray1D<Real> *tcg_plane_balance = nullptr;

constexpr int TCG_THERMAL_MASK_TANH = 0;
constexpr int TCG_THERMAL_MASK_OUTER_TANH = 1;
constexpr int TCG_THERMAL_MASK_HARD = 2;
constexpr int TCG_MAGNETIC_NONE = 0;
constexpr int TCG_MAGNETIC_SPECTRAL = 1;

struct TcgDiagData {
  Real min_temp, max_temp, min_rho, max_rho;
  Real max_mach, max_abs_v, max_signal_speed, max_source_frac;
  Real max_signal_x, max_signal_y, max_signal_z;

  KOKKOS_INLINE_FUNCTION
  TcgDiagData() :
      min_temp(1.0e300), max_temp(-1.0e300), min_rho(1.0e300), max_rho(-1.0e300),
      max_mach(-1.0e300), max_abs_v(-1.0e300), max_signal_speed(-1.0e300),
      max_source_frac(-1.0e300), max_signal_x(0.0), max_signal_y(0.0),
      max_signal_z(0.0) {}

  KOKKOS_INLINE_FUNCTION
  TcgDiagData& operator += (const TcgDiagData &src) {
    min_temp = fmin(min_temp, src.min_temp);
    max_temp = fmax(max_temp, src.max_temp);
    min_rho = fmin(min_rho, src.min_rho);
    max_rho = fmax(max_rho, src.max_rho);
    max_mach = fmax(max_mach, src.max_mach);
    max_abs_v = fmax(max_abs_v, src.max_abs_v);
    max_source_frac = fmax(max_source_frac, src.max_source_frac);
    if (src.max_signal_speed > max_signal_speed) {
      max_signal_speed = src.max_signal_speed;
      max_signal_x = src.max_signal_x;
      max_signal_y = src.max_signal_y;
      max_signal_z = src.max_signal_z;
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  void operator += (const volatile TcgDiagData &src) volatile {
    min_temp = fmin(min_temp, src.min_temp);
    max_temp = fmax(max_temp, src.max_temp);
    min_rho = fmin(min_rho, src.min_rho);
    max_rho = fmax(max_rho, src.max_rho);
    max_mach = fmax(max_mach, src.max_mach);
    max_abs_v = fmax(max_abs_v, src.max_abs_v);
    max_source_frac = fmax(max_source_frac, src.max_source_frac);
    if (src.max_signal_speed > max_signal_speed) {
      max_signal_speed = src.max_signal_speed;
      max_signal_x = src.max_signal_x;
      max_signal_y = src.max_signal_y;
      max_signal_z = src.max_signal_z;
    }
  }
};

KOKKOS_INLINE_FUNCTION
Real TcgAbsZ(const Real z) {
  return fabs(z);
}

KOKKOS_INLINE_FUNCTION
Real TcgSgn(const Real z) {
  return (z > 0.0) - (z < 0.0);
}

KOKKOS_INLINE_FUNCTION
Real TcgGravity(const Real z, const Real theta0, const Real z0, const Real h) {
  const Real az = TcgAbsZ(z);
  if (az <= 0.0) return 0.0;
  const Real taper = tanh(az/h);
  return -TcgSgn(z)*theta0/z0*taper*taper;
}

KOKKOS_INLINE_FUNCTION
Real TcgDensityProfile(const Real z, const Real rho_z0, const Real z0, const Real h) {
  const Real az = TcgAbsZ(z);
  const Real exponent = 1.0 - az/z0 + (h/z0)*(tanh(az/h) - tanh(z0/h));
  return rho_z0*exp(exponent);
}

KOKKOS_INLINE_FUNCTION
Real TcgThermalMask(const Real z, const Real h, const Real transition, const int mode) {
  const Real az = TcgAbsZ(z);
  if (mode == TCG_THERMAL_MASK_OUTER_TANH) {
    if (az < h) return 0.0;
    if (transition <= 0.0) return 1.0;
    return tanh((az - h)/transition);
  }
  if (mode == TCG_THERMAL_MASK_HARD) return (az >= h) ? 1.0 : 0.0;
  if (transition <= 0.0) return (az >= h) ? 1.0 : 0.0;
  return 0.5*(1.0 + tanh((az - h)/transition));
}

KOKKOS_INLINE_FUNCTION
Real TcgDiskBalanceWeight(const Real z, const Real width, const Real transition) {
  if (width <= 0.0) return 0.0;
  const Real az = TcgAbsZ(z);
  if (transition <= 0.0) return (az <= width) ? 1.0 : 0.0;
  if (az >= width) return 0.0;
  if (az <= width - transition) return 1.0;
  const Real s = (width - az)/transition;
  return s*s*(3.0 - 2.0*s);
}

KOKKOS_INLINE_FUNCTION
Real TcgDiskBalanceWeightDz(const Real z, const Real width, const Real transition) {
  if (width <= 0.0 || transition <= 0.0) return 0.0;
  const Real az = TcgAbsZ(z);
  if (az <= width - transition || az >= width) return 0.0;
  const Real s = (width - az)/transition;
  const Real dwdaz = -6.0*s*(1.0 - s)/transition;
  return dwdaz*TcgSgn(z);
}

int TcgThermalMaskFromString(const std::string &mode) {
  if (mode == "tanh" || mode == "smooth_tanh") {
    return TCG_THERMAL_MASK_TANH;
  }
  if (mode == "outer_tanh" || mode == "smooth_outer_tanh") {
    return TCG_THERMAL_MASK_OUTER_TANH;
  }
  if (mode == "hard" || mode == "hard_cutoff") {
    return TCG_THERMAL_MASK_HARD;
  }
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Unknown turb_cgm thermal_mask='" << mode << "'. Valid masks are "
            << "'tanh', 'outer_tanh', and 'hard'." << std::endl;
  std::exit(EXIT_FAILURE);
}

int TcgMagneticFieldFromString(const std::string &mode) {
  if (mode == "none" || mode == "off" || mode == "zero") {
    return TCG_MAGNETIC_NONE;
  }
  if (mode == "spectral" || mode == "random" || mode == "random_vector_potential" ||
      mode == "random_divergence_free" || mode == "spectral_ic") {
    return TCG_MAGNETIC_SPECTRAL;
  }
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Unknown turb_cgm magnetic_field='" << mode << "'. Valid choices are "
            << "'none' and 'spectral'." << std::endl;
  std::exit(EXIT_FAILURE);
}

KOKKOS_INLINE_FUNCTION
int TcgGlobalK(const Real z, const Real x3min, const Real dx3, const int nx3) {
  int kg = static_cast<int>(floor((z - x3min)/dx3));
  if (kg < 0) kg = 0;
  if (kg >= nx3) kg = nx3 - 1;
  return kg;
}

KOKKOS_INLINE_FUNCTION
Real TcgBackgroundCoolingTime(const Real z, const Real rho_z0, const Real theta0,
                              const Real z0, const Real h, const Real cooling_coef,
                              const Real gm1) {
  const Real rho_bg = TcgDensityProfile(z, rho_z0, z0, h);
  const Real cooling = cooling_coef*rho_bg*rho_bg;
  if (cooling <= 0.0) return 1.0e30;
  return rho_bg*theta0/(gm1*cooling);
}

Real TcgRhoZ0FromTcoolTff(ParameterInput *pin, MeshBlockPack *pmbp, const Real temp0,
                          const Real z0_code, const Real h_code, const Real xmu,
                          const Real x_h, const Real lambda0_cgs,
                          const Real tcool_tff_z0) {
  if (pin->DoesParameterExist("problem", "rho_z0")) {
    return pin->GetReal("problem", "rho_z0");
  }

  const Real z0_cgs = z0_code*pmbp->punit->length_cgs();
  const Real h_cgs = h_code*pmbp->punit->length_cgs();
  const Real kb = units::Units::k_boltzmann_cgs;
  const Real mp = units::Units::atomic_mass_unit_cgs;
  const Real numerator = 3.0*std::pow(kb*temp0, 1.5)*std::tanh(z0_cgs/h_cgs);
  const Real denominator = 2.0*lambda0_cgs*SQR(x_h*xmu)*std::sqrt(2.0*xmu*mp)*z0_cgs;
  const Real n_z0_cgs = numerator/denominator/tcool_tff_z0;
  const Real rho_z0_cgs = n_z0_cgs*xmu*mp;
  return rho_z0_cgs/pmbp->punit->density_cgs();
}

void TcgPreparePlaneAverages(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  tcg_last_turb_power = (pmbp->pturb != nullptr) ? pmbp->pturb->last_power : 0.0;
  tcg_disk_turb_power = tcg_last_turb_power;
  tcg_global_balance_target = 0.0;
  tcg_disk_balance_target = 0.0;
  tcg_disk_balance_rate = 0.0;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  const int mesh_nx3 = tcg.mesh_nx3;
  const Real x3min_mesh = pm->mesh_size.x3min;
  const Real dx3_mesh = pm->mesh_size.dx3;
  const Real h = tcg.h;
  const Real cooling_coef = tcg.cooling_coef;
  const Real thermal_transition = tcg.thermal_transition;
  const Real disk_balance_width = tcg.disk_balance_width;
  const Real disk_balance_transition = tcg.disk_balance_transition;
  const int thermal_mask = tcg.thermal_mask;
  const bool cooling = tcg.cooling;
  DvceArray5D<Real> w0;
  if (pmbp->phydro != nullptr) {
    w0 = pmbp->phydro->w0;
  } else {
    w0 = pmbp->pmhd->w0;
  }
  auto &size = pmbp->pmb->mb_size;
  auto plane_sums = tcg_plane_sums->d_view;

  Kokkos::deep_copy(plane_sums, 0.0);
  Kokkos::parallel_for("turb_cgm_plane_sums",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real z = CellCenterX(k - ks, nx3, x3min, x3max);
    int kg = TcgGlobalK(z, x3min_mesh, dx3_mesh, mesh_nx3);
    Real rho = w0(m, IDN, k, j, i);
    Real cooling_mask = TcgThermalMask(z, h, thermal_transition, thermal_mask);
    Real disk_weight = TcgDiskBalanceWeight(z, disk_balance_width,
                                            disk_balance_transition);
    Real atm_cool_rate = cooling ? cooling_mask*cooling_coef*rho*rho : 0.0;
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Kokkos::atomic_add(&plane_sums(kg), atm_cool_rate*vol);
    Kokkos::atomic_add(&plane_sums(mesh_nx3 + kg), rho*vol);
    Kokkos::atomic_add(&plane_sums(2*mesh_nx3 + kg), disk_weight*rho*vol);
  });

  tcg_plane_sums->modify_device();
  tcg_plane_sums->sync_host();

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, tcg_plane_sums->h_view.data(), 3*mesh_nx3,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  Real total_cooling = 0.0;
  Real total_disk_balance_mass = 0.0;
  for (int k = 0; k < mesh_nx3; ++k) {
    Real mass = tcg_plane_sums->h_view(mesh_nx3 + k);
    tcg_plane_balance->h_view(k) =
        (mass > 0.0) ? tcg_plane_sums->h_view(k)/mass : 0.0;
    total_cooling += tcg_plane_sums->h_view(k);
    total_disk_balance_mass += tcg_plane_sums->h_view(2*mesh_nx3 + k);
  }
  tcg_disk_balance_target = -tcg_disk_turb_power;
  tcg_disk_balance_rate =
      (total_disk_balance_mass > 0.0) ? tcg_disk_balance_target/total_disk_balance_mass : 0.0;
  tcg_global_balance_target = total_cooling;
  tcg_plane_balance->modify_host();
  tcg_plane_balance->sync_device();
}

void TcgSourceTerms(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if ((pmbp->phydro == nullptr && pmbp->pmhd == nullptr) || !tcg.initialized) return;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  int nmb1 = nmb - 1;
  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  const bool is_mhd = (pmbp->pmhd != nullptr);
  DvceArray5D<Real> w0, u0, bcc0;
  EquationOfState *peos;
  if (pmbp->phydro != nullptr) {
    w0 = pmbp->phydro->w0;
    u0 = pmbp->phydro->u0;
    peos = pmbp->phydro->peos;
  } else {
    w0 = pmbp->pmhd->w0;
    u0 = pmbp->pmhd->u0;
    bcc0 = pmbp->pmhd->bcc0;
    peos = pmbp->pmhd->peos;
  }
  auto &size = pmbp->pmb->mb_size;
  EOS_Data eos = peos->eos_data;
  TcgPreparePlaneAverages(pm);

  const Real theta0 = tcg.theta0;
  const Real theta_floor = tcg.theta_floor;
  const Real z0 = tcg.z0;
  const Real h = tcg.h;
  const Real cooling_coef = tcg.cooling_coef;
  const Real cooling_cfl = tcg.cooling_cfl;
  const Real thermal_transition = tcg.thermal_transition;
  const Real disk_balance_width = tcg.disk_balance_width;
  const Real disk_balance_transition = tcg.disk_balance_transition;
  const bool gravity = tcg.gravity;
  const bool cooling = tcg.cooling;
  const bool thermostat = tcg.thermostat;
  const bool floor = tcg.enforce_temp_floor;
  const int thermal_mask = tcg.thermal_mask;
  const Real disk_balance_rate = tcg_disk_balance_rate;
  const Real gm1 = eos.gamma - 1.0;
  const int mesh_nx3 = tcg.mesh_nx3;
  const Real x3min_mesh = pm->mesh_size.x3min;
  const Real dx3_mesh = pm->mesh_size.dx3;
  auto plane_balance = tcg_plane_balance->d_view;

  array_sum::GlobalSum applied_sum;
  Kokkos::parallel_reduce("turb_cgm_src", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real z = CellCenterX(k - ks, nx3, x3min, x3max);
    Real rho = w0(m, IDN, k, j, i);

    if (gravity) {
      Real g3 = TcgGravity(z, theta0, z0, h);
      Real src = bdt*rho*g3;
      u0(m, IM3, k, j, i) += src;
      u0(m, IEN, k, j, i) += src*w0(m, IVZ, k, j, i);
    }

    if (cooling || thermostat || disk_balance_rate != 0.0) {
      Real inv_rho = 1.0/u0(m, IDN, k, j, i);
      Real ekin = 0.5*(SQR(u0(m, IM1, k, j, i)) + SQR(u0(m, IM2, k, j, i)) +
                       SQR(u0(m, IM3, k, j, i)))*inv_rho;
      Real emag = 0.0;
      if (is_mhd) {
        emag = 0.5*(SQR(bcc0(m, IBX, k, j, i)) + SQR(bcc0(m, IBY, k, j, i)) +
                    SQR(bcc0(m, IBZ, k, j, i)));
      }
      Real eint = fmax(u0(m, IEN, k, j, i) - ekin - emag, 0.0);
      Real thermal_rate = 0.0;
      Real cool_rate = 0.0;
      Real heat_rate = 0.0;
      Real disk_rate = 0.0;
      Real cooling_mask = TcgThermalMask(z, h, thermal_transition, thermal_mask);
      Real disk_weight = TcgDiskBalanceWeight(z, disk_balance_width,
                                              disk_balance_transition);
      if (cooling) {
        cool_rate = cooling_mask*cooling_coef*rho*rho;
        thermal_rate -= cool_rate;
      }
      if (thermostat) {
        int kg = TcgGlobalK(z, x3min_mesh, dx3_mesh, mesh_nx3);
        heat_rate = rho*plane_balance(kg);
        thermal_rate += heat_rate;
      }
      disk_rate = disk_weight*rho*disk_balance_rate;
      thermal_rate += disk_rate;
      Real denergy = bdt*thermal_rate;
      Real clip_scale = 1.0;
      if (cooling_cfl > 0.0) {
        Real max_delta = cooling_cfl*eint;
        if (fabs(denergy) > max_delta) {
          Real clipped = (denergy >= 0.0) ? max_delta : -max_delta;
          clip_scale = (denergy != 0.0) ? clipped/denergy : 1.0;
          denergy = clipped;
        }
      }
      u0(m, IEN, k, j, i) += denergy;
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      mb_sum.the_array[0] += clip_scale*cool_rate*vol;
      mb_sum.the_array[1] += clip_scale*heat_rate*vol;
      mb_sum.the_array[2] += clip_scale*disk_rate*vol;
      mb_sum.the_array[3] += (bdt > 0.0 ? denergy/bdt : 0.0)*vol;
    }

    if (floor) {
      Real inv_rho = 1.0/u0(m, IDN, k, j, i);
      Real ekin = 0.5*(SQR(u0(m, IM1, k, j, i)) + SQR(u0(m, IM2, k, j, i)) +
                       SQR(u0(m, IM3, k, j, i)))*inv_rho;
      Real emag = 0.0;
      if (is_mhd) {
        emag = 0.5*(SQR(bcc0(m, IBX, k, j, i)) + SQR(bcc0(m, IBY, k, j, i)) +
                    SQR(bcc0(m, IBZ, k, j, i)));
      }
      Real eint_floor = u0(m, IDN, k, j, i)*theta_floor/gm1;
      u0(m, IEN, k, j, i) = fmax(u0(m, IEN, k, j, i), ekin + emag + eint_floor);
    }
  }, Kokkos::Sum<array_sum::GlobalSum>(applied_sum));

  tcg_applied_cool_src = applied_sum.the_array[0];
  tcg_applied_heat_src = applied_sum.the_array[1];
  tcg_applied_disk_src = applied_sum.the_array[2];
  tcg_applied_net_src = applied_sum.the_array[3];
#if MPI_PARALLEL_ENABLED
  Real applied_rates[4] = {tcg_applied_cool_src, tcg_applied_heat_src,
                           tcg_applied_disk_src, tcg_applied_net_src};
  MPI_Allreduce(MPI_IN_PLACE, applied_rates, 4, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  tcg_applied_cool_src = applied_rates[0];
  tcg_applied_heat_src = applied_rates[1];
  tcg_applied_disk_src = applied_rates[2];
  tcg_applied_net_src = applied_rates[3];
#endif
}

void TcgHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 47;
  pdata->label[0] = "tcg_raw";

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    for (int n = 0; n < pdata->nhist; ++n) pdata->hdata[n] = 0.0;
    return;
  }

  const bool is_mhd = (pmbp->pmhd != nullptr);
  DvceArray5D<Real> w0, bcc0;
  EOS_Data eos;
  if (pmbp->phydro != nullptr) {
    w0 = pmbp->phydro->w0;
    eos = pmbp->phydro->peos->eos_data;
  } else {
    w0 = pmbp->pmhd->w0;
    bcc0 = pmbp->pmhd->bcc0;
    eos = pmbp->pmhd->peos->eos_data;
  }
  auto force = (pmbp->pturb != nullptr) ? pmbp->pturb->force : w0;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  TcgPreparePlaneAverages(pm);

  const Real midplane_width = 10.0;
  const Real cold_temp = tcg.theta_floor*1.01;
  const Real theta0 = tcg.theta0;
  const Real z0 = tcg.z0;
  const Real h = tcg.h;
  const Real rho_z0 = tcg.rho_z0;
  const Real cooling_coef = tcg.cooling_coef;
  const Real thermal_transition = tcg.thermal_transition;
  const Real disk_balance_width = tcg.disk_balance_width;
  const Real disk_balance_transition = tcg.disk_balance_transition;
  const Real gm1 = eos.gamma - 1.0;
  const Real dt = pm->dt;
  const bool cooling = tcg.cooling;
  const bool thermostat = tcg.thermostat;
  const bool gravity = tcg.gravity;
  const bool has_turbulence = pmbp->pturb != nullptr;
  const int thermal_mask = tcg.thermal_mask;
  const Real disk_balance_rate = tcg_disk_balance_rate;
  const int mesh_nx3 = tcg.mesh_nx3;
  const Real x3min_mesh = pm->mesh_size.x3min;
  const Real dx3_mesh = pm->mesh_size.dx3;
  auto plane_balance = tcg_plane_balance->d_view;
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("turb_cgm_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real z = CellCenterX(k - ks, nx3, x3min, x3max);
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real rho = w0(m, IDN, k, j, i);
    Real pgas = eos.IdealGasPressure(w0(m, IEN, k, j, i));
    Real temp = pgas/rho;
    Real vx = w0(m, IVX, k, j, i);
    Real vy = w0(m, IVY, k, j, i);
    Real vz = w0(m, IVZ, k, j, i);
    Real cs = eos.IdealHydroSoundSpeed(rho, pgas);
    Real mach = sqrt(vx*vx + vy*vy + vz*vz)/cs;
    Real mass = rho*vol;
    bool in_midplane = TcgAbsZ(z) <= midplane_width;
    bool in_thermal_atm = TcgAbsZ(z) >= h;
    Real tcool_bg = TcgBackgroundCoolingTime(z, rho_z0, theta0, z0, h,
                                             cooling_coef, gm1);
    Real cooling_mask = TcgThermalMask(z, h, thermal_transition, thermal_mask);
    Real disk_weight = TcgDiskBalanceWeight(z, disk_balance_width,
                                            disk_balance_transition);
    Real disk_weight_dz = TcgDiskBalanceWeightDz(z, disk_balance_width,
                                                 disk_balance_transition);
    Real atm_cool_rate = cooling ? cooling_mask*cooling_coef*rho*rho : 0.0;
    Real disk_cool_rate = 0.0;
    Real cool_rate = atm_cool_rate + disk_cool_rate;
    Real heat_rate = 0.0;
    Real disk_heat_rate = 0.0;
    if (thermostat) {
      int kg = TcgGlobalK(z, x3min_mesh, dx3_mesh, mesh_nx3);
      heat_rate = rho*plane_balance(kg);
    }
    Real disk_sink_rate = disk_weight*rho*disk_balance_rate;
    Real v2 = vx*vx + vy*vy + vz*vz;
    Real eint = pgas/gm1;
    Real ekin = 0.5*rho*v2;
    Real emag = 0.0;
    if (is_mhd) {
      emag = 0.5*(SQR(bcc0(m, IBX, k, j, i)) + SQR(bcc0(m, IBY, k, j, i)) +
                  SQR(bcc0(m, IBZ, k, j, i)));
    }
    Real etot = eint + ekin + emag;
    Real disk_mass_adv = rho*vz*disk_weight_dz;
    Real disk_energy_adv = (etot + pgas)*vz*disk_weight_dz;
    Real disk_eint_adv = eint*vz*disk_weight_dz;
    Real disk_grav_work = gravity ?
        disk_weight*rho*TcgGravity(z, theta0, z0, h)*vz : 0.0;
    Real disk_turb_work = 0.0;
    if (has_turbulence) {
      Real a1 = force(m, 0, k, j, i);
      Real a2 = force(m, 1, k, j, i);
      Real a3 = force(m, 2, k, j, i);
      Real force_dot_v = a1*vx + a2*vy + a3*vz;
      Real force_sq = a1*a1 + a2*a2 + a3*a3;
      disk_turb_work = disk_weight*rho*(force_dot_v + 0.5*force_sq*dt);
    }
    Real source_frac = (eint > 0.0) ?
        fabs(dt*(heat_rate - cool_rate + disk_sink_rate))/eint : 0.0;

    array_sum::GlobalSum hvars;
    hvars.the_array[0] = vol;
    hvars.the_array[1] = mass;
    hvars.the_array[2] = in_midplane ? mass : 0.0;
    hvars.the_array[3] = (in_midplane && temp <= cold_temp) ? mass : 0.0;
    hvars.the_array[4] = temp*vol;
    hvars.the_array[5] = rho*vol;
    hvars.the_array[6] = pgas*vol;
    hvars.the_array[7] = fabs(vz)*vol;
    hvars.the_array[8] = in_midplane ? temp*vol : 0.0;
    hvars.the_array[9] = in_midplane ? vol : 0.0;
    hvars.the_array[10] = in_thermal_atm ? temp*vol : 0.0;
    hvars.the_array[11] = in_thermal_atm ? vol : 0.0;
    hvars.the_array[12] = cool_rate*vol;
    hvars.the_array[13] = heat_rate*vol;
    hvars.the_array[14] = mach*vol;
    hvars.the_array[15] = in_thermal_atm ? tcool_bg*vol : 0.0;
    hvars.the_array[16] = source_frac*vol;
    hvars.the_array[17] = disk_cool_rate*vol;
    hvars.the_array[18] = disk_heat_rate*vol;
    for (int n = 19; n < NHISTORY_VARIABLES; ++n) hvars.the_array[n] = 0.0;
    hvars.the_array[38] = disk_weight*mass;
    hvars.the_array[39] = disk_weight*eint*vol;
    hvars.the_array[40] = disk_weight*ekin*vol;
    hvars.the_array[41] = disk_weight*etot*vol;
    hvars.the_array[42] = disk_mass_adv*vol;
    hvars.the_array[43] = disk_energy_adv*vol;
    hvars.the_array[44] = disk_eint_adv*vol;
    hvars.the_array[45] = disk_grav_work*vol;
    hvars.the_array[46] = disk_turb_work*vol;
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sum.the_array[n];
  }

  TcgDiagData diag;
  Kokkos::parallel_reduce("turb_cgm_diag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, TcgDiagData &mb_diag) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x = CellCenterX(i - is, nx1, x1min, x1max);
    Real y = CellCenterX(j - js, nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, nx3, x3min, x3max);
    Real rho = w0(m, IDN, k, j, i);
    Real pgas = eos.IdealGasPressure(w0(m, IEN, k, j, i));
    Real temp = pgas/rho;
    Real vx = w0(m, IVX, k, j, i);
    Real vy = w0(m, IVY, k, j, i);
    Real vz = w0(m, IVZ, k, j, i);
    Real abs_v = sqrt(vx*vx + vy*vy + vz*vz);
    Real cs = eos.IdealHydroSoundSpeed(rho, pgas);
    Real mach = abs_v/cs;
    Real va2 = 0.0;
    if (is_mhd) {
      va2 = (SQR(bcc0(m, IBX, k, j, i)) + SQR(bcc0(m, IBY, k, j, i)) +
             SQR(bcc0(m, IBZ, k, j, i)))/rho;
    }
    Real signal_speed = abs_v + sqrt(cs*cs + va2);

    Real cooling_mask = TcgThermalMask(z, h, thermal_transition, thermal_mask);
    Real disk_weight = TcgDiskBalanceWeight(z, disk_balance_width,
                                            disk_balance_transition);
    Real cool_rate = cooling ? cooling_mask*cooling_coef*rho*rho : 0.0;
    Real heat_rate = 0.0;
    if (thermostat) {
      int kg = TcgGlobalK(z, x3min_mesh, dx3_mesh, mesh_nx3);
      heat_rate = rho*plane_balance(kg);
    }
    Real disk_sink_rate = disk_weight*rho*disk_balance_rate;
    Real eint = pgas/gm1;
    Real source_frac = (eint > 0.0) ?
        fabs(dt*(heat_rate - cool_rate + disk_sink_rate))/eint : 0.0;

    mb_diag.min_temp = fmin(mb_diag.min_temp, temp);
    mb_diag.max_temp = fmax(mb_diag.max_temp, temp);
    mb_diag.min_rho = fmin(mb_diag.min_rho, rho);
    mb_diag.max_rho = fmax(mb_diag.max_rho, rho);
    mb_diag.max_mach = fmax(mb_diag.max_mach, mach);
    mb_diag.max_abs_v = fmax(mb_diag.max_abs_v, abs_v);
    mb_diag.max_source_frac = fmax(mb_diag.max_source_frac, source_frac);
    if (signal_speed > mb_diag.max_signal_speed) {
      mb_diag.max_signal_speed = signal_speed;
      mb_diag.max_signal_x = x;
      mb_diag.max_signal_y = y;
      mb_diag.max_signal_z = z;
    }
  }, diag);

  Real min_temp = diag.min_temp;
  Real max_temp = diag.max_temp;
  Real min_rho = diag.min_rho;
  Real max_rho = diag.max_rho;
  Real max_mach = diag.max_mach;
  Real max_abs_v = diag.max_abs_v;
  Real max_signal_speed = diag.max_signal_speed;
  Real max_source_frac = diag.max_source_frac;
  Real max_signal_x = diag.max_signal_x;
  Real max_signal_y = diag.max_signal_y;
  Real max_signal_z = diag.max_signal_z;
  Real disk_budget[9] = {sum.the_array[38], sum.the_array[39], sum.the_array[40],
                         sum.the_array[41], sum.the_array[42], sum.the_array[43],
                         sum.the_array[44], sum.the_array[45], sum.the_array[46]};

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &min_temp, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_temp, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &min_rho, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_rho, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_mach, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_abs_v, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_signal_speed, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_source_frac, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, disk_budget, 9, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);

  Real signal_loc[4] = {0.0, 0.0, 0.0, 0.0};
  if (diag.max_signal_speed >= max_signal_speed) {
    signal_loc[0] = diag.max_signal_x;
    signal_loc[1] = diag.max_signal_y;
    signal_loc[2] = diag.max_signal_z;
    signal_loc[3] = 1.0;
  }
  MPI_Allreduce(MPI_IN_PLACE, signal_loc, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (signal_loc[3] > 0.0) {
    max_signal_x = signal_loc[0]/signal_loc[3];
    max_signal_y = signal_loc[1]/signal_loc[3];
    max_signal_z = signal_loc[2]/signal_loc[3];
  }
#endif

  pdata->hdata[19] = min_temp;
  pdata->hdata[20] = max_temp;
  pdata->hdata[21] = min_rho;
  pdata->hdata[22] = max_rho;
  pdata->hdata[23] = max_mach;
  pdata->hdata[24] = max_abs_v;
  pdata->hdata[25] = max_signal_speed;
  pdata->hdata[26] = max_source_frac;
  pdata->hdata[27] = max_signal_x;
  pdata->hdata[28] = max_signal_y;
  pdata->hdata[29] = max_signal_z;
  pdata->hdata[30] = tcg_last_turb_power;
  pdata->hdata[31] = tcg_global_balance_target;
  pdata->hdata[32] = tcg_disk_turb_power;
  pdata->hdata[33] = tcg_disk_balance_target;
  pdata->hdata[34] = tcg_applied_cool_src;
  pdata->hdata[35] = tcg_applied_heat_src;
  pdata->hdata[36] = tcg_applied_disk_src;
  pdata->hdata[37] = tcg_applied_net_src;
  pdata->hdata[38] = disk_budget[0];
  pdata->hdata[39] = disk_budget[1];
  pdata->hdata[40] = disk_budget[2];
  pdata->hdata[41] = disk_budget[3];
  pdata->hdata[42] = disk_budget[4];
  pdata->hdata[43] = disk_budget[5];
  pdata->hdata[44] = disk_budget[6];
  pdata->hdata[45] = disk_budget[7];
  pdata->hdata[46] = disk_budget[8];

#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank != 0) {
    for (int n = 19; n < pdata->nhist; ++n) {
      pdata->hdata[n] = 0.0;
    }
  }
#endif
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Initialize a stratified, isothermal, hydrostatic CGM atmosphere.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool has_hydro = (pmbp->phydro != nullptr);
  const bool has_mhd = (pmbp->pmhd != nullptr);
  if (!has_hydro && !has_mhd) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm requires either a <hydro> or <mhd> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (has_hydro && has_mhd) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm expects pure hydro or pure MHD, not both." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  EquationOfState *peos = has_hydro ? pmbp->phydro->peos : pmbp->pmhd->peos;
  EOS_Data &eos = peos->eos_data;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm requires an ideal-gas EOS." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  tcg.temp0 = pin->GetOrAddReal("problem", "temp0_cgs", 1.0e7);
  tcg.temp_floor = pin->GetOrAddReal("problem", "temp_floor_cgs", 1.0e6);
  tcg.z0 = pin->GetOrAddReal("problem", "z0", 50.0);
  tcg.h = pin->GetOrAddReal("problem", "h", 5.0);
  tcg.xmu = pin->GetOrAddReal("problem", "mu", pmbp->punit->mu());
  tcg.hydrogen_mass_fraction =
      pin->GetOrAddReal("problem", "hydrogen_mass_fraction", 0.75);
  tcg.lambda0_cgs = pin->GetOrAddReal("problem", "lambda0_cgs", 1.0e-22);
  tcg.tcool_tff_z0 = pin->GetOrAddReal("problem", "tcool_tff_z0", 4.0);
  std::string thermal_mask =
      pin->GetOrAddString("problem", "thermal_mask", "outer_tanh");
  tcg.thermal_mask = TcgThermalMaskFromString(thermal_mask);
  tcg.cooling_cfl = pin->GetOrAddReal("problem", "cooling_cfl", 0.1);
  tcg.thermal_transition = pin->GetOrAddReal("problem", "thermal_transition", 2.5);
  tcg.disk_balance_width =
      pin->GetOrAddReal("problem", "disk_balance_width", tcg.h);
  tcg.disk_balance_transition =
      pin->GetOrAddReal("problem", "disk_balance_transition", tcg.thermal_transition);
  tcg.perturb_amp = pin->GetOrAddReal("problem", "perturb_amp", 0.0);
  std::string magnetic_field = pin->GetOrAddString("problem", "magnetic_field", "none");
  tcg.magnetic_field = TcgMagneticFieldFromString(magnetic_field);
  tcg.beta_z0 = pin->GetOrAddReal("problem", "beta_z0", 100.0);
  tcg.gravity = pin->GetOrAddBoolean("problem", "gravity", true);
  tcg.cooling = pin->GetOrAddBoolean("problem", "cooling", false);
  tcg.thermostat = pin->GetOrAddBoolean("problem", "thermostat", false);
  tcg.enforce_temp_floor = pin->GetOrAddBoolean("problem", "enforce_temp_floor", true);
  tcg.theta0 = tcg.temp0*pmbp->punit->kelvin();
  tcg.theta_floor = tcg.temp_floor*pmbp->punit->kelvin();
  tcg.cooling_coef = SQR(tcg.hydrogen_mass_fraction)*
      tcg.lambda0_cgs*SQR(pmbp->punit->density_cgs()/units::Units::atomic_mass_unit_cgs)/
      (pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs());
  tcg.rho_z0 = TcgRhoZ0FromTcoolTff(pin, pmbp, tcg.temp0, tcg.z0, tcg.h, tcg.xmu,
                                    tcg.hydrogen_mass_fraction, tcg.lambda0_cgs,
                                    tcg.tcool_tff_z0);
  tcg.mesh_nx3 = pmy_mesh_->mesh_indcs.nx3;
  if (tcg_plane_sums == nullptr) {
    tcg_plane_sums = new DualArray1D<Real>("turb_cgm_plane_sums", 3*tcg.mesh_nx3);
  }
  if (tcg_plane_balance == nullptr) {
    tcg_plane_balance = new DualArray1D<Real>("turb_cgm_plane_balance", tcg.mesh_nx3);
  }
  for (int k = 0; k < tcg.mesh_nx3; ++k) {
    tcg_plane_balance->h_view(k) = 0.0;
  }
  tcg_plane_balance->modify_host();
  tcg_plane_balance->sync_device();
  tcg.initialized = true;

  user_srcs = true;
  user_srcs_func = TcgSourceTerms;
  user_hist = true;
  user_hist_func = TcgHistory;
  if (restart) return;

  if (tcg.z0 <= 0.0 || tcg.h <= 0.0 || tcg.rho_z0 <= 0.0 || tcg.theta0 <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm requires z0 > 0, h > 0, rho_z0 > 0, and temp0_cgs > 0."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (tcg.cooling_cfl <= 0.0 || tcg.cooling_coef <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm requires cooling_cfl > 0 and lambda0_cgs > 0."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (tcg.thermal_transition < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm requires thermal_transition >= 0. Use 0 for a hard "
              << "Wibking thermal cutoff." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (tcg.disk_balance_width < 0.0 || tcg.disk_balance_transition < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm requires disk_balance_width >= 0 and "
              << "disk_balance_transition >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  tcg.disk_balance_transition = std::min(tcg.disk_balance_transition,
                                         tcg.disk_balance_width);
  if (has_hydro && tcg.magnetic_field != TCG_MAGNETIC_NONE) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm magnetic_field requires an <mhd> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (has_mhd && tcg.magnetic_field == TCG_MAGNETIC_SPECTRAL && tcg.beta_z0 <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm spectral magnetic field requires beta_z0 > 0."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (tcg.thermostat && pmy_mesh_->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_cgm thermostat currently supports uniform, non-AMR meshes only."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;

  const Real rho_z0 = tcg.rho_z0;
  const Real theta0 = tcg.theta0;
  const Real z0 = tcg.z0;
  const Real h = tcg.h;
  const Real gm1 = eos.gamma - 1.0;
  const Real perturb_amp = tcg.perturb_amp;
  const Real lx = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  const Real ly = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  const Real lz = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;
  const Real x1min_mesh = pmy_mesh_->mesh_size.x1min;
  const Real x2min_mesh = pmy_mesh_->mesh_size.x2min;
  const Real x3min_mesh = pmy_mesh_->mesh_size.x3min;
  const Real two_pi = 2.0*std::acos(-1.0);

  if (has_hydro) {
    auto &u0 = pmbp->phydro->u0;
    par_for("pgen_turb_cgm_hydro", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x = CellCenterX(i - is, nx1, x1min, x1max);
      Real y = CellCenterX(j - js, nx2, x2min, x2max);
      Real z = CellCenterX(k - ks, nx3, x3min, x3max);
      Real rho = TcgDensityProfile(z, rho_z0, z0, h);
      if (perturb_amp != 0.0) {
        Real sx = sin(two_pi*(x - x1min_mesh)/lx);
        Real sy = sin(two_pi*(y - x2min_mesh)/ly);
        Real sz = sin(two_pi*(z - x3min_mesh)/lz);
        rho *= fmax(1.0 + perturb_amp*sx*sy*sz, 0.01);
      }
      Real pgas = rho*theta0;
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = pgas/gm1;
    });
  } else {
    auto &w0 = pmbp->pmhd->w0;
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    int nscalars = pmbp->pmhd->nscalars;
    Kokkos::deep_copy(b0.x1f, 0.0);
    Kokkos::deep_copy(b0.x2f, 0.0);
    Kokkos::deep_copy(b0.x3f, 0.0);

    par_for("pgen_turb_cgm_mhd_prim", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x = CellCenterX(i - is, nx1, x1min, x1max);
      Real y = CellCenterX(j - js, nx2, x2min, x2max);
      Real z = CellCenterX(k - ks, nx3, x3min, x3max);
      Real rho = TcgDensityProfile(z, rho_z0, z0, h);
      if (perturb_amp != 0.0) {
        Real sx = sin(two_pi*(x - x1min_mesh)/lx);
        Real sy = sin(two_pi*(y - x2min_mesh)/ly);
        Real sz = sin(two_pi*(z - x3min_mesh)/lz);
        rho *= fmax(1.0 + perturb_amp*sx*sy*sz, 0.01);
      }
      Real pgas = rho*theta0;
      w0(m, IDN, k, j, i) = rho;
      w0(m, IVX, k, j, i) = 0.0;
      w0(m, IVY, k, j, i) = 0.0;
      w0(m, IVZ, k, j, i) = 0.0;
      w0(m, IEN, k, j, i) = pgas/gm1;
      for (int n = 0; n < nscalars; ++n) {
        w0(m, IYF + n, k, j, i) = 0.0;
      }
      bcc0(m, IBX, k, j, i) = 0.0;
      bcc0(m, IBY, k, j, i) = 0.0;
      bcc0(m, IBZ, k, j, i) = 0.0;
    });

    std::string spectral_backend = "none";
    if (tcg.magnetic_field == TCG_MAGNETIC_SPECTRAL) {
      int nmb = pmbp->nmb_thispack;
      int ncells1 = indcs.nx1 + 2*indcs.ng;
      int ncells2 = indcs.nx2 + 2*indcs.ng;
      int ncells3 = indcs.nx3 + 2*indcs.ng;
      DvceArray4D<Real> ax("turb_cgm_ax_spec", nmb, ncells3, ncells2, ncells1);
      DvceArray4D<Real> ay("turb_cgm_ay_spec", nmb, ncells3, ncells2, ncells1);
      DvceArray4D<Real> az("turb_cgm_az_spec", nmb, ncells3, ncells2, ncells1);
      par_for("pgen_turb_cgm_zero_a", DevExeSpace(), 0, nmb - 1,
              0, ncells3 - 1, 0, ncells2 - 1, 0, ncells1 - 1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        ax(m, k, j, i) = 0.0;
        ay(m, k, j, i) = 0.0;
        az(m, k, j, i) = 0.0;
      });

      SpectralICGenerator gen(pmbp, pin);
      spectral_backend = gen.GenerateVectorPotentialFFT(ax, ay, az);

      par_for("pgen_turb_cgm_curl_a", DevExeSpace(), 0, nmb - 1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        b0.x1f(m, k, j, i) =
            (az(m, k, j + 1, i) - az(m, k, j, i))/dx2
          - (ay(m, k + 1, j, i) - ay(m, k, j, i))/dx3;
        b0.x2f(m, k, j, i) =
            (ax(m, k + 1, j, i) - ax(m, k, j, i))/dx3
          - (az(m, k, j, i + 1) - az(m, k, j, i))/dx1;
        b0.x3f(m, k, j, i) =
            (ay(m, k, j, i + 1) - ay(m, k, j, i))/dx1
          - (ax(m, k, j + 1, i) - ax(m, k, j, i))/dx2;
        if (i == ie) {
          b0.x1f(m, k, j, i + 1) =
              (az(m, k, j + 1, i + 1) - az(m, k, j, i + 1))/dx2
            - (ay(m, k + 1, j, i + 1) - ay(m, k, j, i + 1))/dx3;
        }
        if (j == je) {
          b0.x2f(m, k, j + 1, i) =
              (ax(m, k + 1, j + 1, i) - ax(m, k, j + 1, i))/dx3
            - (az(m, k, j + 1, i + 1) - az(m, k, j + 1, i))/dx1;
        }
        if (k == ke) {
          b0.x3f(m, k + 1, j, i) =
              (ay(m, k + 1, j, i + 1) - ay(m, k + 1, j, i))/dx1
            - (ax(m, k + 1, j + 1, i) - ax(m, k + 1, j, i))/dx2;
        }
      });

      SubtractGlobalMeanB(pmbp, b0);
      const Real rms_b = std::sqrt(2.0*rho_z0*theta0/tcg.beta_z0);
      NormalizeRmsB(pmbp, b0, rms_b);

      const Real bmean_x = pin->GetOrAddReal("spectral_ic", "b_mean_x", 0.0);
      const Real bmean_y = pin->GetOrAddReal("spectral_ic", "b_mean_y", 0.0);
      const Real bmean_z = pin->GetOrAddReal("spectral_ic", "b_mean_z", 0.0);
      if (bmean_x != 0.0 || bmean_y != 0.0 || bmean_z != 0.0) {
        par_for("pgen_turb_cgm_bg_b", DevExeSpace(), 0, nmb - 1,
                ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          b0.x1f(m, k, j, i) += bmean_x;
          b0.x2f(m, k, j, i) += bmean_y;
          b0.x3f(m, k, j, i) += bmean_z;
          if (i == ie) b0.x1f(m, k, j, i + 1) += bmean_x;
          if (j == je) b0.x2f(m, k, j + 1, i) += bmean_y;
          if (k == ke) b0.x3f(m, k + 1, j, i) += bmean_z;
        });
      }
    }

    const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji = nx2*nx1;
    Kokkos::parallel_for("pgen_turb_cgm_bcc",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real bx = 0.5*(b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i + 1));
      Real by = 0.5*(b0.x2f(m, k, j, i) + b0.x2f(m, k, j + 1, i));
      Real bz = 0.5*(b0.x3f(m, k, j, i) + b0.x3f(m, k + 1, j, i));
      bcc0(m, IBX, k, j, i) = bx;
      bcc0(m, IBY, k, j, i) = by;
      bcc0(m, IBZ, k, j, i) = bz;
    });
    if (global_variable::my_rank == 0 && tcg.magnetic_field == TCG_MAGNETIC_SPECTRAL) {
      std::cout << "turb_cgm: spectral magnetic field generated using backend='"
                << spectral_backend << "'" << std::endl;
    }
    pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
  }

  if (global_variable::my_rank == 0) {
    std::cout << "turb_cgm: temp0=" << tcg.temp0 << " K, theta0=" << tcg.theta0
              << ", z0=" << tcg.z0 << ", h=" << tcg.h
              << ", rho_z0=" << tcg.rho_z0
              << ", tcool_tff_z0=" << tcg.tcool_tff_z0
              << ", gravity=" << tcg.gravity
              << ", cooling=" << tcg.cooling
              << ", thermostat=" << tcg.thermostat
              << ", thermal_mask=" << thermal_mask
              << ", disk_balance_width=" << tcg.disk_balance_width
              << ", disk_balance_transition=" << tcg.disk_balance_transition
              << ", cooling_cfl=" << tcg.cooling_cfl
              << ", thermal_transition=" << tcg.thermal_transition;
    if (has_mhd) {
      std::cout << ", magnetic_field=" << magnetic_field
                << ", beta_z0=" << tcg.beta_z0;
    }
    std::cout << std::endl;
  }
}
