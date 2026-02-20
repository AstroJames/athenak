//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  Implements various (physics) source terms to be added to the Hydro or MHD eqns.
//  Source terms objects are stored in the respective fluid class, so that
//  Hydro/MHD can have different source terms

#include "srcterms.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string> // string
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "ismcooling.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "turb_driver.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  shearing_box_r_phi(false),
  sn_rng_(0),
  sn_event_count_(0),
  sn_event_accum_(0.0),
  sn_rinj_warned_(false) {
  // (1) (constant) gravitational acceleration
  const_accel = pin->GetOrAddBoolean(block, "const_accel", false);
  if (const_accel) {
    const_accel_val = pin->GetReal(block, "const_accel_val");
    const_accel_dir = pin->GetInteger(block, "const_accel_dir");
    if (const_accel_dir < 1 || const_accel_dir > 3) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "const_accle_dir must be 1,2, or 3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (2) Optically thin (ISM) cooling
  ism_cooling = pin->GetOrAddBoolean(block, "ism_cooling", false);
  if (ism_cooling) {
    hrate = pin->GetReal(block, "hrate");
  }

  // (3) beam source (radiation)
  beam = pin->GetOrAddBoolean(block, "beam_source", false);
  if (beam) {
    dii_dt = pin->GetReal(block, "dii_dt");
  }

  // (4) cooling (relativistic)
  rel_cooling = pin->GetOrAddBoolean(block, "rel_cooling", false);
  if (rel_cooling) {
    crate_rel = pin->GetReal(block, "crate_rel");
    cpower_rel = pin->GetOrAddReal(block, "cpower_rel", 1.);
  }

  // (5) stochastic supernova driving
  sn_driving = pin->GetOrAddBoolean(block, "sn_driving", false);
  if (sn_driving) {
    sn_rate = pin->GetReal(block, "sn_rate");
    sn_seed = pin->GetOrAddInteger(block, "sn_seed", 12345);
    sn_rng_.seed(static_cast<unsigned long long>(sn_seed));

    if (pin->DoesParameterExist(block, "sn_energy")) {
      sn_einj = pin->GetReal(block, "sn_energy");
    } else {
      Real sn_energy_cgs = pin->GetOrAddReal(block, "sn_energy_cgs", 1.0e51);
      sn_einj = sn_energy_cgs * pmy_pack->punit->erg();
    }

    if (pin->DoesParameterExist(block, "sn_rinj")) {
      sn_rinj = pin->GetReal(block, "sn_rinj");
    } else {
      Real sn_radius_pc = pin->GetOrAddReal(block, "sn_radius_pc", 8.0);
      sn_rinj = sn_radius_pc * pmy_pack->punit->pc();
    }

    if (pin->DoesParameterExist(block, "sn_zmin")) {
      sn_zmin = pin->GetReal(block, "sn_zmin");
    } else if (pin->DoesParameterExist(block, "sn_zmin_pc")) {
      sn_zmin = pin->GetReal(block, "sn_zmin_pc")*pmy_pack->punit->pc();
    } else {
      sn_zmin = pmy_pack->pmesh->mesh_size.x3min;
    }

    if (pin->DoesParameterExist(block, "sn_zmax")) {
      sn_zmax = pin->GetReal(block, "sn_zmax");
    } else if (pin->DoesParameterExist(block, "sn_zmax_pc")) {
      sn_zmax = pin->GetReal(block, "sn_zmax_pc")*pmy_pack->punit->pc();
    } else {
      sn_zmax = pmy_pack->pmesh->mesh_size.x3max;
    }

    sn_zmin = std::max(sn_zmin, pmy_pack->pmesh->mesh_size.x3min);
    sn_zmax = std::min(sn_zmax, pmy_pack->pmesh->mesh_size.x3max);
    if (sn_zmax <= sn_zmin) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "SN z-band is invalid: require sn_zmax > sn_zmin after clipping."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    sn_log_events = pin->GetOrAddBoolean(block, "sn_log_events", true);
    std::string basename = pin->GetOrAddString("job", "basename", "AthenaK");
    sn_log_file = pin->GetOrAddString(block, "sn_log_file",
                                      basename + ".sn_events.dat");
  }

  // (6) shearing box
  if (pin->DoesBlockExist("shearing_box")) {
    shearing_box = true;
    qshear = pin->GetReal("shearing_box","qshear");
    omega0 = pin->GetReal("shearing_box","omega0");
  } else {
    shearing_box = false;
  }
}

//----------------------------------------------------------------------------------------
// destructor

SourceTerms::~SourceTerms() {
}

//----------------------------------------------------------------------------------------
//! \fn
// Add constant acceleration
// NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  Real &g = const_accel_val;
  int &dir = const_accel_dir;

  par_for("const_acc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real src = bdt*g*w0(m,IDN,k,j,i);
    u0(m,dir,k,j,i) += src;
    if (eos_data.is_ideal) { u0(m,IEN,k,j,i) += src*w0(m,dir,k,j,i); }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::ISMCooling()
//! \brief Add explict ISM cooling and heating source terms in the energy equations.
// NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real use_e = eos_data.use_e;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = hrate;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()/n_unit;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = 1.0;
    if (use_e) {
      temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    } else {
      temp = temp_unit*w0(m,ITM,k,j,i);
    }

    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real gamma_heating = heating_rate/heating_unit;

    u0(m,IEN,k,j,i) -= bdt * w0(m,IDN,k,j,i) *
                        (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::RelCooling()
//! \brief Add explict relativistic cooling in the energy and momentum equations.
// NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real use_e = eos_data.use_e;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real cooling_rate = crate_rel;
  Real cooling_power = cpower_rel;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = 1.0;
    if (use_e) {
      temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    } else {
      temp = w0(m,ITM,k,j,i);
    }

    auto &ux = w0(m,IVX,k,j,i);
    auto &uy = w0(m,IVY,k,j,i);
    auto &uz = w0(m,IVZ,k,j,i);

    auto ut = 1.0 + ux*ux + uy*uy + uz*uz;
    ut = sqrt(ut);

    u0(m,IEN,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ut*pow((temp*cooling_rate), cooling_power);
    u0(m,IM1,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ux*pow((temp*cooling_rate), cooling_power);
    u0(m,IM2,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uy*pow((temp*cooling_rate), cooling_power);
    u0(m,IM3,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uz*pow((temp*cooling_rate), cooling_power);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::SupernovaDriving()
//! \brief Add stochastic supernova thermal energy source terms in the energy equation.
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars.

void SourceTerms::SupernovaDriving(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                   const Real bdt, DvceArray5D<Real> &u0) {
  if (!eos_data.is_ideal) return;
  if (bdt <= 0.0 || sn_rate <= 0.0 || sn_rinj <= 0.0 || sn_einj <= 0.0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  int nmb1 = nmb - 1;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto &size = pmy_pack->pmb->mb_size;
  auto &msz = pmy_pack->pmesh->mesh_size;

  Real x1min = msz.x1min, x1max = msz.x1max;
  Real x2min = msz.x2min, x2max = msz.x2max;
  Real x3min = msz.x3min, x3max = msz.x3max;

  Real lx1 = x1max - x1min;
  Real lx2 = x2max - x2min;
  Real lx3 = x3max - x3min;

  bool x1_periodic = (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] ==
                      BoundaryFlag::periodic ||
                      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] ==
                      BoundaryFlag::shear_periodic);
  bool x2_periodic = (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x2] ==
                      BoundaryFlag::periodic);
  bool x3_periodic = (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] ==
                      BoundaryFlag::periodic);

  // Enforce a minimum injection radius of two cell widths on the active mesh.
  Real min_dx_local = std::numeric_limits<Real>::max();
  for (int m = 0; m < nmb; ++m) {
    min_dx_local = std::min(min_dx_local, size.h_view(m).dx1);
    if (nx2 > 1) min_dx_local = std::min(min_dx_local, size.h_view(m).dx2);
    if (nx3 > 1) min_dx_local = std::min(min_dx_local, size.h_view(m).dx3);
  }

  Real min_dx = min_dx_local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &min_dx, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  Real sn_rinj_eff = std::max(sn_rinj, 2.0*min_dx);
  if (!sn_rinj_warned_ && sn_rinj_eff > sn_rinj && global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__
              << ": sn_rinj=" << sn_rinj << " is under-resolved on this mesh; using "
              << "minimum resolved radius sn_rinj=2*dx_min=" << sn_rinj_eff << "\n";
    sn_rinj_warned_ = true;
  }

  int nevents = 0;
  if (global_variable::my_rank == 0) {
    sn_event_accum_ += sn_rate*bdt;
    nevents = static_cast<int>(sn_event_accum_);
    sn_event_accum_ -= static_cast<Real>(nevents);
  }

#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&nevents, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  if (nevents <= 0) return;

  std::vector<Real> event_xyz(3*nevents, 0.0);

  if (global_variable::my_rank == 0) {
    std::uniform_real_distribution<Real> xdist(x1min, x1max);
    std::uniform_real_distribution<Real> ydist(x2min, x2max);
    Real zband_min = std::max(sn_zmin, x3min);
    Real zband_max = std::min(sn_zmax, x3max);
    std::uniform_real_distribution<Real> zdist(zband_min, zband_max);

    for (int n = 0; n < nevents; ++n) {
      Real xsn = xdist(sn_rng_);
      Real ysn = (nx2 > 1) ? ydist(sn_rng_) : 0.5*(x2min + x2max);
      Real zsn = 0.5*(zband_min + zband_max);

      if (nx3 > 1) {
        zsn = zdist(sn_rng_);
      }

      event_xyz[3*n + 0] = xsn;
      event_xyz[3*n + 1] = ysn;
      event_xyz[3*n + 2] = zsn;
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Bcast(event_xyz.data(), 3*nevents, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  Real r2inj = sn_rinj_eff*sn_rinj_eff;

  for (int n = 0; n < nevents; ++n) {
    Real xsn = event_xyz[3*n + 0];
    Real ysn = event_xyz[3*n + 1];
    Real zsn = event_xyz[3*n + 2];

    Real vol_local = 0.0;
    Kokkos::parallel_reduce("sn_volume", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_vol) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real &x1mbmin = size.d_view(m).x1min;
      Real &x1mbmax = size.d_view(m).x1max;
      Real x1v = CellCenterX(i - is, nx1, x1mbmin, x1mbmax);

      Real &x2mbmin = size.d_view(m).x2min;
      Real &x2mbmax = size.d_view(m).x2max;
      Real x2v = CellCenterX(j - js, nx2, x2mbmin, x2mbmax);

      Real &x3mbmin = size.d_view(m).x3min;
      Real &x3mbmax = size.d_view(m).x3max;
      Real x3v = CellCenterX(k - ks, nx3, x3mbmin, x3mbmax);

      Real dxx = x1v - xsn;
      if (x1_periodic) {
        if (dxx > 0.5*lx1) dxx -= lx1;
        if (dxx < -0.5*lx1) dxx += lx1;
      }

      Real dyy = x2v - ysn;
      if (x2_periodic) {
        if (dyy > 0.5*lx2) dyy -= lx2;
        if (dyy < -0.5*lx2) dyy += lx2;
      }

      Real dzz = x3v - zsn;
      if (x3_periodic) {
        if (dzz > 0.5*lx3) dzz -= lx3;
        if (dzz < -0.5*lx3) dzz += lx3;
      }

      Real r2 = dxx*dxx + dyy*dyy + dzz*dzz;
      if (r2 < r2inj) {
        sum_vol += size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      }
    }, Kokkos::Sum<Real>(vol_local));

    Real vol_global = vol_local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &vol_global, 1, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif

    if (vol_global > 0.0) {
      Real de = sn_einj/vol_global;
      par_for("sn_inject", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real &x1mbmin = size.d_view(m).x1min;
        Real &x1mbmax = size.d_view(m).x1max;
        Real x1v = CellCenterX(i - is, nx1, x1mbmin, x1mbmax);

        Real &x2mbmin = size.d_view(m).x2min;
        Real &x2mbmax = size.d_view(m).x2max;
        Real x2v = CellCenterX(j - js, nx2, x2mbmin, x2mbmax);

        Real &x3mbmin = size.d_view(m).x3min;
        Real &x3mbmax = size.d_view(m).x3max;
        Real x3v = CellCenterX(k - ks, nx3, x3mbmin, x3mbmax);

        Real dxx = x1v - xsn;
        if (x1_periodic) {
          if (dxx > 0.5*lx1) dxx -= lx1;
          if (dxx < -0.5*lx1) dxx += lx1;
        }

        Real dyy = x2v - ysn;
        if (x2_periodic) {
          if (dyy > 0.5*lx2) dyy -= lx2;
          if (dyy < -0.5*lx2) dyy += lx2;
        }

        Real dzz = x3v - zsn;
        if (x3_periodic) {
          if (dzz > 0.5*lx3) dzz -= lx3;
          if (dzz < -0.5*lx3) dzz += lx3;
        }

        Real r2 = dxx*dxx + dyy*dyy + dzz*dzz;
        if (r2 < r2inj) {
          u0(m, IEN, k, j, i) += de;
        }
      });
    }
  }

  if (sn_log_events && global_variable::my_rank == 0) {
    bool write_header = false;
    if (sn_event_count_ == 0) {
      std::ifstream ifs(sn_log_file);
      write_header = !ifs.good();
    }

    std::ofstream ofs(sn_log_file, std::ios::out | std::ios::app);
    if (ofs.is_open()) {
      if (write_header) {
        ofs << "# event_id time x y z\n";
      }
      for (int n = 0; n < nevents; ++n) {
        ofs << (sn_event_count_ + n + 1) << " " << pmy_pack->pmesh->time << " "
            << event_xyz[3*n + 0] << " " << event_xyz[3*n + 1] << " "
            << event_xyz[3*n + 2] << "\n";
      }
    } else {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__
                << ": could not open SN log file '" << sn_log_file << "'\n";
    }
  }

  sn_event_count_ += nevents;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::BeamSource()
// \brief Add beam of radiation

void SourceTerms::BeamSource(DvceArray5D<Real> &i0, const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = (pmy_pack->nmb_thispack-1);
  int nang1 = (pmy_pack->prad->prgeo->nangles-1);

  auto &nh_c_ = pmy_pack->prad->nh_c;
  auto &tt = pmy_pack->prad->tet_c;
  auto &tc = pmy_pack->prad->tetcov_c;

  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = pmy_pack->prad->n_0_floor;

  auto &beam_mask_ = pmy_pack->prad->beam_mask;
  Real &dii_dt_ = dii_dt;
  par_for("beam_source",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    if (beam_mask_(m,n,k,j,i)) {
      Real n0 = tt(m,0,0,k,j,i);
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
               + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
      i0(m,n,k,j,i) += n0*n_0*dii_dt_*bdt;
      // handle excision
      // NOTE(@pdmullen): exicision criterion are not finalized.  The below zeroes all
      // intensities within rks <= 1.0 and zeroes intensities within angles where n_0
      // is about zero.  This needs future attention.
      if (excise) {
        if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0(m,n,k,j,i) = 0.0; }
      }
    }
  });

  return;
}
