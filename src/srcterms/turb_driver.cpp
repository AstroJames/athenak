//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.cpp
//  \brief implementation of functions in TurbulenceDriver

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "driver/driver.hpp"
#include "utils/random.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "turb_driver.hpp"

namespace {
constexpr int kGeometryIsotropic = 0;
constexpr int kGeometryAnisotropic = 1;
constexpr int kProfilePowerLaw = 0;
constexpr int kProfileParabola = 1;
constexpr int kProfileBand = 2;

bool BasisActive(int n, int nkx, int nky, int nkz) {
  bool sx = (n & 4) != 0;
  bool sy = (n & 2) != 0;
  bool sz = (n & 1) != 0;
  return !((nkx == 0 && sx) || (nky == 0 && sy) || (nkz == 0 && sz));
}

void ApplySolenoidalWeight(Real kx, Real ky, Real kz, Real sol_weight,
                           Real xcoef[8], Real ycoef[8], Real zcoef[8]) {
  Real k2 = kx*kx + ky*ky + kz*kz;
  if (k2 <= 0.0) return;

  Real phi[8];
  for (int q=0; q<8; ++q) {
    Real sx = ((q & 4) == 0) ? 1.0 : -1.0;
    Real sy = ((q & 2) == 0) ? 1.0 : -1.0;
    Real sz = ((q & 1) == 0) ? 1.0 : -1.0;
    Real div = sx*kx*xcoef[q ^ 4] + sy*ky*ycoef[q ^ 2] + sz*kz*zcoef[q ^ 1];
    phi[q] = -div/k2;
  }

  for (int p=0; p<8; ++p) {
    Real sx = ((p & 4) == 0) ? 1.0 : -1.0;
    Real sy = ((p & 2) == 0) ? 1.0 : -1.0;
    Real sz = ((p & 1) == 0) ? 1.0 : -1.0;
    Real xcmp = sx*kx*phi[p ^ 4];
    Real ycmp = sy*ky*phi[p ^ 2];
    Real zcmp = sz*kz*phi[p ^ 1];
    xcoef[p] = sol_weight*(xcoef[p] - xcmp) + (1.0 - sol_weight)*xcmp;
    ycoef[p] = sol_weight*(ycoef[p] - ycmp) + (1.0 - sol_weight)*ycmp;
    zcoef[p] = sol_weight*(zcoef[p] - zcmp) + (1.0 - sol_weight)*zcmp;
  }
}

void StoreModeCoefficients(int nmode,
                           DualArray1D<Real> &xccc, DualArray1D<Real> &xccs,
                           DualArray1D<Real> &xcsc, DualArray1D<Real> &xcss,
                           DualArray1D<Real> &xscc, DualArray1D<Real> &xscs,
                           DualArray1D<Real> &xssc, DualArray1D<Real> &xsss,
                           DualArray1D<Real> &yccc, DualArray1D<Real> &yccs,
                           DualArray1D<Real> &ycsc, DualArray1D<Real> &ycss,
                           DualArray1D<Real> &yscc, DualArray1D<Real> &yscs,
                           DualArray1D<Real> &yssc, DualArray1D<Real> &ysss,
                           DualArray1D<Real> &zccc, DualArray1D<Real> &zccs,
                           DualArray1D<Real> &zcsc, DualArray1D<Real> &zcss,
                           DualArray1D<Real> &zscc, DualArray1D<Real> &zscs,
                           DualArray1D<Real> &zssc, DualArray1D<Real> &zsss,
                           Real xcoef[8], Real ycoef[8], Real zcoef[8]) {
  xccc.h_view(nmode) = xcoef[0]; xccs.h_view(nmode) = xcoef[1];
  xcsc.h_view(nmode) = xcoef[2]; xcss.h_view(nmode) = xcoef[3];
  xscc.h_view(nmode) = xcoef[4]; xscs.h_view(nmode) = xcoef[5];
  xssc.h_view(nmode) = xcoef[6]; xsss.h_view(nmode) = xcoef[7];
  yccc.h_view(nmode) = ycoef[0]; yccs.h_view(nmode) = ycoef[1];
  ycsc.h_view(nmode) = ycoef[2]; ycss.h_view(nmode) = ycoef[3];
  yscc.h_view(nmode) = ycoef[4]; yscs.h_view(nmode) = ycoef[5];
  yssc.h_view(nmode) = ycoef[6]; ysss.h_view(nmode) = ycoef[7];
  zccc.h_view(nmode) = zcoef[0]; zccs.h_view(nmode) = zcoef[1];
  zcsc.h_view(nmode) = zcoef[2]; zcss.h_view(nmode) = zcoef[3];
  zscc.h_view(nmode) = zcoef[4]; zscs.h_view(nmode) = zcoef[5];
  zssc.h_view(nmode) = zcoef[6]; zsss.h_view(nmode) = zcoef[7];
}
} // namespace

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceDriver::TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  force("force",1,1,1,1,1),
  force_tmp("force_tmp",1,1,1,1,1),
  xccc("xccc",1),xccs("xccs",1),xcsc("xcsc",1),xcss("xcss",1),
  xscc("xscc",1),xscs("xscs",1),xssc("xssc",1),xsss("xsss",1),
  yccc("yccc",1),yccs("yccs",1),ycsc("ycsc",1),ycss("ycss",1),
  yscc("yscc",1),yscs("yscs",1),yssc("yssc",1),ysss("ysss",1),
  zccc("zccc",1),zccs("zccs",1),zcsc("zcsc",1),zcss("zcss",1),
  zscc("zscc",1),zscs("zscs",1),zssc("zssc",1),zsss("zsss",1),
  kx_mode("kx_mode",1),ky_mode("ky_mode",1),kz_mode("kz_mode",1),
  xcos("xcos",1,1,1),xsin("xsin",1,1,1),ycos("ycos",1,1,1),
  ysin("ysin",1,1,1),zcos("zcos",1,1,1),zsin("zsin",1,1,1) {
  // allocate memory for force registers
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(force, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_tmp, nmb, 3, ncells3, ncells2, ncells1);

  // range of modes including, corresponding to kmin and kmax
  nlow = pin->GetOrAddInteger("turb_driving", "nlow", 1);
  nhigh = pin->GetOrAddInteger("turb_driving", "nhigh", 2);
  // Driving geometry.  Legacy driving_type = 0/1 is still accepted when the new
  // driving_geometry key is absent.
  bool has_geometry = pin->DoesParameterExist("turb_driving", "driving_geometry");
  bool has_legacy_type = pin->DoesParameterExist("turb_driving", "driving_type");
  if (has_geometry && has_legacy_type) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Use only one of <turb_driving>/driving_geometry "
              << "or legacy <turb_driving>/driving_type" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (has_legacy_type) {
    driving_geometry = pin->GetInteger("turb_driving", "driving_type");
    if (driving_geometry < 0 || driving_geometry > 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Legacy driving_type must be 0 or 1" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    std::string geometry = pin->GetOrAddString("turb_driving", "driving_geometry",
                                               "isotropic");
    if (geometry == "isotropic" || geometry == "iso" || geometry == "0") {
      driving_geometry = kGeometryIsotropic;
    } else if (geometry == "anisotropic" || geometry == "aniso" || geometry == "1") {
      driving_geometry = kGeometryAnisotropic;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unknown driving_geometry = '" << geometry << "'"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // power-law exponent for isotropic driving
  expo = pin->GetOrAddReal("turb_driving", "expo", 5.0/3.0);
  exp_prp = pin->GetOrAddReal("turb_driving", "exp_prp", 5.0/3.0);
  exp_prl = pin->GetOrAddReal("turb_driving", "exp_prl", 0.0);
  // Projection weight for isotropic driving: 1 is solenoidal, 0 is compressive,
  // and 0.5 gives the natural mixture.
  sol_weight = pin->GetOrAddReal("turb_driving", "sol_weight", 1.0);
  if (sol_weight < 0.0 || sol_weight > 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "sol_weight must be in [0, 1]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Driving profile.  Legacy parabola_driving=true is still accepted when the new
  // driving_profile key is absent.
  bool has_profile = pin->DoesParameterExist("turb_driving", "driving_profile");
  bool has_legacy_parabola = pin->DoesParameterExist("turb_driving", "parabola_driving");
  if (has_profile && has_legacy_parabola) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Use only one of <turb_driving>/driving_profile "
              << "or legacy <turb_driving>/parabola_driving" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (has_legacy_parabola) {
    bool parabola_driving = pin->GetBoolean("turb_driving", "parabola_driving");
    driving_profile = parabola_driving ? kProfileParabola : kProfilePowerLaw;
  } else {
    std::string profile = pin->GetOrAddString("turb_driving", "driving_profile",
                                              "powerlaw");
    if (profile == "powerlaw" || profile == "power_law" || profile == "power-law") {
      driving_profile = kProfilePowerLaw;
    } else if (profile == "parabola" || profile == "parabolic") {
      driving_profile = kProfileParabola;
    } else if (profile == "band" || profile == "rectangle" || profile == "constant" ||
               profile == "top_hat" || profile == "tophat") {
      driving_profile = kProfileBand;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unknown driving_profile = '" << profile << "'"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  parabola_peak = pin->GetOrAddReal("turb_driving", "parabola_peak", 2.0);
  parabola_width = pin->GetOrAddReal("turb_driving", "parabola_width", 1.0);
  if (driving_profile != kProfilePowerLaw && driving_geometry != kGeometryIsotropic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "driving_profile = band/parabola is currently implemented "
              << "only for driving_geometry = isotropic" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (driving_geometry != kGeometryIsotropic && sol_weight != 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "sol_weight != 1 is currently implemented only for "
              << "driving_geometry = isotropic" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (driving_profile == kProfileParabola && parabola_width <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "parabola_width must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // energy injection rate
  dedt = pin->GetOrAddReal("turb_driving", "dedt", 0.0);
  // correlation time
  tcorr = pin->GetOrAddReal("turb_driving", "tcorr", 0.0);

  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;

  mode_count = 0;

  int nkx, nky, nkz;
  Real nsqr;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
      for (nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        nsqr = 0.0;
        bool flag_prl = true;
        if (driving_geometry == kGeometryIsotropic) {
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
        } else if (driving_geometry == kGeometryAnisotropic) {
          nsqr = SQR(nkx) + SQR(nky);
          Real nprlsqr = SQR(nkz);
          if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
            flag_prl = true;
          } else {
            flag_prl = false;
          }
        }
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
          mode_count++;
        }
      }
    }
  }

  Kokkos::realloc(xccc, mode_count);
  Kokkos::realloc(xccs, mode_count);
  Kokkos::realloc(xcsc, mode_count);
  Kokkos::realloc(xcss, mode_count);
  Kokkos::realloc(xscc, mode_count);
  Kokkos::realloc(xscs, mode_count);
  Kokkos::realloc(xssc, mode_count);
  Kokkos::realloc(xsss, mode_count);

  Kokkos::realloc(yccc, mode_count);
  Kokkos::realloc(yccs, mode_count);
  Kokkos::realloc(ycsc, mode_count);
  Kokkos::realloc(ycss, mode_count);
  Kokkos::realloc(yscc, mode_count);
  Kokkos::realloc(yscs, mode_count);
  Kokkos::realloc(yssc, mode_count);
  Kokkos::realloc(ysss, mode_count);

  Kokkos::realloc(zccc, mode_count);
  Kokkos::realloc(zccs, mode_count);
  Kokkos::realloc(zcsc, mode_count);
  Kokkos::realloc(zcss, mode_count);
  Kokkos::realloc(zscc, mode_count);
  Kokkos::realloc(zscs, mode_count);
  Kokkos::realloc(zssc, mode_count);
  Kokkos::realloc(zsss, mode_count);

  Kokkos::realloc(kx_mode, mode_count);
  Kokkos::realloc(ky_mode, mode_count);
  Kokkos::realloc(kz_mode, mode_count);

  Kokkos::realloc(xcos, nmb, mode_count, ncells1);
  Kokkos::realloc(xsin, nmb, mode_count, ncells1);
  Kokkos::realloc(ycos, nmb, mode_count, ncells2);
  Kokkos::realloc(ysin, nmb, mode_count, ncells2);
  Kokkos::realloc(zcos, nmb, mode_count, ncells3);
  Kokkos::realloc(zsin, nmb, mode_count, ncells3);

  Initialize();
}

//----------------------------------------------------------------------------------------
// destructor

TurbulenceDriver::~TurbulenceDriver() {
}

//----------------------------------------------------------------------------------------
//! \fn  noid Initialize
//  \brief Function to initialize the driver

void TurbulenceDriver::Initialize() {
  Mesh *pm = pmy_pack->pmesh;
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  auto force_ = force;
  par_for("force_init_pgen",DevExeSpace(),
          0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    force_(m,n,k,j,i) = 0.0;
  });

  rstate.idum = -1;

  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto kz_mode_ = kz_mode;

  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;

  Real dkx, dky, dkz, kx, ky, kz;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  dkx = 2.0*M_PI/lx;
  dky = 2.0*M_PI/ly;
  dkz = 2.0*M_PI/lz;

  int nmode = 0;
  int nkx, nky, nkz;
  Real nsqr;
  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
      for (nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        nsqr = 0.0;
        bool flag_prl = true;
        if (driving_geometry == kGeometryIsotropic) {
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
        } else if (driving_geometry == kGeometryAnisotropic) {
          nsqr = SQR(nkx) + SQR(nky);
          Real nprlsqr = SQR(nkz);
          if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
            flag_prl = true;
          } else {
            flag_prl = false;
          }
        }
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
          kx = dkx*nkx;
          ky = dky*nky;
          kz = dkz*nkz;
          kx_mode_.h_view(nmode) = kx;
          ky_mode_.h_view(nmode) = ky;
          kz_mode_.h_view(nmode) = kz;
          nmode++;
        }
      }
    }
  }

  kx_mode_.template modify<HostMemSpace>();
  kx_mode_.template sync<DevExeSpace>();
  ky_mode_.template modify<HostMemSpace>();
  ky_mode_.template sync<DevExeSpace>();
  kz_mode_.template modify<HostMemSpace>();
  kz_mode_.template sync<DevExeSpace>();

  auto &size = pmy_pack->pmb->mb_size;

  par_for("xsin/xcos", DevExeSpace(),0,nmb-1,0,mode_count-1,is,ie,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real k1v = kx_mode_.d_view(n);
    xsin_(m,n,i) = sin(k1v*x1v);
    xcos_(m,n,i) = cos(k1v*x1v);
  });

  par_for("ysin/ycos", DevExeSpace(),0,nmb-1,0,mode_count-1,js,je,
  KOKKOS_LAMBDA(int m, int n, int j) {
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real k2v = ky_mode_.d_view(n);
    ysin_(m,n,j) = sin(k2v*x2v);
    ycos_(m,n,j) = cos(k2v*x2v);
    if (ncells2-1 == 0) {
      ysin_(m,n,j) = 0.0;
      ycos_(m,n,j) = 1.0;
    }
  });

  par_for("zsin/zcos", DevExeSpace(),0,nmb-1,0,mode_count-1,ks,ke,
  KOKKOS_LAMBDA(int m, int n, int k) {
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real k3v = kz_mode_.d_view(n);
    zsin_(m,n,k) = sin(k3v*x3v);
    zcos_(m,n,k) = cos(k3v*x3v);
    if (ncells3-1 == 0) {
      zsin_(m,n,k) = 0.0;
      zcos_(m,n,k) = 1.0;
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeModeEvolutionTasks
//  \brief Includes task in the operator split task list that constructs new modes with
//  random amplitudes and phases that can be used to evolve the force via an O-U process
//  Called by MeshBlockPack::AddPhysics() function

void TurbulenceDriver::IncludeInitializeModesTask(std::shared_ptr<TaskList> tl,
                                                  TaskID start) {
  auto id_init = tl->AddTask(&TurbulenceDriver::InitializeModes, this, start);
  auto id_add = tl->AddTask(&TurbulenceDriver::UpdateForcing, this, id_init);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeAddForcingTask
//  \brief includes task in the stage_run task list for adding random forcing to fluid
//  as an explicit source terms in each stage of integrator
//  Called by MeshBlockPack::AddPhysics() function

void TurbulenceDriver::IncludeAddForcingTask(std::shared_ptr<TaskList> tl, TaskID start) {
  // These must be inserted after the RK update, but before ordinary source terms.
  if (pmy_pack->pionn == nullptr) {
    if (pmy_pack->phydro != nullptr) {
      auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                              pmy_pack->phydro->id.rkupdt, pmy_pack->phydro->id.srctrms);
    }
    if (pmy_pack->pmhd != nullptr) {
      auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                              pmy_pack->pmhd->id.rkupdt, pmy_pack->pmhd->id.srctrms);
    }
  } else {
    auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                            pmy_pack->pionn->id.n_rkupdt, pmy_pack->pionn->id.impl);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn InitializeModes()
// \brief Initializes driving, and so is only executed once at start of calc.
// Cannot be included in constructor since (it seems) Kokkos::par_for not allowed in cons.

TaskStatus TurbulenceDriver::InitializeModes(Driver *pdrive, int stage) {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  auto &gindcs = pm->mesh_indcs;
  int &gnx1 = gindcs.nx1;
  int &gnx2 = gindcs.nx2;
  int &gnx3 = gindcs.nx3;

  // Now compute new force using new random amplitudes and phases

  // Zero out new force array
  auto force_tmp_ = force_tmp;
  int &nmb = pmy_pack->nmb_thispack;
  par_for("force_init", DevExeSpace(),0,nmb-1,0,2,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    force_tmp_(m,n,k,j,i) = 0.0;
  });

  int nlow_sqr = SQR(nlow);
  int nhigh_sqr = SQR(nhigh);
  auto mode_count_ = mode_count;

  auto xccc_ = xccc;
  auto xccs_ = xccs;
  auto xcsc_ = xcsc;
  auto xcss_ = xcss;
  auto xscc_ = xscc;
  auto xscs_ = xscs;
  auto xssc_ = xssc;
  auto xsss_ = xsss;

  auto yccc_ = yccc;
  auto yccs_ = yccs;
  auto ycsc_ = ycsc;
  auto ycss_ = ycss;
  auto yscc_ = yscc;
  auto yscs_ = yscs;
  auto yssc_ = yssc;
  auto ysss_ = ysss;

  auto zccc_ = zccc;
  auto zccs_ = zccs;
  auto zcsc_ = zcsc;
  auto zcss_ = zcss;
  auto zscc_ = zscc;
  auto zscs_ = zscs;
  auto zssc_ = zssc;
  auto zsss_ = zsss;

  Real dkx, dky, dkz, kx, ky, kz;
  Real iky;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  dkx = 2.0*M_PI/lx;
  dky = 2.0*M_PI/ly;
  dkz = 2.0*M_PI/lz;

  Real &ex = expo;
  Real &ex_prp = exp_prp;
  Real &ex_prl = exp_prl;
  Real &sol_weight_ = sol_weight;
  Real &parabola_peak_ = parabola_peak;
  Real &parabola_width_ = parabola_width;
  int &driving_profile_ = driving_profile;
  Real norm, kprl, kprp, kiso;

  int nmode = 0;
  int nkx, nky, nkz, nsqr;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
      for (nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        norm = 0.0;
        nsqr = 0.0;
        bool flag_prl = true;
        if (driving_geometry == kGeometryIsotropic) {
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
        } else if (driving_geometry == kGeometryAnisotropic) {
          nsqr = SQR(nkx) + SQR(nky);
          Real nprlsqr = SQR(nkz);
          if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
            flag_prl = true;
          } else {
            flag_prl = false;
          }
        }
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
          kx = dkx*nkx;
          ky = dky*nky;
          kz = dkz*nkz;

          // Generate Fourier amplitudes
          if (driving_geometry == kGeometryIsotropic) {
            kiso = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
            if (kiso > 1e-16) {
              if (driving_profile_ == kProfileParabola) {
                Real nmag = sqrt(static_cast<Real>(nsqr));
                Real profile = 1.0 - SQR((nmag - parabola_peak_)/parabola_width_);
                norm = (profile > 0.0) ? sqrt(profile) : 0.0;
              } else if (driving_profile_ == kProfileBand) {
                norm = 1.0;
              } else {
                norm = 1.0/pow(kiso,(ex+2.0)/2.0);
              }
            } else {
              norm = 0.0;
            }
            Real xcoef[8], ycoef[8], zcoef[8];
            for (int n=0; n<8; ++n) {
              if (BasisActive(n, nkx, nky, nkz)) {
                xcoef[n] = RanGaussianSt(&(rstate));
                ycoef[n] = RanGaussianSt(&(rstate));
                zcoef[n] = RanGaussianSt(&(rstate));
              } else {
                xcoef[n] = 0.0;
                ycoef[n] = 0.0;
                zcoef[n] = 0.0;
              }
            }
            ApplySolenoidalWeight(kx, ky, kz, sol_weight_, xcoef, ycoef, zcoef);
            StoreModeCoefficients(nmode, xccc_, xccs_, xcsc_, xcss_, xscc_, xscs_,
                                  xssc_, xsss_, yccc_, yccs_, ycsc_, ycss_, yscc_,
                                  yscs_, yssc_, ysss_, zccc_, zccs_, zcsc_, zcss_,
                                  zscc_, zscs_, zssc_, zsss_, xcoef, ycoef, zcoef);
          } else if (driving_geometry == kGeometryAnisotropic) {
            kprl = sqrt(SQR(kx));
            kprp = sqrt(SQR(ky) + SQR(kz));
            if (kprl > 1e-16 && kprp > 1e-16) {
              norm = 1.0/pow(kprp,(ex_prp+1.0)/2.0)/pow(kprl,ex_prl/2.0);
            } else {
              norm = 0.0;
            }

            if (nky != 0) {
              iky = 1.0/(dky*((Real) nky));

              xccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              xccs_.h_view(nmode) = RanGaussianSt(&(rstate));
              xcsc_.h_view(nmode) = RanGaussianSt(&(rstate));
              xcss_.h_view(nmode) = RanGaussianSt(&(rstate));
              xscc_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              xscs_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              xssc_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              xsss_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));

              // incompressibility
              yccc_.h_view(nmode) =  iky*(kx*xssc_.h_view(nmode));
              yccs_.h_view(nmode) =  iky*(kx*xsss_.h_view(nmode));
              ycsc_.h_view(nmode) = -iky*(kx*xscc_.h_view(nmode));
              ycss_.h_view(nmode) = -iky*(kx*xscs_.h_view(nmode));
              yscc_.h_view(nmode) = -iky*(kx*xcsc_.h_view(nmode));
              yscs_.h_view(nmode) = -iky*(kx*xcss_.h_view(nmode));
              yssc_.h_view(nmode) =  iky*(kx*xccc_.h_view(nmode));
              ysss_.h_view(nmode) =  iky*(kx*xccs_.h_view(nmode));

              zccc_.h_view(nmode) = 0.0;
              zccs_.h_view(nmode) = 0.0;
              zcsc_.h_view(nmode) = 0.0;
              zcss_.h_view(nmode) = 0.0;
              zscc_.h_view(nmode) = 0.0;
              zscs_.h_view(nmode) = 0.0;
              zssc_.h_view(nmode) = 0.0;
              zsss_.h_view(nmode) = 0.0;
            } else {  // ky == 0
              yccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              yscc_.h_view(nmode) = RanGaussianSt(&(rstate));
              ycsc_.h_view(nmode) = 0.0;
              yssc_.h_view(nmode) = 0.0;
              yccs_.h_view(nmode) = 0.0;
              ycss_.h_view(nmode) = 0.0;
              yscs_.h_view(nmode) = 0.0;
              ysss_.h_view(nmode) = 0.0;

              // incompressibility
              xccc_.h_view(nmode) = 0.0;
              xscc_.h_view(nmode) = 0.0;
              xcsc_.h_view(nmode) = 0.0;
              xssc_.h_view(nmode) = 0.0;
              xccs_.h_view(nmode) = 0.0;
              xscs_.h_view(nmode) = 0.0;
              xcss_.h_view(nmode) = 0.0;
              xsss_.h_view(nmode) = 0.0;

              zccc_.h_view(nmode) = 0.0;
              zscc_.h_view(nmode) = 0.0;
              zcsc_.h_view(nmode) = 0.0;
              zssc_.h_view(nmode) = 0.0;
              zccs_.h_view(nmode) = 0.0;
              zcss_.h_view(nmode) = 0.0;
              zscs_.h_view(nmode) = 0.0;
              zsss_.h_view(nmode) = 0.0;
            }
          }
          // normalization
          xccc_.h_view(nmode) *= norm;
          xscc_.h_view(nmode) *= norm;
          xcsc_.h_view(nmode) *= norm;
          xssc_.h_view(nmode) *= norm;
          xccs_.h_view(nmode) *= norm;
          xscs_.h_view(nmode) *= norm;
          xcss_.h_view(nmode) *= norm;
          xsss_.h_view(nmode) *= norm;
          yccc_.h_view(nmode) *= norm;
          yscc_.h_view(nmode) *= norm;
          ycsc_.h_view(nmode) *= norm;
          yssc_.h_view(nmode) *= norm;
          yccs_.h_view(nmode) *= norm;
          yscs_.h_view(nmode) *= norm;
          ycss_.h_view(nmode) *= norm;
          ysss_.h_view(nmode) *= norm;
          zccc_.h_view(nmode) *= norm;
          zscc_.h_view(nmode) *= norm;
          zcsc_.h_view(nmode) *= norm;
          zssc_.h_view(nmode) *= norm;
          zccs_.h_view(nmode) *= norm;
          zscs_.h_view(nmode) *= norm;
          zcss_.h_view(nmode) *= norm;
          zsss_.h_view(nmode) *= norm;

          nmode++;
        }
      }
    }
  }

  xccc_.template modify<HostMemSpace>();
  xccc_.template sync<DevExeSpace>();
  xccs_.template modify<HostMemSpace>();
  xccs_.template sync<DevExeSpace>();
  xcsc_.template modify<HostMemSpace>();
  xcsc_.template sync<DevExeSpace>();
  xcss_.template modify<HostMemSpace>();
  xcss_.template sync<DevExeSpace>();
  xscc_.template modify<HostMemSpace>();
  xscc_.template sync<DevExeSpace>();
  xscs_.template modify<HostMemSpace>();
  xscs_.template sync<DevExeSpace>();
  xssc_.template modify<HostMemSpace>();
  xssc_.template sync<DevExeSpace>();
  xsss_.template modify<HostMemSpace>();
  xsss_.template sync<DevExeSpace>();

  yccc_.template modify<HostMemSpace>();
  yccc_.template sync<DevExeSpace>();
  yccs_.template modify<HostMemSpace>();
  yccs_.template sync<DevExeSpace>();
  ycsc_.template modify<HostMemSpace>();
  ycsc_.template sync<DevExeSpace>();
  ycss_.template modify<HostMemSpace>();
  ycss_.template sync<DevExeSpace>();
  yscc_.template modify<HostMemSpace>();
  yscc_.template sync<DevExeSpace>();
  yscs_.template modify<HostMemSpace>();
  yscs_.template sync<DevExeSpace>();
  yssc_.template modify<HostMemSpace>();
  yssc_.template sync<DevExeSpace>();
  ysss_.template modify<HostMemSpace>();
  ysss_.template sync<DevExeSpace>();

  zccc_.template modify<HostMemSpace>();
  zccc_.template sync<DevExeSpace>();
  zccs_.template modify<HostMemSpace>();
  zccs_.template sync<DevExeSpace>();
  zcsc_.template modify<HostMemSpace>();
  zcsc_.template sync<DevExeSpace>();
  zcss_.template modify<HostMemSpace>();
  zcss_.template sync<DevExeSpace>();
  zscc_.template modify<HostMemSpace>();
  zscc_.template sync<DevExeSpace>();
  zscs_.template modify<HostMemSpace>();
  zscs_.template sync<DevExeSpace>();
  zssc_.template modify<HostMemSpace>();
  zssc_.template sync<DevExeSpace>();
  zsss_.template modify<HostMemSpace>();
  zsss_.template sync<DevExeSpace>();

  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;

  for (int n=0; n<mode_count_; n++) {
    par_for("force_compute", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      force_tmp_(m,0,k,j,i) += xccc_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,0,k,j,i) += xccs_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,0,k,j,i) += xcsc_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,0,k,j,i) += xcss_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,0,k,j,i) += xscc_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,0,k,j,i) += xscs_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,0,k,j,i) += xssc_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,0,k,j,i) += xsss_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);

      force_tmp_(m,1,k,j,i) += yccc_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,1,k,j,i) += yccs_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,1,k,j,i) += ycsc_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,1,k,j,i) += ycss_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,1,k,j,i) += yscc_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,1,k,j,i) += yscs_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,1,k,j,i) += yssc_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,1,k,j,i) += ysss_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);

      force_tmp_(m,2,k,j,i) += zccc_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,2,k,j,i) += zccs_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,2,k,j,i) += zcsc_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,2,k,j,i) += zcss_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,2,k,j,i) += zscc_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,2,k,j,i) += zscs_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,2,k,j,i) += zssc_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,2,k,j,i) += zsss_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);
    });
  }

  DvceArray5D<Real> u0, u0_;
  if (pmy_pack->phydro != nullptr) u0 = (pmy_pack->phydro->u0);
  if (pmy_pack->pmhd != nullptr) u0 = (pmy_pack->pmhd->u0);
  bool flag_twofl = false;
  if (pmy_pack->pionn != nullptr) {
    u0 = (pmy_pack->phydro->u0);
    u0_ = (pmy_pack->pmhd->u0);
    flag_twofl = true;
  }

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;

  Kokkos::parallel_reduce("net_mom_1", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1,
                                Real &sum_t2, Real &sum_t3) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real den = u0(m,IDN,k,j,i);
    if (flag_twofl) {
      den += u0_(m,IDN,k,j,i);
    }
    sum_t0 += den;
    sum_t1 += den*force_tmp_(m,0,k,j,i);
    sum_t2 += den*force_tmp_(m,1,k,j,i);
    sum_t3 += den*force_tmp_(m,2,k,j,i);
  }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
     Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));


#if MPI_PARALLEL_ENABLED
  Real m[4], gm[4];
  m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
  MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
#endif

  par_for("force_remove_net_mom", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_tmp_(m,0,k,j,i) -= t1/t0;
    force_tmp_(m,1,k,j,i) -= t2/t0;
    force_tmp_(m,2,k,j,i) -= t3/t0;
  });

  t0 = 0.0;
  t1 = 0.0;
  Kokkos::parallel_reduce("net_mom_2", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real den  = u0(m,IDN,k,j,i);
    Real mom1 = u0(m,IM1,k,j,i);
    Real mom2 = u0(m,IM2,k,j,i);
    Real mom3 = u0(m,IM3,k,j,i);
    if (flag_twofl) {
      den  += u0_(m,IDN,k,j,i);
      mom1 += u0_(m,IM1,k,j,i);
      mom2 += u0_(m,IM2,k,j,i);
      mom3 += u0_(m,IM3,k,j,i);
    }
    Real v1 = force_tmp_(m,0,k,j,i);
    Real v2 = force_tmp_(m,1,k,j,i);
    Real v3 = force_tmp_(m,2,k,j,i);

    sum_t0 += den*(v1*v1+v2*v2+v3*v3);
    sum_t1 += mom1*v1+mom2*v2+mom3*v3;
  }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1));

#if MPI_PARALLEL_ENABLED
  m[0] = t0; m[1] = t1;
  MPI_Allreduce(m, gm, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  t0 = gm[0]; t1 = gm[1];
#endif

  t0 = std::max(t0, 1.0e-20);

  Real m0 = t0, m1 = t1;
  Real dt = pm->dt;
  Real dvol = 1.0/(gnx1*gnx2*gnx3);
  m0 = 0.5*m0*dvol*dt;
  m1 = m1*dvol;

  Real s;
  if (m1 >= 0) {
    s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  } else {
    s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  }
  if (m0 == 0.0) s = 0.0;

  par_for("force_norm", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_tmp_(m,0,k,j,i) *= s;
    force_tmp_(m,1,k,j,i) *= s;
    force_tmp_(m,2,k,j,i) *= s;
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn update forcing

TaskStatus TurbulenceDriver::UpdateForcing(Driver *pdrive, int stage) {
  (void)pdrive;
  (void)stage;

  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nmb = pmy_pack->nmb_thispack;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  auto &gindcs = pm->mesh_indcs;
  int &gnx1 = gindcs.nx1;
  int &gnx2 = gindcs.nx2;
  int &gnx3 = gindcs.nx3;

  Real dt = pm->dt;
  Real fcorr, gcorr;
  if (tcorr <= 1e-6) {  // use whitenoise
    fcorr = 0.0;
    gcorr = 1.0;
  } else {
    fcorr = std::exp(-dt/tcorr);
    gcorr = std::sqrt(1.0 - fcorr*fcorr);
  }

  auto force_ = force;
  auto force_tmp_ = force_tmp;

  par_for("force_OU_process",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_(m,0,k,j,i) = fcorr*force_(m,0,k,j,i) + gcorr*force_tmp_(m,0,k,j,i);
    force_(m,1,k,j,i) = fcorr*force_(m,1,k,j,i) + gcorr*force_tmp_(m,1,k,j,i);
    force_(m,2,k,j,i) = fcorr*force_(m,2,k,j,i) + gcorr*force_tmp_(m,2,k,j,i);
  });

  DvceArray5D<Real> u0, u0_;
  if (pmy_pack->phydro != nullptr) u0 = (pmy_pack->phydro->u0);
  if (pmy_pack->pmhd != nullptr) u0 = (pmy_pack->pmhd->u0);
  bool flag_twofl = false;
  if (pmy_pack->pionn != nullptr) {
    u0 = (pmy_pack->phydro->u0);
    u0_ = (pmy_pack->pmhd->u0);
    flag_twofl = true;
  }

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;

  Kokkos::parallel_reduce("force_net_mom", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1,
                                Real &sum_t2, Real &sum_t3) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real den = u0(m,IDN,k,j,i);
    if (flag_twofl) {
      den += u0_(m,IDN,k,j,i);
    }
    sum_t0 += den;
    sum_t1 += den*force_(m,0,k,j,i);
    sum_t2 += den*force_(m,1,k,j,i);
    sum_t3 += den*force_(m,2,k,j,i);
  }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
     Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));

#if MPI_PARALLEL_ENABLED
  Real m[4], gm[4];
  m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
  MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
#endif

  par_for("force_remove_net_mom", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_(m,0,k,j,i) -= t1/t0;
    force_(m,1,k,j,i) -= t2/t0;
    force_(m,2,k,j,i) -= t3/t0;
  });

  t0 = 0.0;
  t1 = 0.0;
  Kokkos::parallel_reduce("force_dedt_norm", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real den  = u0(m,IDN,k,j,i);
    Real mom1 = u0(m,IM1,k,j,i);
    Real mom2 = u0(m,IM2,k,j,i);
    Real mom3 = u0(m,IM3,k,j,i);
    if (flag_twofl) {
      den  += u0_(m,IDN,k,j,i);
      mom1 += u0_(m,IM1,k,j,i);
      mom2 += u0_(m,IM2,k,j,i);
      mom3 += u0_(m,IM3,k,j,i);
    }
    Real v1 = force_(m,0,k,j,i);
    Real v2 = force_(m,1,k,j,i);
    Real v3 = force_(m,2,k,j,i);

    sum_t0 += den*(v1*v1+v2*v2+v3*v3);
    sum_t1 += mom1*v1+mom2*v2+mom3*v3;
  }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1));

#if MPI_PARALLEL_ENABLED
  m[0] = t0; m[1] = t1;
  MPI_Allreduce(m, gm, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  t0 = gm[0]; t1 = gm[1];
#endif

  t0 = std::max(t0, 1.0e-20);

  Real m0 = t0, m1 = t1;
  Real dvol = 1.0/(gnx1*gnx2*gnx3);
  m0 = 0.5*m0*dvol*dt;
  m1 = m1*dvol;

  Real s;
  if (m1 >= 0) {
    s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  } else {
    s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  }
  if (dedt == 0.0 || m0 == 0.0) s = 0.0;

  par_for("force_norm", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_(m,0,k,j,i) *= s;
    force_(m,1,k,j,i) *= s;
    force_(m,2,k,j,i) *= s;
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn apply forcing

TaskStatus TurbulenceDriver::AddForcing(Driver *pdrive, int stage) {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nmb = pmy_pack->nmb_thispack;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  Real dt = pm->dt;
  Real bdt = (pdrive->beta[stage-1])*dt;

  EquationOfState *peos;

  DvceArray5D<Real> u0, u0_;
  DvceArray5D<Real> w0, w0_;
  DvceFaceFld4D<Real> *bcc0;
  bool primary_is_ideal = false;
  bool secondary_is_ideal = false;
  if (pmy_pack->phydro != nullptr) {
    u0 = (pmy_pack->phydro->u0);
    w0 = (pmy_pack->phydro->w0);
    peos = (pmy_pack->phydro->peos);
    primary_is_ideal = pmy_pack->phydro->peos->eos_data.is_ideal;
  }
  if (pmy_pack->pmhd != nullptr) {
    u0 = (pmy_pack->pmhd->u0);
    w0 = (pmy_pack->pmhd->w0);
    peos = pmy_pack->pmhd->peos;
    primary_is_ideal = pmy_pack->pmhd->peos->eos_data.is_ideal;
  }
  if (pmy_pack->pmhd != nullptr) bcc0 = &(pmy_pack->pmhd->b0);
  bool flag_twofl = false;
  if (pmy_pack->pionn != nullptr) {
    u0 = (pmy_pack->phydro->u0);
    u0_ = (pmy_pack->pmhd->u0);
    w0 = (pmy_pack->phydro->w0);
    w0_ = (pmy_pack->pmhd->w0);
    peos = pmy_pack->phydro->peos;
    primary_is_ideal = pmy_pack->phydro->peos->eos_data.is_ideal;
    secondary_is_ideal = pmy_pack->pmhd->peos->eos_data.is_ideal;
    flag_twofl = true;
  }

  bool flag_relativistic = pmy_pack->pcoord->is_special_relativistic;

  auto force_ = force;

  par_for("push",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real a1 = force_(m,0,k,j,i);
    Real a2 = force_(m,1,k,j,i);
    Real a3 = force_(m,2,k,j,i);

    Real den = w0(m,IDN,k,j,i);
    Real ux = w0(m,IVX,k,j,i);
    Real uy = w0(m,IVY,k,j,i);
    Real uz = w0(m,IVZ,k,j,i);
    Real Fv = a1*ux + a2*uy + a3*uz;
    if (flag_relativistic) {
      // Compute Lorentz factor
      Real ut = 1. + ux*ux + uy*uy + uz*uz;
      ut = sqrt(ut);
      den /= ut;
      Fv /= ut;
    }
    u0(m,IM1,k,j,i) += den*a1*bdt;
    u0(m,IM2,k,j,i) += den*a2*bdt;
    u0(m,IM3,k,j,i) += den*a3*bdt;
    if (primary_is_ideal) {
      u0(m,IEN,k,j,i) += (Fv + 0.5*(a1*a1 + a2*a2 + a3*a3)*bdt)*den*bdt;
    }

    if (flag_twofl) {
      den = w0_(m,IDN,k,j,i);
      ux = w0_(m,IVX,k,j,i);
      uy = w0_(m,IVY,k,j,i);
      uz = w0_(m,IVZ,k,j,i);
      Fv = a1*ux + a2*uy + a3*uz;
      if (flag_relativistic) {
        Real ut = 1. + ux*ux + uy*uy + uz*uz;
        ut = sqrt(ut);
        den /= ut;
        Fv /= ut;
      }
      u0_(m,IM1,k,j,i) += den*a1*bdt;
      u0_(m,IM2,k,j,i) += den*a2*bdt;
      u0_(m,IM3,k,j,i) += den*a3*bdt;
      if (secondary_is_ideal) {
        u0_(m,IEN,k,j,i) += (Fv + 0.5*(a1*a1 + a2*a2 + a3*a3)*bdt)*den*bdt;
      }
    }
  });

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  // Relativistic case will require a Lorentz transformation
  if (flag_relativistic) {
    if (pmy_pack->pmhd != nullptr) {
      auto &b = *bcc0;
      auto &eos = peos->eos_data;

      par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // load single state conserved variables
        MHDCons1D u;
        u.d = u0(m,IDN,k,j,i);
        u.mx = u0(m,IM1,k,j,i);
        u.my = u0(m,IM2,k,j,i);
        u.mz = u0(m,IM3,k,j,i);
        u.e = u0(m,IEN,k,j,i);

        u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

        // Compute (S^i S_i) (eqn C2)
        Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);
        Real b2 = SQR(u.bx) + SQR(u.by) + SQR(u.bz);
        Real rpar = (u.bx*u.mx + u.by*u.my + u.bz*u.mz)/u.d;

        // call c2p function
        // (inline function in ideal_c2p_mhd.hpp file)
        HydPrim1D w;
        bool dfloor_used = false, efloor_used = false;
        //bool vceiling_used = false;
        bool c2p_failure = false;
        int iter_used = 0;
        SingleC2P_IdealSRMHD(u, eos, s2, b2, rpar, w, dfloor_used,
                             efloor_used, c2p_failure, iter_used);
        // apply velocity ceiling if necessary
        Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
        if (lor > eos.gamma_max) {
          //vceiling_used = true;
          Real factor = sqrt((SQR(eos.gamma_max) - 1.0) / (SQR(lor) - 1.0));
          w.vx *= factor;
          w.vy *= factor;
          w.vz *= factor;
        }

        // Temporarily store primitives in conserved state
        u0(m,IDN,k,j,i) = w.d;
        u0(m,IM1,k,j,i) = w.vx;
        u0(m,IM2,k,j,i) = w.vy;
        u0(m,IM3,k,j,i) = w.vz;
        u0(m,IEN,k,j,i) = w.e;
      });
    } else {
      auto &eos = peos->eos_data;

      par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // load single state conserved variables
        HydCons1D u;
        u.d = u0(m,IDN,k,j,i);
        u.mx = u0(m,IM1,k,j,i);
        u.my = u0(m,IM2,k,j,i);
        u.mz = u0(m,IM3,k,j,i);
        u.e = u0(m,IEN,k,j,i);

        // Compute (S^i S_i) (eqn C2)
        Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);

        // call c2p function
        // (inline function in ideal_c2p_mhd.hpp file)
        HydPrim1D w;
        bool dfloor_used = false, efloor_used = false;
        //bool vceiling_used = false;
        bool c2p_failure = false;
        int iter_used = 0;
        SingleC2P_IdealSRHyd(u, eos, s2, w, dfloor_used, efloor_used,
                             c2p_failure, iter_used);
        // apply velocity ceiling if necessary
        Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
        if (lor > eos.gamma_max) {
          //vceiling_used = true;
          Real factor = sqrt((SQR(eos.gamma_max) - 1.0) / (SQR(lor) - 1.0));
          w.vx *= factor;
          w.vy *= factor;
          w.vz *= factor;
        }

        u0(m,IDN,k,j,i) = w.d;
        u0(m,IM1,k,j,i) = w.vx;
        u0(m,IM2,k,j,i) = w.vy;
        u0(m,IM3,k,j,i) = w.vz;
        u0(m,IEN,k,j,i) = w.e;
      });
    }

    // remove net momentum
    Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
    Kokkos::parallel_reduce("net_mom_3", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1, Real &sum_t2,
                  Real &sum_t3) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real u_t = sqrt(1. + u0(m,IVX,k,j,i)*u0(m,IVX,k,j,i) +
                           u0(m,IVY,k,j,i)*u0(m,IVY,k,j,i) +
                           u0(m,IVZ,k,j,i)*u0(m,IVZ,k,j,i));

      Real den = u0(m,IDN,k,j,i)*u_t;
      Real mom1 = den*u0(m,IVX,k,j,i);
      Real mom2 = den*u0(m,IVY,k,j,i);
      Real mom3 = den*u0(m,IVZ,k,j,i);

      sum_t0 += den;
      sum_t1 += mom1;
      sum_t2 += mom2;
      sum_t3 += mom3;
    }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
       Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));

#if MPI_PARALLEL_ENABLED
    Real m[4], gm[4];
    m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
    MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
#endif

    // Compute average velocity
    Real uA_x = t1/t0;
    Real uA_y = t2/t0;
    Real uA_z = t3/t0;

    Real uA_0 = sqrt(1. + uA_x*uA_x + uA_y*uA_y + uA_z*uA_z);
    Real betaA = sqrt(uA_x*uA_x + uA_y*uA_y + uA_z*uA_z)/uA_0;

    Real vx = uA_x/uA_0;
    Real vy = uA_y/uA_0;
    Real vz = uA_z/uA_0;

    // LIMIT temp

    if (pmy_pack->pmhd != nullptr) {
      auto &b = *bcc0;
      auto &eos = peos->eos_data;

      par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m,IEN,k,j,i) = fmin(u0(m,IEN,k,j,i), 40.*u0(m,IDN,k,j,i));

        // load single state conserved variables
        MHDPrim1D u;
        u.d = u0(m,IDN,k,j,i);
        u.vx = u0(m,IM1,k,j,i);
        u.vy = u0(m,IM2,k,j,i);
        u.vz = u0(m,IM3,k,j,i);
        u.e = u0(m,IEN,k,j,i);

        u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

        HydCons1D u_out;
        SingleP2C_IdealSRMHD(u, eos.gamma, u_out);

        Real en = u_out.d + u_out.e;
        Real sx = u_out.mx;
        Real sy = u_out.my;
        Real sz = u_out.mz;

        Real dens = u_out.d;

        auto &w = u;

        Real lorentz = sqrt(1. + w.vx*w.vx + w.vy*w.vy + w.vz*w.vz);
        Real beta = sqrt(w.vx*w.vx + w.vy*w.vy + w.vz*w.vz)/lorentz;

        u0(m,IDN,k,j,i) = dens;  // *uA_0*(1.-beta*betaA);

        // Does not require knowledge of v
        u0(m,IEN,k,j,i) = uA_0*en - uA_0*(sx*vx + sy*vy + sz*vz);
        u0(m,IEN,k,j,i) -= u0(m,IDN,k,j,i);

        u0(m,IM1,k,j,i) = sx + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vx;
        u0(m,IM2,k,j,i) = sy + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vy;
        u0(m,IM3,k,j,i) = sz + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vz;

        u0(m,IM1,k,j,i) -= uA_0*en*vx;
        u0(m,IM2,k,j,i) -= uA_0*en*vy;
        u0(m,IM3,k,j,i) -= uA_0*en*vz;
      });
    } else {
      auto &eos = peos->eos_data;

      par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m,IEN,k,j,i) = fmin(u0(m,IEN,k,j,i), 40.*u0(m,IDN,k,j,i));

        // load single state conserved variables
        HydPrim1D u;
        u.d = u0(m,IDN,k,j,i);
        u.vx = u0(m,IM1,k,j,i);
        u.vy = u0(m,IM2,k,j,i);
        u.vz = u0(m,IM3,k,j,i);
        u.e = u0(m,IEN,k,j,i);

        HydCons1D u_out;
        SingleP2C_IdealSRHyd(u, eos.gamma, u_out);

        Real en = u_out.d + u_out.e;
        Real sx = u_out.mx;
        Real sy = u_out.my;
        Real sz = u_out.mz;

        Real dens = u_out.d;

        auto &w = u;

        Real lorentz = sqrt(1. + w.vx*w.vx + w.vy*w.vy + w.vz*w.vz);
        Real beta = sqrt(w.vx*w.vx + w.vy*w.vy + w.vz*w.vz)/lorentz;

        u0(m,IDN,k,j,i) = dens;  //*uA_0*(1.-beta*betaA);

        // Does not require knowledge of v
        u0(m,IEN,k,j,i) = uA_0*en - uA_0*(sx*vx + sy*vy + sz*vz);
        u0(m,IEN,k,j,i) -= u0(m,IDN,k,j,i);
        u0(m,IM1,k,j,i) = sx + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vx;
        u0(m,IM2,k,j,i) = sy + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vy;
        u0(m,IM3,k,j,i) = sz + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vz;
        u0(m,IM1,k,j,i) -= uA_0*en*vx;
        u0(m,IM2,k,j,i) -= uA_0*en*vy;
        u0(m,IM3,k,j,i) -= uA_0*en*vz;
      });
    }

  } else {
    // remove net momentum
    Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
    Kokkos::parallel_reduce("net_mom_3", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1, Real &sum_t2,
                  Real &sum_t3) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real den = u0(m,IDN,k,j,i);
      Real mom1 = u0(m,IM1,k,j,i);
      Real mom2 = u0(m,IM2,k,j,i);
      Real mom3 = u0(m,IM3,k,j,i);
      if (flag_twofl) {
        den += u0_(m,IDN,k,j,i);
        mom1 += u0_(m,IM1,k,j,i);
        mom2 += u0_(m,IM2,k,j,i);
        mom3 += u0_(m,IM3,k,j,i);
      }

      sum_t0 += den;
      sum_t1 += mom1;
      sum_t2 += mom2;
      sum_t3 += mom3;
    }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
       Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));

#if MPI_PARALLEL_ENABLED
    Real m[4], gm[4];
    m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
    MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
#endif

    par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real den = u0(m,IDN,k,j,i);

      if (flag_relativistic) {
        auto &ux = w0(m,IVX,k,j,i);
        auto &uy = w0(m,IVY,k,j,i);
        auto &uz = w0(m,IVZ,k,j,i);

        Real ut = 1. + ux*ux + uy*uy + uz*uz;
        ut = sqrt(ut);
        den /= ut;

        Real Fv_avg = den*(t1*ux + t2*uy + t3*uz)/ut/t0;

        u0(m,IEN,k,j,i) -= Fv_avg;
      }
      u0(m,IM1,k,j,i) -= den*t1/t0;
      u0(m,IM2,k,j,i) -= den*t2/t0;
      u0(m,IM3,k,j,i) -= den*t3/t0;
      if (flag_twofl) {
        den = u0_(m,IDN,k,j,i);
        u0_(m,IM1,k,j,i) -= den*t1/t0;
        u0_(m,IM2,k,j,i) -= den*t2/t0;
        u0_(m,IM3,k,j,i) -= den*t3/t0;
      }
    });
  }

  return TaskStatus::complete;
}
