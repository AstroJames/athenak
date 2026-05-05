//========================================================================================
// AthenaK: astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectral_2d_field_gen.cpp
//! \brief Implements a spectral generator for 2D divergence-free vector fields.

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"
#include "utils/spectral_2d_field_gen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

constexpr Real kPi = 3.14159265358979323846264338327950288;

}  // namespace

Spectral2DFieldGenerator::Spectral2DFieldGenerator(MeshBlockPack *pmbp, ParameterInput *pin,
                                                   const std::string &block) :
    mode_count(0),
    pmy_pack(pmbp),
    kx_mode("k_spec_x", 1),
    ky_mode("k_spec_y", 1),
    az_cc("az_cc", 1),
    az_cs("az_cs", 1),
    az_sc("az_sc", 1),
    az_ss("az_ss", 1),
    xcos("xcos", 1, 1, 1),
    xsin("xsin", 1, 1, 1),
    ycos("ycos", 1, 1, 1),
    ysin("ysin", 1, 1, 1) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (indcs.nx3 != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Spectral2DFieldGenerator requires nx3 = 1." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string spec = pin->GetOrAddString(block, "spectrum", "gaussian");
  if (spec.compare("band") == 0) {
    spectrum_form = SpectrumForm::band;
  } else {
    spectrum_form = SpectrumForm::gaussian;
  }

  k_peak = pin->GetOrAddReal(block, "k_peak", 80.0);
  k_width = pin->GetOrAddReal(block, "k_width", 10.0);
  if (pin->DoesParameterExist(block, "k_min") || pin->DoesParameterExist(block, "k_max")) {
    k_min = pin->GetOrAddReal(block, "k_min", 0.0);
    k_max = pin->GetReal(block, "k_max");
  } else if (spectrum_form == SpectrumForm::band) {
    Real delta = pin->GetOrAddReal("problem", "a0", 1.0/std::sqrt(1.0e3));
    Real lx = pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min;
    Real ly = pmy_pack->pmesh->mesh_size.x2max - pmy_pack->pmesh->mesh_size.x2min;
    Real dx1 = lx/static_cast<Real>(pmy_pack->pmesh->mesh_indcs.nx1);
    Real dx2 = ly/static_cast<Real>(pmy_pack->pmesh->mesh_indcs.nx2);
    if (delta <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "problem/a0 must be positive." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    k_min = 1.0/delta;
    k_max = std::sqrt(SQR(kPi/dx1) + SQR(kPi/dx2));
  } else {
    if (k_width <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "spectral_ic/k_width must be positive." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    k_min = fmax(0.0, k_peak - k_width);
    k_max = k_peak + k_width;
  }
  if (k_min < 0.0 || k_max <= k_min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "spectral_ic requires 0 <= k_min < k_max." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int64_t iseed = -std::abs(pin->GetOrAddInteger("problem", "rng_seed", 42));
  if (pin->DoesParameterExist(block, "iseed")) {
    iseed = pin->GetInteger(block, "iseed");
    if (iseed > 0) iseed = -iseed;
  }
  rstate.idum = iseed;

  CountModes();

  if (mode_count == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "No spectral modes fall inside the requested shell." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int nmb = pmy_pack->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);

  Kokkos::realloc(kx_mode, mode_count);
  Kokkos::realloc(ky_mode, mode_count);
  Kokkos::realloc(az_cc, mode_count);
  Kokkos::realloc(az_cs, mode_count);
  Kokkos::realloc(az_sc, mode_count);
  Kokkos::realloc(az_ss, mode_count);
  Kokkos::realloc(xcos, nmb, mode_count, ncells1);
  Kokkos::realloc(xsin, nmb, mode_count, ncells1);
  Kokkos::realloc(ycos, nmb, mode_count, ncells2);
  Kokkos::realloc(ysin, nmb, mode_count, ncells2);

  GenerateModeCoefficients();
  PrecomputeTrigTables();
}

Real Spectral2DFieldGenerator::ModeAmplitude(Real kmag) const {
  if (kmag <= std::numeric_limits<Real>::epsilon()) return 0.0;
  if (spectrum_form == SpectrumForm::band) return 1.0;
  Real arg = (kmag - k_peak)/k_width;
  return std::exp(-0.5*arg*arg);
}

void Spectral2DFieldGenerator::CountModes() {
  Mesh *pm = pmy_pack->pmesh;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real dkx = 2.0*kPi/lx;
  Real dky = 2.0*kPi/ly;
  int nkx_max = std::max(1, static_cast<int>(std::ceil(k_max/dkx)));
  int nky_max = std::max(1, static_cast<int>(std::ceil(k_max/dky)));

  mode_count = 0;
  for (int nkx = 0; nkx <= nkx_max; ++nkx) {
    for (int nky = 0; nky <= nky_max; ++nky) {
      if (nkx == 0 && nky == 0) continue;
      Real kmag = std::sqrt(SQR(dkx*nkx) + SQR(dky*nky));
      if (kmag >= k_min && kmag <= k_max) mode_count++;
    }
  }
}

void Spectral2DFieldGenerator::GenerateModeCoefficients() {
  Mesh *pm = pmy_pack->pmesh;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real dkx = 2.0*kPi/lx;
  Real dky = 2.0*kPi/ly;
  int nkx_max = std::max(1, static_cast<int>(std::ceil(k_max/dkx)));
  int nky_max = std::max(1, static_cast<int>(std::ceil(k_max/dky)));

  int n = 0;
  for (int nkx = 0; nkx <= nkx_max; ++nkx) {
    for (int nky = 0; nky <= nky_max; ++nky) {
      if (nkx == 0 && nky == 0) continue;
      Real kx = dkx*nkx;
      Real ky = dky*nky;
      Real kmag = std::sqrt(SQR(kx) + SQR(ky));
      if (kmag < k_min || kmag > k_max) continue;

      Real amp = ModeAmplitude(kmag);
      kx_mode.h_view(n) = kx;
      ky_mode.h_view(n) = ky;
      az_cc.h_view(n) = amp*RanGaussianSt(&rstate);
      az_cs.h_view(n) = (nky > 0) ? amp*RanGaussianSt(&rstate) : 0.0;
      az_sc.h_view(n) = (nkx > 0) ? amp*RanGaussianSt(&rstate) : 0.0;
      az_ss.h_view(n) = (nkx > 0 && nky > 0) ? amp*RanGaussianSt(&rstate) : 0.0;
      ++n;
    }
  }

  kx_mode.template modify<HostMemSpace>();
  kx_mode.template sync<DevExeSpace>();
  ky_mode.template modify<HostMemSpace>();
  ky_mode.template sync<DevExeSpace>();
  az_cc.template modify<HostMemSpace>();
  az_cc.template sync<DevExeSpace>();
  az_cs.template modify<HostMemSpace>();
  az_cs.template sync<DevExeSpace>();
  az_sc.template modify<HostMemSpace>();
  az_sc.template sync<DevExeSpace>();
  az_ss.template modify<HostMemSpace>();
  az_ss.template sync<DevExeSpace>();
}

void Spectral2DFieldGenerator::PrecomputeTrigTables() {
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int ie = indcs.ie;
  int js = indcs.js;
  int je = indcs.je;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  auto &size = pmy_pack->pmb->mb_size;

  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;

  par_for("spec2d_xtrig", DevExeSpace(), 0, nmb - 1, 0, mode_count - 1, is, ie,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real x1 = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real kx = kx_mode_.d_view(n);
    xcos_(m,n,i) = cos(kx*x1);
    xsin_(m,n,i) = sin(kx*x1);
  });

  par_for("spec2d_ytrig", DevExeSpace(), 0, nmb - 1, 0, mode_count - 1, js, je,
  KOKKOS_LAMBDA(int m, int n, int j) {
    Real x2 = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real ky = ky_mode_.d_view(n);
    ycos_(m,n,j) = cos(ky*x2);
    ysin_(m,n,j) = sin(ky*x2);
  });
}

void Spectral2DFieldGenerator::GenerateCurlField(DvceArray4D<Real> &vx,
                                                 DvceArray4D<Real> &vy) {
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int ie = indcs.ie;
  int js = indcs.js;
  int je = indcs.je;
  int ks = indcs.ks;
  int ke = indcs.ke;

  auto az_cc_ = az_cc;
  auto az_cs_ = az_cs;
  auto az_sc_ = az_sc;
  auto az_ss_ = az_ss;
  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;

  for (int n = 0; n < mode_count; ++n) {
    par_for("spec2d_curl_field", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real cx = xcos_(m,n,i);
      Real sx = xsin_(m,n,i);
      Real cy = ycos_(m,n,j);
      Real sy = ysin_(m,n,j);
      Real kx = kx_mode_.d_view(n);
      Real ky = ky_mode_.d_view(n);

      vx(m,k,j,i) +=
          -ky*az_cc_.d_view(n)*cx*sy
          + ky*az_cs_.d_view(n)*cx*cy
          - ky*az_sc_.d_view(n)*sx*sy
          + ky*az_ss_.d_view(n)*sx*cy;

      vy(m,k,j,i) +=
           kx*az_cc_.d_view(n)*sx*cy
         + kx*az_cs_.d_view(n)*sx*sy
         - kx*az_sc_.d_view(n)*cx*cy
         - kx*az_ss_.d_view(n)*cx*sy;
    });
  }
}

void RemoveVectorFieldMean(MeshBlockPack *pmbp, DvceArray4D<Real> &vx, DvceArray4D<Real> &vy,
                           DvceArray4D<Real> &vz) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int nkji = nx3*nx2*nx1;
  int nji = nx2*nx1;
  int nmkji = pmbp->nmb_thispack*nkji;

  Real sumx = 0.0;
  Real sumy = 0.0;
  Real sumz = 0.0;
  Kokkos::parallel_reduce("spec2d_mean_x", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = idx - m*nkji - k*nji - j*nx1;
    sum += vx(m,ks + k,js + j,is + i);
  }, Kokkos::Sum<Real>(sumx));
  Kokkos::parallel_reduce("spec2d_mean_y", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = idx - m*nkji - k*nji - j*nx1;
    sum += vy(m,ks + k,js + j,is + i);
  }, Kokkos::Sum<Real>(sumy));
  Kokkos::parallel_reduce("spec2d_mean_z", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = idx - m*nkji - k*nji - j*nx1;
    sum += vz(m,ks + k,js + j,is + i);
  }, Kokkos::Sum<Real>(sumz));

  Real count = static_cast<Real>(nmkji);
#if MPI_PARALLEL_ENABLED
  Real sendbuf[4] = {sumx, sumy, sumz, count};
  Real recvbuf[4];
  MPI_Allreduce(sendbuf, recvbuf, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sumx = recvbuf[0];
  sumy = recvbuf[1];
  sumz = recvbuf[2];
  count = recvbuf[3];
#endif

  if (count <= 0.0) return;
  Real meanx = sumx/count;
  Real meany = sumy/count;
  Real meanz = sumz/count;

  int nmb = pmbp->nmb_thispack;
  int ie = indcs.ie;
  int je = indcs.je;
  int ke = indcs.ke;
  par_for("spec2d_subtract_mean", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    vx(m,k,j,i) -= meanx;
    vy(m,k,j,i) -= meany;
    vz(m,k,j,i) -= meanz;
  });
}

void NormalizeVectorFieldRms(MeshBlockPack *pmbp, DvceArray4D<Real> &vx,
                             DvceArray4D<Real> &vy, DvceArray4D<Real> &vz,
                             Real rms_target) {
  if (rms_target <= 0.0) return;

  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int nkji = nx3*nx2*nx1;
  int nji = nx2*nx1;
  int nmkji = pmbp->nmb_thispack*nkji;

  Real sumsq = 0.0;
  Kokkos::parallel_reduce("spec2d_rms", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = idx - m*nkji - k*nji - j*nx1;
    Real v1 = vx(m,ks + k,js + j,is + i);
    Real v2 = vy(m,ks + k,js + j,is + i);
    Real v3 = vz(m,ks + k,js + j,is + i);
    sum += v1*v1 + v2*v2 + v3*v3;
  }, Kokkos::Sum<Real>(sumsq));

  Real count = static_cast<Real>(nmkji);
#if MPI_PARALLEL_ENABLED
  Real sendbuf[2] = {sumsq, count};
  Real recvbuf[2];
  MPI_Allreduce(sendbuf, recvbuf, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sumsq = recvbuf[0];
  count = recvbuf[1];
#endif

  if (count <= 0.0) return;
  Real rms = std::sqrt(sumsq/count);
  if (rms <= std::numeric_limits<Real>::epsilon()) return;
  Real scale = rms_target/rms;

  int nmb = pmbp->nmb_thispack;
  int ie = indcs.ie;
  int je = indcs.je;
  int ke = indcs.ke;
  par_for("spec2d_scale", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    vx(m,k,j,i) *= scale;
    vy(m,k,j,i) *= scale;
    vz(m,k,j,i) *= scale;
  });
}
