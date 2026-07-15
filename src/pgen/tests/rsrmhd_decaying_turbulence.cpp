//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_decaying_turbulence.cpp
//! \brief Driven or decaying turbulence for causal viscous and resistive SRMHD.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "mhd/mhd.hpp"
#include "mhd/relativistic_viscosity.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/turb_driver.hpp"
#include "utils/spectral_ic_gen.hpp"

namespace {

//----------------------------------------------------------------------------------------
//! \brief Generate a normalized solenoidal face field from a spectral vector potential.

void GenerateSolenoidalField(MeshBlockPack *pmbp, ParameterInput *pin,
                             const std::string &block, const std::string &label,
                             const Real rms, DvceFaceFld4D<Real> &field) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = indcs.nx3 + 2*indcs.ng;
  const bool three_d = pmbp->pmesh->three_d;
  auto &size = pmbp->pmb->mb_size;

  Kokkos::deep_copy(field.x1f, 0.0);
  Kokkos::deep_copy(field.x2f, 0.0);
  Kokkos::deep_copy(field.x3f, 0.0);
  if (rms == 0.0) return;

  DvceArray4D<Real> ax(label + "_ax", nmb, ncells3, ncells2, ncells1);
  DvceArray4D<Real> ay(label + "_ay", nmb, ncells3, ncells2, ncells1);
  DvceArray4D<Real> az(label + "_az", nmb, ncells3, ncells2, ncells1);
  SpectralICGenerator generator(pmbp, pin, block);
  generator.GenerateVectorPotential(ax, ay, az);

  if (three_d) {
    par_for("pgen_srr_turb_curl_3d", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real idx1 = 1.0/size.d_view(m).dx1;
      const Real idx2 = 1.0/size.d_view(m).dx2;
      const Real idx3 = 1.0/size.d_view(m).dx3;
      field.x1f(m,k,j,i) = (az(m,k,j+1,i) - az(m,k,j,i))*idx2
          - (ay(m,k+1,j,i) - ay(m,k,j,i))*idx3;
      field.x2f(m,k,j,i) = (ax(m,k+1,j,i) - ax(m,k,j,i))*idx3
          - (az(m,k,j,i+1) - az(m,k,j,i))*idx1;
      field.x3f(m,k,j,i) = (ay(m,k,j,i+1) - ay(m,k,j,i))*idx1
          - (ax(m,k,j+1,i) - ax(m,k,j,i))*idx2;
      if (i == ie) {
        field.x1f(m,k,j,i+1) =
            (az(m,k,j+1,i+1) - az(m,k,j,i+1))*idx2
            - (ay(m,k+1,j,i+1) - ay(m,k,j,i+1))*idx3;
      }
      if (j == je) {
        field.x2f(m,k,j+1,i) =
            (ax(m,k+1,j+1,i) - ax(m,k,j+1,i))*idx3
            - (az(m,k,j+1,i+1) - az(m,k,j+1,i))*idx1;
      }
      if (k == ke) {
        field.x3f(m,k+1,j,i) =
            (ay(m,k+1,j,i+1) - ay(m,k+1,j,i))*idx1
            - (ax(m,k+1,j+1,i) - ax(m,k+1,j,i))*idx2;
      }
    });
  } else {
    par_for("pgen_srr_turb_curl_2d", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real idx1 = 1.0/size.d_view(m).dx1;
      const Real idx2 = 1.0/size.d_view(m).dx2;
      field.x1f(m,k,j,i) = (az(m,k,j+1,i) - az(m,k,j,i))*idx2;
      field.x2f(m,k,j,i) = -(az(m,k,j,i+1) - az(m,k,j,i))*idx1;
      field.x3f(m,k,j,i) = 0.0;
      if (i == ie) {
        field.x1f(m,k,j,i+1) =
            (az(m,k,j+1,i+1) - az(m,k,j,i+1))*idx2;
      }
      if (j == je) {
        field.x2f(m,k,j+1,i) =
            -(az(m,k,j+1,i+1) - az(m,k,j+1,i))*idx1;
      }
      if (k == ke) field.x3f(m,k+1,j,i) = 0.0;
    });
  }
  SubtractGlobalMeanB(pmbp, field);
  NormalizeRmsB(pmbp, field, rms);
}

void SRRMHDDecayingTurbulenceHistory(HistoryData *pdata, Mesh *pm);
void SRRMHDDecayingTurbulenceFinal(ParameterInput *pin, Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! \brief Initialize driven or decaying turbulence from configurable fields.

void ProblemGenerator::ResistiveSRMHDDecayingTurbulence(ParameterInput *pin,
                                                         const bool restart) {
  pgen_final_func = SRRMHDDecayingTurbulenceFinal;
  user_hist_func = SRRMHDDecayingTurbulenceHistory;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  if (pmhd == nullptr || !(pmy_mesh_->two_d || pmy_mesh_->three_d)
      || !pmbp->pcoord->is_special_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_decaying_turbulence requires a two- or "
              << "three-dimensional "
              << "special-relativistic MHD mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real rho0 = pin->GetOrAddReal("problem", "density", 1.0);
  const Real pressure0 = pin->GetOrAddReal("problem", "pressure", 1.0);
  const Real velocity_rms = pin->GetOrAddReal("problem", "velocity_rms", 0.15);
  const Real uniform_velocity1 = pin->GetOrAddReal(
      "problem", "uniform_velocity1", 0.0);
  const Real uniform_velocity2 = pin->GetOrAddReal(
      "problem", "uniform_velocity2", 0.0);
  const Real uniform_velocity3 = pin->GetOrAddReal(
      "problem", "uniform_velocity3", 0.0);
  const std::string magnetic_configuration = pin->GetOrAddString(
      "problem", "magnetic_configuration", "random");
  const Real magnetic_rms = pin->GetOrAddReal("problem", "magnetic_rms", 0.15);
  const Real plasma_beta = pin->GetOrAddReal("problem", "plasma_beta", 1.0);
  if (rho0 <= 0.0 || pressure0 <= 0.0 || velocity_rms < 0.0
      || magnetic_rms < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Decaying turbulence requires positive density, "
              << "and pressure, with velocity_rms and magnetic_rms >= 0"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (SQR(uniform_velocity1) + SQR(uniform_velocity2)
      + SQR(uniform_velocity3) >= 0.81) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "The uniform initial velocity must have |v| < 0.9"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmhd->use_electric_ct && (velocity_rms != 0.0
      || uniform_velocity1 != 0.0 || uniform_velocity2 != 0.0
      || uniform_velocity3 != 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_decaying_turbulence currently requires an "
              << "exactly zero initial velocity with face-centered electric CT; "
              << "the driven-box perturbation is supplied by the forcing" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (magnetic_configuration != "random"
      && magnetic_configuration != "uniform_z") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<problem>/magnetic_configuration must be random "
              << "or uniform_z" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (magnetic_configuration == "uniform_z"
      && (!pmy_mesh_->three_d || plasma_beta <= 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "uniform_z requires a three-dimensional mesh and "
              << "<problem>/plasma_beta > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = indcs.nx3 + 2*indcs.ng;
  DvceFaceFld4D<Real> velocity_face("srr_turb_velocity", nmb, ncells3,
                                    ncells2, ncells1);
  GenerateSolenoidalField(pmbp, pin, "velocity_spectral_ic", "srr_turb_v",
                          velocity_rms, velocity_face);
  if (magnetic_configuration == "random") {
    GenerateSolenoidalField(pmbp, pin, "magnetic_spectral_ic", "srr_turb_b",
                            magnetic_rms, pmhd->b0);
  } else {
    Kokkos::deep_copy(pmhd->b0.x1f, 0.0);
    Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
    Kokkos::deep_copy(pmhd->b0.x3f, sqrt(2.0*pressure0/plasma_beta));
  }
  if (pmhd->use_electric_ct) {
    // With the required zero initial velocity, ideal E = -v cross B vanishes.
    Kokkos::deep_copy(pmhd->e0.x1f, 0.0);
    Kokkos::deep_copy(pmhd->e0.x2f, 0.0);
    Kokkos::deep_copy(pmhd->e0.x3f, 0.0);
  }

  auto velocity = velocity_face;
  auto w = pmhd->w0;
  auto u = pmhd->u0;
  auto b = pmhd->b0;
  auto bcc = pmhd->bcc0;
  const bool three_d = pmy_mesh_->three_d;
  const bool resistive = pmhd->is_resistive_rel;
  const Real gm1 = pmhd->peos->eos_data.gamma - 1.0;
  int superluminal = 0;
  Kokkos::parallel_reduce("pgen_srr_turb_prim", Kokkos::MDRangePolicy<DevExeSpace,
      Kokkos::Rank<4>>({0,ks,js,is}, {nmb,ke+1,je+1,ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, int &sum) {
    const Real vx = uniform_velocity1
        + 0.5*(velocity.x1f(m,k,j,i) + velocity.x1f(m,k,j,i+1));
    const Real vy = uniform_velocity2
        + 0.5*(velocity.x2f(m,k,j,i) + velocity.x2f(m,k,j+1,i));
    const Real vz = uniform_velocity3 + (three_d
        ? 0.5*(velocity.x3f(m,k,j,i) + velocity.x3f(m,k+1,j,i)) : 0.0);
    const Real v2 = SQR(vx) + SQR(vy) + SQR(vz);
    if (v2 >= 0.81) ++sum;
    const Real lor = 1.0/sqrt(1.0 - fmin(v2, 0.81));
    const Real bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
    const Real by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
    const Real bz = three_d
        ? 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i)) : 0.0;
    bcc(m,IBX,k,j,i) = bx;
    bcc(m,IBY,k,j,i) = by;
    bcc(m,IBZ,k,j,i) = bz;
    w(m,IDN,k,j,i) = rho0;
    w(m,IVX,k,j,i) = lor*vx;
    w(m,IVY,k,j,i) = lor*vy;
    w(m,IVZ,k,j,i) = lor*vz;
    w(m,IEN,k,j,i) = pressure0/gm1;
    if (resistive) {
      w(m,srrmhd::IRE1,k,j,i) = -(vy*bz - vz*by);
      w(m,srrmhd::IRE2,k,j,i) = -(vz*bx - vx*bz);
      w(m,srrmhd::IRE3,k,j,i) = -(vx*by - vy*bx);
    }
  }, superluminal);
  if (superluminal > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << superluminal << " turbulence cells have |v| >= 0.9; "
              << "reduce <problem>/velocity_rms" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pmhd->peos->PrimToCons(w, bcc, u, is, ie, js, je, ks, ke);
  if (pmhd->relativistic_viscosity_data.enabled) {
    Kokkos::deep_copy(pmhd->visc_u0, 0.0);
    Kokkos::deep_copy(pmhd->visc_w0, 0.0);
  }
}

namespace {

//----------------------------------------------------------------------------------------
//! \brief Volume-integrated decay diagnostics; history machinery performs MPI sums.

void SRRMHDDecayingTurbulenceHistory(HistoryData *pdata, Mesh *pm) {
  const bool driven = (pm->pmb_pack->pturb != nullptr);
  auto *pmhd = pm->pmb_pack->pmhd;
  const bool entropy_cooling =
      (pmhd->psrc->relativistic_cooling_model == RelativisticCoolingModel::entropy);
  const int cooling_offset = driven ? 36 : 17;
  pdata->nhist = entropy_cooling ? cooling_offset + 10 : (driven ? 36 : 10);
  pdata->label[0] = "volume";
  pdata->label[1] = "ekin";
  pdata->label[2] = "emag";
  pdata->label[3] = "eelec";
  pdata->label[4] = "enstrophy";
  pdata->label[5] = "current2";
  pdata->label[6] = "shear2";
  pdata->label[7] = "rho";
  pdata->label[8] = "pgas";
  pdata->label[9] = "divb2";
  if (driven) {
    pdata->label[10] = "f_power";
    pdata->label[11] = "a_rms";
    pdata->label[12] = "net_g1";
    pdata->label[13] = "net_g2";
    pdata->label[14] = "net_g3";
    pdata->label[15] = "mom1";
    pdata->label[16] = "mom2";
    pdata->label[17] = "mom3";
    pdata->label[18] = "etot";
    pdata->label[19] = "e_inj";
    pdata->label[20] = "p_inj1";
    pdata->label[21] = "p_inj2";
    pdata->label[22] = "p_inj3";
    pdata->label[23] = "cycle";
    pdata->label[24] = "mass";
    pdata->label[25] = "eint";
    pdata->label[26] = "v2";
    pdata->label[27] = "mach2";
    pdata->label[28] = "divv2";
    pdata->label[29] = "lorentz";
    pdata->label[30] = "cross_h";
    pdata->label[31] = "sigma";
    pdata->label[32] = "entropy";
    pdata->label[33] = "q_ohm";
    pdata->label[34] = "q_visc";
    pdata->label[35] = "v_alfven";
  } else if (entropy_cooling) {
    pdata->label[10] = "mom1";
    pdata->label[11] = "mom2";
    pdata->label[12] = "mom3";
    pdata->label[13] = "etot";
    pdata->label[14] = "mass";
    pdata->label[15] = "eint";
    pdata->label[16] = "entropy";
  }
  if (entropy_cooling) {
    pdata->label[cooling_offset] = "cool_power";
    pdata->label[cooling_offset+1] = "cool_mom1";
    pdata->label[cooling_offset+2] = "cool_mom2";
    pdata->label[cooling_offset+3] = "cool_mom3";
    pdata->label[cooling_offset+4] = "e_cool";
    pdata->label[cooling_offset+5] = "p_cool1";
    pdata->label[cooling_offset+6] = "p_cool2";
    pdata->label[cooling_offset+7] = "p_cool3";
    pdata->label[cooling_offset+8] = "cool_limit";
    pdata->label[cooling_offset+9] = "e_cool_lim";
  }

  auto w = pmhd->w0;
  auto u = pmhd->u0;
  auto b = pmhd->b0;
  auto bcc = pmhd->bcc0;
  auto visc_w = pmhd->visc_w0;
  auto &indcs = pm->mb_indcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const Real gamma = pmhd->peos->eos_data.gamma;
  const Real resistivity = pmhd->resistivity;
  const Real shear_nu = pmhd->relativistic_viscosity_data.nu;
  const bool three_d = pm->three_d;
  const bool resistive = pmhd->is_resistive_rel;
  const bool viscosity = pmhd->relativistic_viscosity_data.enabled;

  array_sum::GlobalSum total;
  Kokkos::parallel_reduce("srr_turb_history", Kokkos::MDRangePolicy<DevExeSpace,
      Kokkos::Rank<4>>({0,ks,js,is}, {nmb,ke+1,je+1,ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, array_sum::GlobalSum &sum) {
    const Real idx1 = 1.0/size.d_view(m).dx1;
    const Real idx2 = 1.0/size.d_view(m).dx2;
    const Real idx3 = three_d ? 1.0/size.d_view(m).dx3 : 0.0;
    const Real volume = size.d_view(m).dx1*size.d_view(m).dx2
                        *size.d_view(m).dx3;
    const Real u1 = w(m,IVX,k,j,i);
    const Real u2 = w(m,IVY,k,j,i);
    const Real u3 = w(m,IVZ,k,j,i);
    const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
    const Real vx = u1/lor;
    const Real vy = u2/lor;
    const Real vz = u3/lor;
    const Real rho = w(m,IDN,k,j,i);
    const Real eint = w(m,IEN,k,j,i);
    const Real pgas = (gamma - 1.0)*eint;
    const Real d = rho*lor;
    const Real kinetic = (rho + gamma*eint)*SQR(lor) - pgas - d - eint;
    const Real bx = bcc(m,IBX,k,j,i);
    const Real by = bcc(m,IBY,k,j,i);
    const Real bz = bcc(m,IBZ,k,j,i);
    Real ex = -(vy*bz - vz*by);
    Real ey = -(vz*bx - vx*bz);
    Real ez = -(vx*by - vy*bx);
    if (resistive) {
      ex = w(m,srrmhd::IRE1,k,j,i);
      ey = w(m,srrmhd::IRE2,k,j,i);
      ez = w(m,srrmhd::IRE3,k,j,i);
    }
    const Real lor_ip = sqrt(1.0 + SQR(w(m,IVX,k,j,i+1))
        + SQR(w(m,IVY,k,j,i+1)) + SQR(w(m,IVZ,k,j,i+1)));
    const Real lor_im = sqrt(1.0 + SQR(w(m,IVX,k,j,i-1))
        + SQR(w(m,IVY,k,j,i-1)) + SQR(w(m,IVZ,k,j,i-1)));
    const Real lor_jp = sqrt(1.0 + SQR(w(m,IVX,k,j+1,i))
        + SQR(w(m,IVY,k,j+1,i)) + SQR(w(m,IVZ,k,j+1,i)));
    const Real lor_jm = sqrt(1.0 + SQR(w(m,IVX,k,j-1,i))
        + SQR(w(m,IVY,k,j-1,i)) + SQR(w(m,IVZ,k,j-1,i)));
    const Real vx_ip = w(m,IVX,k,j,i+1)/lor_ip;
    const Real vx_im = w(m,IVX,k,j,i-1)/lor_im;
    const Real vy_ip = w(m,IVY,k,j,i+1)/lor_ip;
    const Real vy_im = w(m,IVY,k,j,i-1)/lor_im;
    const Real vz_ip = w(m,IVZ,k,j,i+1)/lor_ip;
    const Real vz_im = w(m,IVZ,k,j,i-1)/lor_im;
    const Real vx_jp = w(m,IVX,k,j+1,i)/lor_jp;
    const Real vx_jm = w(m,IVX,k,j-1,i)/lor_jm;
    const Real vy_jp = w(m,IVY,k,j+1,i)/lor_jp;
    const Real vy_jm = w(m,IVY,k,j-1,i)/lor_jm;
    const Real vz_jp = w(m,IVZ,k,j+1,i)/lor_jp;
    const Real vz_jm = w(m,IVZ,k,j-1,i)/lor_jm;
    Real vx_kp=vx, vx_km=vx, vy_kp=vy, vy_km=vy;
    Real vz_kp=vz, vz_km=vz;
    if (three_d) {
      const Real lor_kp = sqrt(1.0 + SQR(w(m,IVX,k+1,j,i))
          + SQR(w(m,IVY,k+1,j,i)) + SQR(w(m,IVZ,k+1,j,i)));
      const Real lor_km = sqrt(1.0 + SQR(w(m,IVX,k-1,j,i))
          + SQR(w(m,IVY,k-1,j,i)) + SQR(w(m,IVZ,k-1,j,i)));
      vx_kp = w(m,IVX,k+1,j,i)/lor_kp;
      vx_km = w(m,IVX,k-1,j,i)/lor_km;
      vy_kp = w(m,IVY,k+1,j,i)/lor_kp;
      vy_km = w(m,IVY,k-1,j,i)/lor_km;
      vz_kp = w(m,IVZ,k+1,j,i)/lor_kp;
      vz_km = w(m,IVZ,k-1,j,i)/lor_km;
    }
    const Real omega1 = 0.5*idx2*(vz_jp - vz_jm) - 0.5*idx3*(vy_kp - vy_km);
    const Real omega2 = 0.5*idx3*(vx_kp - vx_km) - 0.5*idx1*(vz_ip - vz_im);
    const Real omega3 = 0.5*idx1*(vy_ip - vy_im) - 0.5*idx2*(vx_jp - vx_jm);
    const Real divv = 0.5*idx1*(vx_ip - vx_im) + 0.5*idx2*(vy_jp - vy_jm)
        + 0.5*idx3*(vz_kp - vz_km);
    Real current1 =
        0.5*idx2*(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,k,j-1,i));
    Real current2 =
        -0.5*idx1*(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1));
    if (three_d) {
      current1 -= 0.5*idx3*(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i));
      current2 += 0.5*idx3*(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i));
    }
    const Real current3 = 0.5*idx1*(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1))
        - 0.5*idx2*(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i));
    const Real divb = idx1*(b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))
        + idx2*(b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))
        + idx3*(b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i));
    Real shear2 = 0.0;
    if (viscosity) {
      srrmhd::ShearStress pi;
      pi.p11=visc_w(m,srrmhd::IVP11,k,j,i);
      pi.p22=visc_w(m,srrmhd::IVP22,k,j,i);
      pi.p33=visc_w(m,srrmhd::IVP33,k,j,i);
      pi.p12=visc_w(m,srrmhd::IVP12,k,j,i);
      pi.p13=visc_w(m,srrmhd::IVP13,k,j,i);
      pi.p23=visc_w(m,srrmhd::IVP23,k,j,i);
      shear2 = SQR(srrmhd::ShearInvariantNorm(u1,u2,u3,pi));
    }
    const Real v2 = SQR(vx) + SQR(vy) + SQR(vz);
    const Real enthalpy = rho + gamma*eint;
    const Real cs2 = gamma*pgas/enthalpy;
    const Real bdotv = bx*vx + by*vy + bz*vz;
    const Real bfluid1 = bx - (vy*ez - vz*ey);
    const Real bfluid2 = by - (vz*ex - vx*ez);
    const Real bfluid3 = bz - (vx*ey - vy*ex);
    const Real bcom2 = SQR(lor)*(SQR(bfluid1) + SQR(bfluid2)
                       + SQR(bfluid3) - SQR(bdotv));
    const Real edotv = ex*vx + ey*vy + ez*vz;
    const Real efluid1 = ex + vy*bz - vz*by;
    const Real efluid2 = ey + vz*bx - vx*bz;
    const Real efluid3 = ez + vx*by - vy*bx;
    const Real ecom2 = SQR(lor)*(SQR(efluid1) + SQR(efluid2)
                       + SQR(efluid3) - SQR(edotv));
    const Real q_ohm = (resistive && resistivity > 0.0)
        ? fmax(0.0, ecom2)/resistivity : 0.0;
    const Real eta_shear = enthalpy*shear_nu;
    const Real q_visc = (viscosity && eta_shear > 0.0)
        ? shear2/(2.0*eta_shear) : 0.0;
    const Real entropy = d*log(pgas/pow(rho, gamma));
    array_sum::GlobalSum cell;
    cell.the_array[0]=volume;
    cell.the_array[1]=volume*kinetic;
    cell.the_array[2]=0.5*volume*(SQR(bx)+SQR(by)+SQR(bz));
    cell.the_array[3]=0.5*volume*(SQR(ex)+SQR(ey)+SQR(ez));
    cell.the_array[4]=0.5*volume*(SQR(omega1)+SQR(omega2)+SQR(omega3));
    cell.the_array[5]=volume*(SQR(current1)+SQR(current2)+SQR(current3));
    cell.the_array[6]=volume*shear2;
    cell.the_array[7]=volume*rho;
    cell.the_array[8]=volume*pgas;
    cell.the_array[9]=volume*SQR(divb);
    for (int n=10; n<NHISTORY_VARIABLES; ++n) cell.the_array[n]=0.0;
    if (driven) {
      cell.the_array[15]=volume*u(m,IM1,k,j,i);
      cell.the_array[16]=volume*u(m,IM2,k,j,i);
      cell.the_array[17]=volume*u(m,IM3,k,j,i);
      cell.the_array[18]=volume*u(m,IEN,k,j,i);
      cell.the_array[24]=volume*d;
      cell.the_array[25]=volume*eint;
      cell.the_array[26]=volume*v2;
      cell.the_array[27]=volume*v2/cs2;
      cell.the_array[28]=volume*SQR(divv);
      cell.the_array[29]=volume*lor;
      cell.the_array[30]=volume*bdotv;
      cell.the_array[31]=volume*fmax(0.0, bcom2)/enthalpy;
      cell.the_array[32]=volume*entropy;
      cell.the_array[33]=volume*q_ohm;
      cell.the_array[34]=volume*q_visc;
      cell.the_array[35]=volume*sqrt(fmax(0.0, bcom2)/(enthalpy + bcom2));
    } else if (entropy_cooling) {
      cell.the_array[10]=volume*u(m,IM1,k,j,i);
      cell.the_array[11]=volume*u(m,IM2,k,j,i);
      cell.the_array[12]=volume*u(m,IM3,k,j,i);
      cell.the_array[13]=volume*u(m,IEN,k,j,i);
      cell.the_array[14]=volume*d;
      cell.the_array[15]=volume*eint;
      cell.the_array[16]=volume*entropy;
    }
    sum += cell;
  }, Kokkos::Sum<array_sum::GlobalSum>(total));
  Kokkos::fence();
  if (driven && global_variable::my_rank == 0) {
    auto *pturb = pm->pmb_pack->pturb;
    total.the_array[10] = pturb->last_power;
    total.the_array[11] = pturb->last_accel_rms;
    total.the_array[12] = pturb->last_net_force1;
    total.the_array[13] = pturb->last_net_force2;
    total.the_array[14] = pturb->last_net_force3;
    total.the_array[19] = pturb->injected_energy;
    total.the_array[20] = pturb->injected_momentum1;
    total.the_array[21] = pturb->injected_momentum2;
    total.the_array[22] = pturb->injected_momentum3;
    total.the_array[23] = static_cast<Real>(pm->ncycle);
  }
  if (entropy_cooling && global_variable::my_rank == 0) {
    auto *psrc = pmhd->psrc;
    total.the_array[cooling_offset] = psrc->last_cooling_power;
    total.the_array[cooling_offset+1] = psrc->last_cooling_momentum1;
    total.the_array[cooling_offset+2] = psrc->last_cooling_momentum2;
    total.the_array[cooling_offset+3] = psrc->last_cooling_momentum3;
    total.the_array[cooling_offset+4] = psrc->cooled_energy;
    total.the_array[cooling_offset+5] = psrc->cooled_momentum1;
    total.the_array[cooling_offset+6] = psrc->cooled_momentum2;
    total.the_array[cooling_offset+7] = psrc->cooled_momentum3;
    total.the_array[cooling_offset+8] = psrc->last_limited_cooling_power;
    total.the_array[cooling_offset+9] = psrc->limited_cooling_energy;
  }
  for (int n=0; n<NHISTORY_VARIABLES; ++n) pdata->hdata[n]=total.the_array[n];
}

//----------------------------------------------------------------------------------------
//! \brief Write a final profile and global safety diagnostics.

void SRRMHDDecayingTurbulenceFinal(ParameterInput *pin, Mesh *pm) {
  auto *pmhd = pm->pmb_pack->pmhd;
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto bcc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->bcc0);
  auto visc_w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_w0);
  auto size = pm->pmb_pack->pmb->mb_size.h_view;
  auto &indcs = pm->mb_indcs;
  const bool three_d = pm->three_d;
  const bool viscosity = pmhd->relativistic_viscosity_data.enabled;
  std::string base_name = pin->GetOrAddString(
      "problem", "profile_name", "rsrmhd_decaying_turbulence");
  std::string name = base_name;
  if (global_variable::nranks > 1) {
    name += "-rank" + std::to_string(global_variable::my_rank);
  }
  std::ofstream file(name + "-profile.dat");
  if (three_d) {
    file << "# x y z vx vy vz bx by bz omega_z current_z pi_norm rho eint\n";
  } else {
    file << "# x y vx vy bx by bz omega_z current_z pi_norm rho eint\n";
  }
  file << std::setprecision(17);
  Real min_rho = std::numeric_limits<Real>::max();
  Real min_eint = std::numeric_limits<Real>::max();
  Real max_lorentz = 0.0;
  for (int m=0; m<pm->pmb_pack->nmb_thispack; ++m) {
    const Real idx1=1.0/size(m).dx1;
    const Real idx2=1.0/size(m).dx2;
    const int kend = three_d ? indcs.ke : indcs.ks;
    for (int k=indcs.ks; k<=kend; ++k) {
      for (int j=indcs.js; j<=indcs.je; ++j) {
        for (int i=indcs.is; i<=indcs.ie; ++i) {
          auto velocity = [&](int kk, int jj, int ii, int component) {
            const Real u1=w(m,IVX,kk,jj,ii), u2=w(m,IVY,kk,jj,ii);
            const Real u3=w(m,IVZ,kk,jj,ii);
            return w(m,component,kk,jj,ii)
                /sqrt(1.0+SQR(u1)+SQR(u2)+SQR(u3));
          };
          const Real vx=velocity(k,j,i,IVX);
          const Real vy=velocity(k,j,i,IVY);
          const Real vz=velocity(k,j,i,IVZ);
          const Real u1=w(m,IVX,k,j,i), u2=w(m,IVY,k,j,i);
          const Real u3=w(m,IVZ,k,j,i);
          const Real lorentz=sqrt(1.0+SQR(u1)+SQR(u2)+SQR(u3));
          const Real omega=0.5*idx1*(velocity(k,j,i+1,IVY)
              - velocity(k,j,i-1,IVY))
              - 0.5*idx2*(velocity(k,j+1,i,IVX)
              - velocity(k,j-1,i,IVX));
          const Real current=0.5*idx1*(bcc(m,IBY,k,j,i+1)
              - bcc(m,IBY,k,j,i-1))
              - 0.5*idx2*(bcc(m,IBX,k,j+1,i)
              - bcc(m,IBX,k,j-1,i));
          Real pi_norm=0.0;
          if (viscosity) {
            srrmhd::ShearStress pi;
            pi.p11=visc_w(m,srrmhd::IVP11,k,j,i);
            pi.p22=visc_w(m,srrmhd::IVP22,k,j,i);
            pi.p33=visc_w(m,srrmhd::IVP33,k,j,i);
            pi.p12=visc_w(m,srrmhd::IVP12,k,j,i);
            pi.p13=visc_w(m,srrmhd::IVP13,k,j,i);
            pi.p23=visc_w(m,srrmhd::IVP23,k,j,i);
            pi_norm=srrmhd::ShearInvariantNorm(u1,u2,u3,pi);
          }
          const Real x=size(m).x1min+(i-indcs.is+0.5)*size(m).dx1;
          const Real y=size(m).x2min+(j-indcs.js+0.5)*size(m).dx2;
          const Real z=size(m).x3min+(k-indcs.ks+0.5)*size(m).dx3;
          min_rho=std::min(min_rho,w(m,IDN,k,j,i));
          min_eint=std::min(min_eint,w(m,IEN,k,j,i));
          max_lorentz=std::max(max_lorentz,lorentz);
          if (three_d) {
            file << x << " " << y << " " << z << " "
                 << vx << " " << vy << " " << vz << " "
                 << bcc(m,IBX,k,j,i) << " " << bcc(m,IBY,k,j,i) << " "
                 << bcc(m,IBZ,k,j,i) << " " << omega << " " << current << " "
                 << pi_norm << " " << w(m,IDN,k,j,i) << " "
                 << w(m,IEN,k,j,i) << "\n";
          } else {
            file << x << " " << y << " " << vx << " " << vy << " "
                 << bcc(m,IBX,k,j,i) << " " << bcc(m,IBY,k,j,i) << " "
                 << bcc(m,IBZ,k,j,i) << " " << omega << " " << current << " "
                 << pi_norm << " " << w(m,IDN,k,j,i) << " "
                 << w(m,IEN,k,j,i) << "\n";
          }
        }
      }
    }
  }
  file.close();

  // Optional cell-wise state dump used by continuous-versus-restart regressions.
  // Store conserved fluid/E variables, conservative shear, one representative value
  // per face family, and both summed and independent OU forcing states.
  if (pin->DoesParameterExist("problem", "restart_state_name")) {
    const std::string state_base = pin->GetString("problem", "restart_state_name");
    if (state_base.compare("none") != 0 && pm->pmb_pack->pturb != nullptr) {
      auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->u0);
      auto visc_u = Kokkos::create_mirror_view_and_copy(
          HostMemSpace(), pmhd->visc_u0);
      auto b1f = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x1f);
      auto b2f = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x2f);
      auto b3f = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x3f);
      auto e1f = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e0.x1f);
      auto e2f = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e0.x2f);
      auto e3f = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e0.x3f);
      auto *pturb = pm->pmb_pack->pturb;
      auto force = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pturb->force);
      auto force_component = Kokkos::create_mirror_view_and_copy(
          HostMemSpace(), pturb->force_component);
      std::string state_name = state_base;
      if (global_variable::nranks > 1) {
        state_name += "-rank" + std::to_string(global_variable::my_rank);
      }
      std::ofstream state_file(state_name + ".dat");
      state_file << std::setprecision(17);
      for (int m=0; m<pm->pmb_pack->nmb_thispack; ++m) {
        const int kend = three_d ? indcs.ke : indcs.ks;
        for (int k=indcs.ks; k<=kend; ++k) {
          for (int j=indcs.js; j<=indcs.je; ++j) {
            for (int i=indcs.is; i<=indcs.ie; ++i) {
              const Real x=size(m).x1min+(i-indcs.is+0.5)*size(m).dx1;
              const Real y=size(m).x2min+(j-indcs.js+0.5)*size(m).dx2;
              const Real z=size(m).x3min+(k-indcs.ks+0.5)*size(m).dx3;
              state_file << x << " " << y << " " << z;
              for (int n=0; n<pmhd->nmhd; ++n) {
                state_file << " " << u(m,n,k,j,i);
              }
              for (int n=0; n<srrmhd::NVISC; ++n) {
                state_file << " " << visc_u(m,n,k,j,i);
              }
              state_file << " " << b1f(m,k,j,i)
                         << " " << b2f(m,k,j,i)
                         << " " << b3f(m,k,j,i);
              if (pmhd->use_electric_ct) {
                state_file << " " << e1f(m,k,j,i)
                           << " " << e2f(m,k,j,i)
                           << " " << e3f(m,k,j,i);
              } else {
                state_file << " " << u(m,srrmhd::IRE1,k,j,i)
                           << " " << u(m,srrmhd::IRE2,k,j,i)
                           << " " << u(m,srrmhd::IRE3,k,j,i);
              }
              for (int n=0; n<3; ++n) {
                state_file << " " << force(m,n,k,j,i);
              }
              for (int c=0; c<pturb->num_components; ++c) {
                for (int n=0; n<3; ++n) {
                  state_file << " " << force_component(c,m,n,k,j,i);
                }
              }
              state_file << "\n";
            }
          }
        }
      }
    }
  }

  if (pm->pmb_pack->pturb != nullptr) {
    Real local_bounds[3] = {min_rho, min_eint, max_lorentz};
    Real global_bounds[3] = {min_rho, min_eint, max_lorentz};
    int local_counters[5] = {pm->ecounter.nfofc, pm->ecounter.neos_dfloor,
                             pm->ecounter.neos_efloor, pm->ecounter.neos_vceil,
                             pm->ecounter.neos_fail};
    int global_counters[5] = {local_counters[0], local_counters[1],
                              local_counters[2], local_counters[3],
                              local_counters[4]};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&local_bounds[0], &global_bounds[0], 1, MPI_ATHENA_REAL,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_bounds[1], &global_bounds[1], 1, MPI_ATHENA_REAL,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_bounds[2], &global_bounds[2], 1, MPI_ATHENA_REAL,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_counters, global_counters, 5, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::ofstream forcing_file(base_name + "-forcing.dat");
      forcing_file << std::setprecision(17)
                   << global_bounds[0] << " " << global_bounds[1] << " "
                   << global_bounds[2];
      for (int n=0; n<5; ++n) forcing_file << " " << global_counters[n];
      forcing_file << " " << pm->time << " " << pm->ncycle << std::endl;
    }
  }
}

} // namespace
