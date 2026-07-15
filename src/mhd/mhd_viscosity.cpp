//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mhd_viscosity.cpp
//! \brief Multidimensional conservative fluxes and sources for relativistic shear.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "mhd/mhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \brief Add the explicit nonlinear shear target and kinematic source.
//!
//! The local loss -pi/(Gamma tau_pi) remains in the implicit IMEX partition.  This
//! routine evaluates pi_NS/(Gamma tau_pi) and v^i A^j+v^j A^i from the same stage
//! primitives and centered spatial gradients.  The acceleration solve eliminates all
//! lab-frame time derivatives with the gamma-law energy equation and the shear closure.

int MHD::AddRelativisticViscousSource(const Real beta_dt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  auto w = w0;
  auto bcc = bcc0;
  auto pi_array = visc_w0;
  auto p_array = visc_u0;
  const auto eos = peos->eos_data;
  const auto viscosity = relativistic_viscosity_data;
  const auto eta_data = resistivity_data;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  int failures = 0;
  Kokkos::parallel_reduce("viscous_nonlinear_source",
  Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0, ks, js, is},
                                         {nmb1+1, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, int &sum_fail) {
    const Real half_idx1 = 0.5/size.d_view(m).dx1;
    const Real half_idx2 = multi_d ? 0.5/size.d_view(m).dx2 : 0.0;
    const Real half_idx3 = three_d ? 0.5/size.d_view(m).dx3 : 0.0;
    const int im = i - 1;
    const int ip = i + 1;
    const int jm = j - 1;
    const int jp = j + 1;
    const int km = k - 1;
    const int kp = k + 1;
    const Real rho = w(m, IDN, k, j, i);
    const Real u1 = w(m, IVX, k, j, i);
    const Real u2 = w(m, IVY, k, j, i);
    const Real u3 = w(m, IVZ, k, j, i);
    const Real internal_energy = w(m, IEN, k, j, i);
    const Real pressure = (eos.gamma - 1.0)*internal_energy;
    const Real enthalpy_density = rho + eos.gamma*internal_energy;
    const Real lor = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
    const Real d = rho*lor;

    srrmhd::ShearStress pi;
    pi.p11 = pi_array(m, srrmhd::IVP11, k, j, i);
    pi.p22 = pi_array(m, srrmhd::IVP22, k, j, i);
    pi.p33 = pi_array(m, srrmhd::IVP33, k, j, i);
    pi.p12 = pi_array(m, srrmhd::IVP12, k, j, i);
    pi.p13 = pi_array(m, srrmhd::IVP13, k, j, i);
    pi.p23 = pi_array(m, srrmhd::IVP23, k, j, i);
    srrmhd::ShearGradient3D gradient;
    gradient.pressure[0] = half_idx1*(eos.gamma - 1.0)
        *(w(m, IEN, k, j, ip) - w(m, IEN, k, j, im));
    for (int n = 0; n < 3; ++n) {
      gradient.du[0][n] = half_idx1*(w(m, IVX+n, k, j, ip)
                                           - w(m, IVX+n, k, j, im));
    }
    gradient.dpi[0].p11 = half_idx1*(pi_array(m, srrmhd::IVP11, k, j, ip)
                                           - pi_array(m, srrmhd::IVP11, k, j, im));
    gradient.dpi[0].p22 = half_idx1*(pi_array(m, srrmhd::IVP22, k, j, ip)
                                           - pi_array(m, srrmhd::IVP22, k, j, im));
    gradient.dpi[0].p33 = half_idx1*(pi_array(m, srrmhd::IVP33, k, j, ip)
                                           - pi_array(m, srrmhd::IVP33, k, j, im));
    gradient.dpi[0].p12 = half_idx1*(pi_array(m, srrmhd::IVP12, k, j, ip)
                                           - pi_array(m, srrmhd::IVP12, k, j, im));
    gradient.dpi[0].p13 = half_idx1*(pi_array(m, srrmhd::IVP13, k, j, ip)
                                           - pi_array(m, srrmhd::IVP13, k, j, im));
    gradient.dpi[0].p23 = half_idx1*(pi_array(m, srrmhd::IVP23, k, j, ip)
                                           - pi_array(m, srrmhd::IVP23, k, j, im));
    if (multi_d) {
      gradient.pressure[1] = half_idx2*(eos.gamma - 1.0)
          *(w(m, IEN, k, jp, i) - w(m, IEN, k, jm, i));
      for (int n = 0; n < 3; ++n) {
        gradient.du[1][n] = half_idx2*(w(m, IVX+n, k, jp, i)
                                             - w(m, IVX+n, k, jm, i));
      }
      gradient.dpi[1].p11 = half_idx2*(pi_array(m, srrmhd::IVP11, k, jp, i)
                                             - pi_array(m, srrmhd::IVP11, k, jm, i));
      gradient.dpi[1].p22 = half_idx2*(pi_array(m, srrmhd::IVP22, k, jp, i)
                                             - pi_array(m, srrmhd::IVP22, k, jm, i));
      gradient.dpi[1].p33 = half_idx2*(pi_array(m, srrmhd::IVP33, k, jp, i)
                                             - pi_array(m, srrmhd::IVP33, k, jm, i));
      gradient.dpi[1].p12 = half_idx2*(pi_array(m, srrmhd::IVP12, k, jp, i)
                                             - pi_array(m, srrmhd::IVP12, k, jm, i));
      gradient.dpi[1].p13 = half_idx2*(pi_array(m, srrmhd::IVP13, k, jp, i)
                                             - pi_array(m, srrmhd::IVP13, k, jm, i));
      gradient.dpi[1].p23 = half_idx2*(pi_array(m, srrmhd::IVP23, k, jp, i)
                                             - pi_array(m, srrmhd::IVP23, k, jm, i));
    }
    if (three_d) {
      gradient.pressure[2] = half_idx3*(eos.gamma - 1.0)
          *(w(m, IEN, kp, j, i) - w(m, IEN, km, j, i));
      for (int n = 0; n < 3; ++n) {
        gradient.du[2][n] = half_idx3*(w(m, IVX+n, kp, j, i)
                                             - w(m, IVX+n, km, j, i));
      }
      gradient.dpi[2].p11 = half_idx3*(pi_array(m, srrmhd::IVP11, kp, j, i)
                                             - pi_array(m, srrmhd::IVP11, km, j, i));
      gradient.dpi[2].p22 = half_idx3*(pi_array(m, srrmhd::IVP22, kp, j, i)
                                             - pi_array(m, srrmhd::IVP22, km, j, i));
      gradient.dpi[2].p33 = half_idx3*(pi_array(m, srrmhd::IVP33, kp, j, i)
                                             - pi_array(m, srrmhd::IVP33, km, j, i));
      gradient.dpi[2].p12 = half_idx3*(pi_array(m, srrmhd::IVP12, kp, j, i)
                                             - pi_array(m, srrmhd::IVP12, km, j, i));
      gradient.dpi[2].p13 = half_idx3*(pi_array(m, srrmhd::IVP13, kp, j, i)
                                             - pi_array(m, srrmhd::IVP13, km, j, i));
      gradient.dpi[2].p23 = half_idx3*(pi_array(m, srrmhd::IVP23, kp, j, i)
                                             - pi_array(m, srrmhd::IVP23, km, j, i));
    }

    const Real e1 = w(m, srrmhd::IRE1, k, j, i);
    const Real e2 = w(m, srrmhd::IRE2, k, j, i);
    const Real e3 = w(m, srrmhd::IRE3, k, j, i);
    const Real b1 = bcc(m, IBX, k, j, i);
    const Real b2 = bcc(m, IBY, k, j, i);
    const Real b3 = bcc(m, IBZ, k, j, i);
    const Real local_eta = srrmhd::EvaluateResistivity(
        eta_data, rho, u1, u2, u3, e1, e2, e3, b1, b2, b3);
    Real charge = half_idx1*(w(m, srrmhd::IRE1, k, j, ip)
                                  - w(m, srrmhd::IRE1, k, j, im));
    if (multi_d) {
      charge += half_idx2*(w(m, srrmhd::IRE2, k, jp, i)
                                - w(m, srrmhd::IRE2, k, jm, i));
    }
    if (three_d) {
      charge += half_idx3*(w(m, srrmhd::IRE3, kp, j, i)
                                - w(m, srrmhd::IRE3, km, j, i));
    }
    srrmhd::FourVector four_force;
    if (!srrmhd::ResistiveMatterFourForce(
            charge, local_eta, u1, u2, u3, e1, e2, e3, b1, b2, b3, four_force)) {
      ++sum_fail;
      return;
    }
    srrmhd::FourVector acceleration;
    srrmhd::ShearStress sigma;
    const Real dynamic_viscosity = enthalpy_density*viscosity.nu;
    if (!srrmhd::SolveShearAcceleration(
            u1, u2, u3, pressure, enthalpy_density, eos.gamma,
            dynamic_viscosity, viscosity.tau, pi, gradient, four_force,
            acceleration, sigma)) {
      ++sum_fail;
      return;
    }
    const srrmhd::ShearStress kinematic = srrmhd::KinematicShearSourcePerD(
        u1, u2, u3, pi, acceleration);
    const Real target_scale = -2.0*dynamic_viscosity/(lor*viscosity.tau);
    srrmhd::ShearStress source = kinematic;
    source.p11 += target_scale*sigma.p11;
    source.p22 += target_scale*sigma.p22;
    source.p33 += target_scale*sigma.p33;
    source.p12 += target_scale*sigma.p12;
    source.p13 += target_scale*sigma.p13;
    source.p23 += target_scale*sigma.p23;
    p_array(m, srrmhd::IVP11, k, j, i) += beta_dt*d*source.p11;
    p_array(m, srrmhd::IVP22, k, j, i) += beta_dt*d*source.p22;
    p_array(m, srrmhd::IVP33, k, j, i) += beta_dt*d*source.p33;
    p_array(m, srrmhd::IVP12, k, j, i) += beta_dt*d*source.p12;
    p_array(m, srrmhd::IVP13, k, j, i) += beta_dt*d*source.p13;
    p_array(m, srrmhd::IVP23, k, j, i) += beta_dt*d*source.p23;
  }, failures);
  return failures;
}

//----------------------------------------------------------------------------------------
//! \brief Add directional LLF transport of P^(ij) and viscous T^(d nu) fluxes.
//!
//! The base resistive-SRMHD LLF solver uses the light speed and omits pi from both its
//! state and physical flux.  Adding 0.5*(F_pi,L+F_pi,R-[U_pi,R-U_pi,L]) therefore gives
//! exactly the missing LLF contribution to total momentum and energy.  The conservative
//! shear variables use advective physical flux v^d P^(ij) with the same causal bound.

void MHD::CalculateViscousFluxes() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int nx1 = indcs.nx1;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nmhd_local = nmhd;
  const auto recon = recon_method;
  const bool extrema = (recon_method == ReconstructionMethod::ppmx);
  auto eos = peos->eos_data;
  auto w = w0;
  auto pi = visc_w0;
  auto flux = uflx.x1f;
  auto pi_flux = visc_flx.x1f;

  // A single periodic 1D block has no boundary exchange after the face-E implicit
  // update.  Rebuild the primitive ghosts from the accepted active state before
  // reconstruction so viscous fluxes do not sample shear from the previous stage.
  if (pmy_pack->pmesh->one_d && pmy_pack->pmesh->nmb_total == 1 &&
      pmy_pack->pmesh->strictly_periodic) {
    par_for(
      "viscous_periodic_primitive_ghosts", DevExeSpace(), 0, nmb1,
      0, nmhd_local-1, 0, ncells1-1,
      KOKKOS_LAMBDA(int m, int n, int i) {
       if (i < is || i > ie) {
        int offset = (i - is) % nx1;
        if (offset < 0) offset += nx1;
        w(m, n, 0, 0, i) = w(m, n, 0, 0, is + offset);
       }
      });
    par_for(
      "viscous_periodic_shear_ghosts", DevExeSpace(), 0, nmb1,
      0, srrmhd::NVISC-1, 0, ncells1-1,
      KOKKOS_LAMBDA(int m, int n, int i) {
       if (i < is || i > ie) {
        int offset = (i - is) % nx1;
        if (offset < 0) offset += nx1;
        pi(m, n, 0, 0, i) = pi(m, n, 0, 0, is + offset);
       }
      });
  }

  const size_t scratch_size =
      2*ScrArray2D<Real>::shmem_size(nmhd_local, ncells1)
      + 2*ScrArray2D<Real>::shmem_size(srrmhd::NVISC, ncells1);
  constexpr int scratch_level = 0;
  par_for_outer("viscous_flux1", DevExeSpace(), scratch_size, scratch_level,
                0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray2D<Real> wl(member.team_scratch(scratch_level), nmhd_local, ncells1);
    ScrArray2D<Real> wr(member.team_scratch(scratch_level), nmhd_local, ncells1);
    ScrArray2D<Real> pil(member.team_scratch(scratch_level), srrmhd::NVISC, ncells1);
    ScrArray2D<Real> pir(member.team_scratch(scratch_level), srrmhd::NVISC, ncells1);

    switch (recon) {
      case ReconstructionMethod::dc:
        DonorCellX1(member, m, k, j, is-1, ie+1, w, wl, wr);
        DonorCellX1(member, m, k, j, is-1, ie+1, pi, pil, pir);
        break;
      case ReconstructionMethod::plm:
        PiecewiseLinearX1(member, m, k, j, is-1, ie+1, w, wl, wr);
        PiecewiseLinearX1(member, m, k, j, is-1, ie+1, pi, pil, pir);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
        PiecewiseParabolicX1(member, eos, extrema, true, m, k, j, is-1, ie+1,
                             w, wl, wr);
        PiecewiseParabolicX1(member, eos, extrema, false, m, k, j, is-1, ie+1,
                             pi, pil, pir);
        break;
      case ReconstructionMethod::wenoz:
        WENOZX1(member, eos, true, m, k, j, is-1, ie+1, w, wl, wr);
        WENOZX1(member, eos, false, m, k, j, is-1, ie+1, pi, pil, pir);
        break;
      default:
        break;
    }
    member.team_barrier();

    par_for_inner(member, is, ie+1, [&](const int i) {
      srrmhd::ShearStress pi_l;
      pi_l.p11 = pil(srrmhd::IVP11, i);
      pi_l.p22 = pil(srrmhd::IVP22, i);
      pi_l.p33 = pil(srrmhd::IVP33, i);
      pi_l.p12 = pil(srrmhd::IVP12, i);
      pi_l.p13 = pil(srrmhd::IVP13, i);
      pi_l.p23 = pil(srrmhd::IVP23, i);
      srrmhd::ShearStress pi_r;
      pi_r.p11 = pir(srrmhd::IVP11, i);
      pi_r.p22 = pir(srrmhd::IVP22, i);
      pi_r.p33 = pir(srrmhd::IVP33, i);
      pi_r.p12 = pir(srrmhd::IVP12, i);
      pi_r.p13 = pir(srrmhd::IVP13, i);
      pi_r.p23 = pir(srrmhd::IVP23, i);

      Real stress_flux[srrmhd::NVISC];
      Real momentum_flux[3];
      Real energy_flux;
      srrmhd::ViscousLLFContributions(
          0, wl(IDN, i), wl(IVX, i), wl(IVY, i), wl(IVZ, i), pi_l,
          wr(IDN, i), wr(IVX, i), wr(IVY, i), wr(IVZ, i), pi_r,
          stress_flux, momentum_flux, energy_flux);
      for (int n = 0; n < srrmhd::NVISC; ++n) {
        pi_flux(m, n, k, j, i) = stress_flux[n];
      }
      flux(m, IM1, k, j, i) += momentum_flux[0];
      flux(m, IM2, k, j, i) += momentum_flux[1];
      flux(m, IM3, k, j, i) += momentum_flux[2];
      flux(m, IEN, k, j, i) += energy_flux;
    });
  });

  if (pmy_pack->pmesh->multi_d) {
    const size_t scratch_size2 =
        3*ScrArray2D<Real>::shmem_size(nmhd_local, ncells1)
        + 3*ScrArray2D<Real>::shmem_size(srrmhd::NVISC, ncells1);
    auto flux2 = uflx.x2f;
    auto pi_flux2 = visc_flx.x2f;
    par_for_outer("viscous_flux2", DevExeSpace(), scratch_size2, scratch_level,
                  0, nmb1, ks, ke,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k) {
      ScrArray2D<Real> w_scr1(member.team_scratch(scratch_level),
                              nmhd_local, ncells1);
      ScrArray2D<Real> w_scr2(member.team_scratch(scratch_level),
                              nmhd_local, ncells1);
      ScrArray2D<Real> w_scr3(member.team_scratch(scratch_level),
                              nmhd_local, ncells1);
      ScrArray2D<Real> pi_scr1(member.team_scratch(scratch_level),
                               srrmhd::NVISC, ncells1);
      ScrArray2D<Real> pi_scr2(member.team_scratch(scratch_level),
                               srrmhd::NVISC, ncells1);
      ScrArray2D<Real> pi_scr3(member.team_scratch(scratch_level),
                               srrmhd::NVISC, ncells1);

      for (int j = js-1; j <= je+1; ++j) {
        auto wl = w_scr1;
        auto wl_jp1 = w_scr2;
        auto wr = w_scr3;
        auto pil = pi_scr1;
        auto pil_jp1 = pi_scr2;
        auto pir = pi_scr3;
        if ((j % 2) == 0) {
          wl = w_scr2;
          wl_jp1 = w_scr1;
          pil = pi_scr2;
          pil_jp1 = pi_scr1;
        }
        switch (recon) {
          case ReconstructionMethod::dc:
            DonorCellX2(member, m, k, j, is, ie, w, wl_jp1, wr);
            DonorCellX2(member, m, k, j, is, ie, pi, pil_jp1, pir);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX2(member, m, k, j, is, ie, w, wl_jp1, wr);
            PiecewiseLinearX2(member, m, k, j, is, ie, pi, pil_jp1, pir);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX2(member, eos, extrema, true,
                                 m, k, j, is, ie, w, wl_jp1, wr);
            PiecewiseParabolicX2(member, eos, extrema, false,
                                 m, k, j, is, ie, pi, pil_jp1, pir);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX2(member, eos, true, m, k, j, is, ie, w, wl_jp1, wr);
            WENOZX2(member, eos, false, m, k, j, is, ie, pi, pil_jp1, pir);
            break;
          default:
            break;
        }
        member.team_barrier();
        if (j > js-1) {
          par_for_inner(member, is, ie, [&](const int i) {
            srrmhd::ShearStress pi_l;
            pi_l.p11 = pil(srrmhd::IVP11, i);
            pi_l.p22 = pil(srrmhd::IVP22, i);
            pi_l.p33 = pil(srrmhd::IVP33, i);
            pi_l.p12 = pil(srrmhd::IVP12, i);
            pi_l.p13 = pil(srrmhd::IVP13, i);
            pi_l.p23 = pil(srrmhd::IVP23, i);
            srrmhd::ShearStress pi_r;
            pi_r.p11 = pir(srrmhd::IVP11, i);
            pi_r.p22 = pir(srrmhd::IVP22, i);
            pi_r.p33 = pir(srrmhd::IVP33, i);
            pi_r.p12 = pir(srrmhd::IVP12, i);
            pi_r.p13 = pir(srrmhd::IVP13, i);
            pi_r.p23 = pir(srrmhd::IVP23, i);
            Real stress_flux[srrmhd::NVISC];
            Real momentum_flux[3];
            Real energy_flux;
            srrmhd::ViscousLLFContributions(
                1, wl(IDN, i), wl(IVX, i), wl(IVY, i), wl(IVZ, i), pi_l,
                wr(IDN, i), wr(IVX, i), wr(IVY, i), wr(IVZ, i), pi_r,
                stress_flux, momentum_flux, energy_flux);
            for (int n = 0; n < srrmhd::NVISC; ++n) {
              pi_flux2(m, n, k, j, i) = stress_flux[n];
            }
            flux2(m, IM1, k, j, i) += momentum_flux[0];
            flux2(m, IM2, k, j, i) += momentum_flux[1];
            flux2(m, IM3, k, j, i) += momentum_flux[2];
            flux2(m, IEN, k, j, i) += energy_flux;
          });
        }
      }
    });
  }

  if (pmy_pack->pmesh->three_d) {
    const size_t scratch_size3 =
        3*ScrArray2D<Real>::shmem_size(nmhd_local, ncells1)
        + 3*ScrArray2D<Real>::shmem_size(srrmhd::NVISC, ncells1);
    auto flux3 = uflx.x3f;
    auto pi_flux3 = visc_flx.x3f;
    par_for_outer("viscous_flux3", DevExeSpace(), scratch_size3, scratch_level,
                  0, nmb1, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray2D<Real> w_scr1(member.team_scratch(scratch_level),
                              nmhd_local, ncells1);
      ScrArray2D<Real> w_scr2(member.team_scratch(scratch_level),
                              nmhd_local, ncells1);
      ScrArray2D<Real> w_scr3(member.team_scratch(scratch_level),
                              nmhd_local, ncells1);
      ScrArray2D<Real> pi_scr1(member.team_scratch(scratch_level),
                               srrmhd::NVISC, ncells1);
      ScrArray2D<Real> pi_scr2(member.team_scratch(scratch_level),
                               srrmhd::NVISC, ncells1);
      ScrArray2D<Real> pi_scr3(member.team_scratch(scratch_level),
                               srrmhd::NVISC, ncells1);

      for (int k = ks-1; k <= ke+1; ++k) {
        auto wl = w_scr1;
        auto wl_kp1 = w_scr2;
        auto wr = w_scr3;
        auto pil = pi_scr1;
        auto pil_kp1 = pi_scr2;
        auto pir = pi_scr3;
        if ((k % 2) == 0) {
          wl = w_scr2;
          wl_kp1 = w_scr1;
          pil = pi_scr2;
          pil_kp1 = pi_scr1;
        }
        switch (recon) {
          case ReconstructionMethod::dc:
            DonorCellX3(member, m, k, j, is, ie, w, wl_kp1, wr);
            DonorCellX3(member, m, k, j, is, ie, pi, pil_kp1, pir);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX3(member, m, k, j, is, ie, w, wl_kp1, wr);
            PiecewiseLinearX3(member, m, k, j, is, ie, pi, pil_kp1, pir);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX3(member, eos, extrema, true,
                                 m, k, j, is, ie, w, wl_kp1, wr);
            PiecewiseParabolicX3(member, eos, extrema, false,
                                 m, k, j, is, ie, pi, pil_kp1, pir);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX3(member, eos, true, m, k, j, is, ie, w, wl_kp1, wr);
            WENOZX3(member, eos, false, m, k, j, is, ie, pi, pil_kp1, pir);
            break;
          default:
            break;
        }
        member.team_barrier();
        if (k > ks-1) {
          par_for_inner(member, is, ie, [&](const int i) {
            srrmhd::ShearStress pi_l;
            pi_l.p11 = pil(srrmhd::IVP11, i);
            pi_l.p22 = pil(srrmhd::IVP22, i);
            pi_l.p33 = pil(srrmhd::IVP33, i);
            pi_l.p12 = pil(srrmhd::IVP12, i);
            pi_l.p13 = pil(srrmhd::IVP13, i);
            pi_l.p23 = pil(srrmhd::IVP23, i);
            srrmhd::ShearStress pi_r;
            pi_r.p11 = pir(srrmhd::IVP11, i);
            pi_r.p22 = pir(srrmhd::IVP22, i);
            pi_r.p33 = pir(srrmhd::IVP33, i);
            pi_r.p12 = pir(srrmhd::IVP12, i);
            pi_r.p13 = pir(srrmhd::IVP13, i);
            pi_r.p23 = pir(srrmhd::IVP23, i);
            Real stress_flux[srrmhd::NVISC];
            Real momentum_flux[3];
            Real energy_flux;
            srrmhd::ViscousLLFContributions(
                2, wl(IDN, i), wl(IVX, i), wl(IVY, i), wl(IVZ, i), pi_l,
                wr(IDN, i), wr(IVX, i), wr(IVY, i), wr(IVZ, i), pi_r,
                stress_flux, momentum_flux, energy_flux);
            for (int n = 0; n < srrmhd::NVISC; ++n) {
              pi_flux3(m, n, k, j, i) = stress_flux[n];
            }
            flux3(m, IM1, k, j, i) += momentum_flux[0];
            flux3(m, IM2, k, j, i) += momentum_flux[1];
            flux3(m, IM3, k, j, i) += momentum_flux[2];
            flux3(m, IEN, k, j, i) += energy_flux;
          });
        }
      }
    });
  }
}

} // namespace mhd
