//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_tasks.cpp
//! \brief functions that control MHD tasks stored in tasklists in MeshBlockPack

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/biermann.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box/shearing_box.hpp"
#include "mhd/dual_ct.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::AssembleMHDTasks
//! \brief Adds mhd tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after MHD constructor
//! See comments Hydro::AssembleHydroTasks() function for more details.

void MHD::AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // assemble "before_timeintegrator" task list
  id.savest = tl["before_timeintegrator"]->AddTask(&MHD::SaveMHDState, this, none);

  // assemble "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&MHD::InitRecv, this, none);

  // assemble "stagen" task list
  if (is_resistive_rel) {
    id.copyu = tl["stagen"]->AddTask(&MHD::FirstTwoImpRK, this, none);
  } else {
    id.copyu = tl["stagen"]->AddTask(&MHD::CopyCons, this, none);
  }
  id.flux      = tl["stagen"]->AddTask(&MHD::Fluxes, this, id.copyu);
  id.sendf     = tl["stagen"]->AddTask(&MHD::SendFlux, this, id.flux);
  id.recvf     = tl["stagen"]->AddTask(&MHD::RecvFlux, this, id.sendf);
  id.rkupdt    = tl["stagen"]->AddTask(&MHD::RKUpdate, this, id.recvf);
  id.ectprep   = tl["stagen"]->AddTask(&MHD::DualCTPrepare, this, id.rkupdt);
  id.sendbedge = tl["stagen"]->AddTask(&MHD::SendBEdge, this, id.ectprep);
  id.recvbedge = tl["stagen"]->AddTask(&MHD::RecvBEdge, this, id.sendbedge);
  id.ect       = tl["stagen"]->AddTask(&MHD::DualCTUpdate, this, id.recvbedge);
  id.srctrms   = tl["stagen"]->AddTask(&MHD::MHDSrcTerms, this, id.ect);
  id.sendu_oa  = tl["stagen"]->AddTask(&MHD::SendU_OA, this, id.srctrms);
  id.recvu_oa  = tl["stagen"]->AddTask(&MHD::RecvU_OA, this, id.sendu_oa);
  id.restu     = tl["stagen"]->AddTask(&MHD::RestrictU, this, id.recvu_oa);
  id.sendu     = tl["stagen"]->AddTask(&MHD::SendU, this, id.restu);
  id.recvu     = tl["stagen"]->AddTask(&MHD::RecvU, this, id.sendu);
  id.sendu_shr = tl["stagen"]->AddTask(&MHD::SendU_Shr, this, id.recvu);
  id.recvu_shr = tl["stagen"]->AddTask(&MHD::RecvU_Shr, this, id.sendu_shr);
  id.efld      = tl["stagen"]->AddTask(&MHD::CornerE, this, id.recvu_shr);
  id.efldsrc   = tl["stagen"]->AddTask(&MHD::EFieldSrc, this, id.efld);
  id.sende     = tl["stagen"]->AddTask(&MHD::SendE, this, id.efldsrc);
  id.recve     = tl["stagen"]->AddTask(&MHD::RecvE, this, id.sende);
  id.ct        = tl["stagen"]->AddTask(&MHD::CT, this, id.recve);
  id.sendb_oa  = tl["stagen"]->AddTask(&MHD::SendB_OA, this, id.ct);
  id.recvb_oa  = tl["stagen"]->AddTask(&MHD::RecvB_OA, this, id.sendb_oa);
  id.restb     = tl["stagen"]->AddTask(&MHD::RestrictB, this, id.recvb_oa);
  id.sendb     = tl["stagen"]->AddTask(&MHD::SendB, this, id.restb);
  id.recvb     = tl["stagen"]->AddTask(&MHD::RecvB, this, id.sendb);
  id.sendb_shr = tl["stagen"]->AddTask(&MHD::SendB_Shr, this, id.recvb);
  id.recvb_shr = tl["stagen"]->AddTask(&MHD::RecvB_Shr, this, id.sendb_shr);
  id.bcs       = tl["stagen"]->AddTask(&MHD::ApplyPhysicalBCs, this, id.recvb_shr);
  id.prol      = tl["stagen"]->AddTask(&MHD::Prolongate, this, id.bcs);
  if (is_resistive_rel) {
    if (use_electric_ct) {
      id.impl = tl["stagen"]->AddTask(&MHD::FaceImpRKUpdate, this, id.prol);
      // The face implicit solve updates only active cells.  Refresh physical ghosts
      // before the next explicit stage; ConToPrim then rebuilds matching ghost
      // primitives.  Internal/MPI ghosts are synchronized inside FaceImpRKUpdate.
      id.impl_bcs = tl["stagen"]->AddTask(&MHD::ApplyPhysicalBCs, this, id.impl);
      id.c2p = tl["stagen"]->AddTask(&MHD::ConToPrim, this, id.impl_bcs);
    } else {
      id.impl = tl["stagen"]->AddTask(&MHD::ImpRKUpdate, this, id.prol);
      id.c2p = tl["stagen"]->AddTask(&MHD::ConToPrim, this, id.impl);
    }
  } else {
    id.c2p = tl["stagen"]->AddTask(&MHD::ConToPrim, this, id.prol);
  }
  id.newdt     = tl["stagen"]->AddTask(&MHD::NewTimeStep, this, id.c2p);

  // assemble "after_stagen" task list
  id.csend = tl["after_stagen"]->AddTask(&MHD::ClearSend, this, none);
  // although RecvFlux/U/E/B functions check that all recvs complete, add ClearRecv to
  // task list anyways to catch potential bugs in MPI communication logic
  id.crecv = tl["after_stagen"]->AddTask(&MHD::ClearRecv, this, id.csend);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SaveMHDState
//! \brief Copy primitives and bcc before step to enable computation of time derivatives,
//! for example to compute jcon in GRMHD.

TaskStatus MHD::SaveMHDState(Driver *pdrive, int stage) {
  if (wbcc_saved) {
    Kokkos::deep_copy(DevExeSpace(), wsaved, w0);
    Kokkos::deep_copy(DevExeSpace(), bccsaved, bcc0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI), and
//! initialize all boundary receive status flags to waiting (with or without MPI).  Note
//! this must be done for communication of BOTH conserved (cell-centered) and
//! face-centered fields AND their fluxes (with SMR/AMR).

TaskStatus MHD::InitRecv(Driver *pdrive, int stage) {
  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nmhd+nscalars);
  if (tstat != TaskStatus::complete) return tstat;
  if (relativistic_viscosity_data.enabled) {
    tstat = pbval_visc->InitRecv(srrmhd::NVISC);
    if (tstat != TaskStatus::complete) return tstat;
  }
  // post receives for B
  tstat = pbval_b->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U, always post receives for fluxes of B
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR, post receives for fluxes of U
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->InitFluxRecv(nmhd+nscalars);
      if (tstat != TaskStatus::complete) return tstat;
    }
    // post receives for fluxes of B, which are used even with uniform grids
    tstat = pbval_b->InitFluxRecv(3);
    if (tstat != TaskStatus::complete) return tstat;
    if (use_electric_ct) {
      tstat = pbval_e->InitFluxRecv(3);
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with orbital advection post receives for U and B
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_u->InitRecv();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = porb_b->InitRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with shearing box boundaries caluclate x2-distance x1-boundaries have sheared and
  // with MPI post receives for U and B
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    Real qom = (psrc->qshear)*(psrc->omega0);
    Real time = pmy_pack->pmesh->time;
    if (stage == pdrive->nexp_stages) {
      time += pmy_pack->pmesh->dt;
    }
    tstat = psbox_u->InitRecv(qom, time);
    if (tstat != TaskStatus::complete) return tstat;
    tstat = psbox_b->InitRecv(qom, time);
    if (tstat != TaskStatus::complete) return tstat;
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::CopyCons
//! \brief Simple task list function that copies u0 --> u1, and b0 --> b1 in first stage

TaskStatus MHD::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    if (relativistic_viscosity_data.enabled) {
      Kokkos::deep_copy(DevExeSpace(), visc_u1, visc_u0);
    }
    Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
    if (use_electric_ct) {
      Kokkos::deep_copy(DevExeSpace(), e1.x1f, e0.x1f);
      Kokkos::deep_copy(DevExeSpace(), e1.x2f, e0.x2f);
      Kokkos::deep_copy(DevExeSpace(), e1.x3f, e0.x3f);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::FirstTwoImpRK()
//! \brief Execute the two leading fully implicit stages of IMEX2 before fluxes.
//!
//! Precondition: u0, w0, b0, and bcc0 are mutually consistent at the beginning of the
//! full step.  Postcondition: u1/b1 contain that beginning state, while u0/w0 contain
//! the second leading implicit state and are ready for reconstruction.  B is unchanged.

TaskStatus MHD::FirstTwoImpRK(Driver *pdrive, int stage) {
  if (stage != 1) return TaskStatus::complete;

  if (use_electric_ct) {
    if (ect_leading_stage == -2) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
      if (relativistic_viscosity_data.enabled) {
        Kokkos::deep_copy(DevExeSpace(), visc_u1, visc_u0);
      }
      Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
      Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
      Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
      Kokkos::deep_copy(DevExeSpace(), e1.x1f, e0.x1f);
      Kokkos::deep_copy(DevExeSpace(), e1.x2f, e0.x2f);
      Kokkos::deep_copy(DevExeSpace(), e1.x3f, e0.x3f);
      ect_leading_stage = -1;
    }
    TaskStatus status = FaceImpRKUpdate(pdrive, ect_leading_stage);
    if (status != TaskStatus::complete) return status;
    if (ect_leading_stage == -1) {
      ect_leading_stage = 0;
      return TaskStatus::incomplete;
    }
    ect_leading_stage = -2;
    // No ordinary boundary/C2P tasks occur between these leading implicit stages and
    // the first flux evaluation.  Make physical ghost conserved and primitive states
    // consistent with the newly recovered active cells now.
    status = ApplyPhysicalBCs(pdrive, stage);
    if (status != TaskStatus::complete) return status;
    return ConToPrim(pdrive, stage);
  }

  Kokkos::deep_copy(DevExeSpace(), u1, u0);
  if (relativistic_viscosity_data.enabled) {
    Kokkos::deep_copy(DevExeSpace(), visc_u1, visc_u0);
  }
  Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
  Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
  Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
  TaskStatus status = ImpRKUpdate(pdrive, -1);
  if (status != TaskStatus::complete) return status;
  return ImpRKUpdate(pdrive, 0);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ImpRKUpdate()
//! \brief Apply one local conductive IMEX stage to resistive SRMHD.
//!
//! For ordinary explicit stages this task runs after CT, B communication, physical
//! boundaries, and prolongation.  Thus u0 contains the partially explicit total
//! conserved state and b0 contains the matching new face field.  Previous stiff-source
//! history is first added to E.  If the tableau has a diagonal solve at this stage, the
//! coupled local recovery updates E and w0 while leaving D, total S, total tau, and B
//! unchanged.  The new conductive source is stored for later tableau rows.

TaskStatus MHD::ImpRKUpdate(Driver *pdriver, int estage) {
  const int istage = estage + 2;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const Real dt = pmy_pack->pmesh->dt;
  auto u = u0;
  auto w = w0;
  auto b = b0;
  auto bcc = bcc0;
  auto visc_u = visc_u0;
  auto visc_w = visc_w0;
  auto ru = pdriver->impl_src;
  auto eta = eta_cell;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const auto eta_data = resistivity_data;
  const Real uniform_eta = resistivity;
  const auto viscosity_data = relativistic_viscosity_data;

  if (istage > 1) {
    auto &a_twid = pdriver->a_twid;
    par_for("srrmhd_imex_history", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1,
            0, n1-1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      for (int s = 0; s <= istage-2; ++s) {
        const Real adt = a_twid[istage-2][s]*dt;
        u(m, srrmhd::IRE1, k, j, i) += adt*ru(s, m, 0, k, j, i);
        u(m, srrmhd::IRE2, k, j, i) += adt*ru(s, m, 1, k, j, i);
        u(m, srrmhd::IRE3, k, j, i) += adt*ru(s, m, 2, k, j, i);
        if (viscosity_data.enabled) {
          for (int n = 0; n < srrmhd::NVISC; ++n) {
            visc_u(m, n, k, j, i) += adt*ru(s, m, 3+n, k, j, i);
          }
        }
      }
    });
  }

  if (estage < pdriver->nexp_stages) {
    const Real a_dt = pdriver->a_impl*dt;
    const bool nonuniform_eta =
        eta_data.model != srrmhd::ResistivityModel::uniform;
    if (nonuniform_eta) {
      par_for("srrmhd_freeze_eta", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1,
              0, n1-1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real bx = 0.5*(b.x1f(m, k, j, i) + b.x1f(m, k, j, i+1));
        const Real by = 0.5*(b.x2f(m, k, j, i) + b.x2f(m, k, j+1, i));
        const Real bz = 0.5*(b.x3f(m, k, j, i) + b.x3f(m, k+1, j, i));
        eta(m, k, j, i) = srrmhd::EvaluateResistivity(
            eta_data, w(m, IDN, k, j, i), w(m, IVX, k, j, i),
            w(m, IVY, k, j, i), w(m, IVZ, k, j, i),
            u(m, srrmhd::IRE1, k, j, i), u(m, srrmhd::IRE2, k, j, i),
            u(m, srrmhd::IRE3, k, j, i), bx, by, bz);
      });
    }
    auto eos = peos->eos_data;
    int failures = 0;
    int max_iterations = 0;
    Kokkos::parallel_reduce("srrmhd_imex_solve",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0, 0, 0, 0},
                                           {nmb1+1, n3, n2, n1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, int &sum_fail, int &max_iter) {
      srrmhd::SRRMHDCons1D state;
      state.d = u(m, IDN, k, j, i);
      state.mx = u(m, IM1, k, j, i);
      state.my = u(m, IM2, k, j, i);
      state.mz = u(m, IM3, k, j, i);
      state.e = u(m, IEN, k, j, i);
      const Real ex_star = u(m, srrmhd::IRE1, k, j, i);
      const Real ey_star = u(m, srrmhd::IRE2, k, j, i);
      const Real ez_star = u(m, srrmhd::IRE3, k, j, i);
      state.ex = ex_star;
      state.ey = ey_star;
      state.ez = ez_star;
      state.bx = 0.5*(b.x1f(m, k, j, i) + b.x1f(m, k, j, i+1));
      state.by = 0.5*(b.x2f(m, k, j, i) + b.x2f(m, k, j+1, i));
      state.bz = 0.5*(b.x3f(m, k, j, i) + b.x3f(m, k+1, j, i));

      srrmhd::SRRMHDPrim1D guess;
      guess.d = w(m, IDN, k, j, i);
      guess.vx = w(m, IVX, k, j, i);
      guess.vy = w(m, IVY, k, j, i);
      guess.vz = w(m, IVZ, k, j, i);
      guess.e = w(m, IEN, k, j, i);
      guess.ex = w(m, srrmhd::IRE1, k, j, i);
      guess.ey = w(m, srrmhd::IRE2, k, j, i);
      guess.ez = w(m, srrmhd::IRE3, k, j, i);
      guess.bx = state.bx;
      guess.by = state.by;
      guess.bz = state.bz;

      const Real local_eta = nonuniform_eta ? eta(m, k, j, i) : uniform_eta;
      const Real kappa = a_dt/local_eta;
      srrmhd::SRRMHDPrim1D recovered = guess;
      srrmhd::ShearStress recovered_pi;
      int iterations = 0;
      bool success;
      srrmhd::ShearStress pi_star;
      if (viscosity_data.enabled) {
        pi_star.p11 = visc_u(m, srrmhd::IVP11, k, j, i)/state.d;
        pi_star.p22 = visc_u(m, srrmhd::IVP22, k, j, i)/state.d;
        pi_star.p33 = visc_u(m, srrmhd::IVP33, k, j, i)/state.d;
        pi_star.p12 = visc_u(m, srrmhd::IVP12, k, j, i)/state.d;
        pi_star.p13 = visc_u(m, srrmhd::IVP13, k, j, i)/state.d;
        pi_star.p23 = visc_u(m, srrmhd::IVP23, k, j, i)/state.d;
        srrmhd::ShearStress pi_ns;
        if (viscosity_data.linearized_target_1d) {
          const int im = (i > 0) ? i - 1 : i;
          const int ip = (i < n1 - 1) ? i + 1 : i;
          const Real inv_length = (ip > im)
              ? 1.0/(static_cast<Real>(ip - im)*mbsize.d_view(m).dx1) : 0.0;
          const Real du1_dx = (w(m, IVX, k, j, ip) - w(m, IVX, k, j, im))
                              *inv_length;
          const Real du2_dx = (w(m, IVY, k, j, ip) - w(m, IVY, k, j, im))
                              *inv_length;
          const Real du3_dx = (w(m, IVZ, k, j, ip) - w(m, IVZ, k, j, im))
                              *inv_length;
          const Real wgas = w(m, IDN, k, j, i) + eos.gamma*w(m, IEN, k, j, i);
          const Real eta_shear = wgas*viscosity_data.nu;
          pi_ns.p11 = -(4.0/3.0)*eta_shear*du1_dx;
          pi_ns.p22 = (2.0/3.0)*eta_shear*du1_dx;
          pi_ns.p33 = (2.0/3.0)*eta_shear*du1_dx;
          pi_ns.p12 = -eta_shear*du2_dx;
          pi_ns.p13 = -eta_shear*du3_dx;
          pi_ns = srrmhd::ProjectShearStress(
              guess.vx, guess.vy, guess.vz, pi_ns);
        }
        success = srrmhd::SingleC2P_IdealSRRMHDImplicitViscous(
            state, eos, ex_star, ey_star, ez_star, kappa, pi_star, pi_ns,
            a_dt/viscosity_data.tau, viscosity_data.chi_max, false, guess,
            recovered, recovered_pi, iterations);
      } else {
        success = srrmhd::SingleC2P_IdealSRRMHDImplicit(
            state, eos, ex_star, ey_star, ez_star, kappa, guess, recovered,
            iterations);
      }
      if (!success) {
        ++sum_fail;
      } else {
        u(m, srrmhd::IRE1, k, j, i) = recovered.ex;
        u(m, srrmhd::IRE2, k, j, i) = recovered.ey;
        u(m, srrmhd::IRE3, k, j, i) = recovered.ez;
        w(m, IDN, k, j, i) = recovered.d;
        w(m, IVX, k, j, i) = recovered.vx;
        w(m, IVY, k, j, i) = recovered.vy;
        w(m, IVZ, k, j, i) = recovered.vz;
        w(m, IEN, k, j, i) = recovered.e;
        w(m, srrmhd::IRE1, k, j, i) = recovered.ex;
        w(m, srrmhd::IRE2, k, j, i) = recovered.ey;
        w(m, srrmhd::IRE3, k, j, i) = recovered.ez;
        bcc(m, IBX, k, j, i) = recovered.bx;
        bcc(m, IBY, k, j, i) = recovered.by;
        bcc(m, IBZ, k, j, i) = recovered.bz;
        const int source_stage = istage - 1;
        ru(source_stage, m, 0, k, j, i) = (recovered.ex - ex_star)/a_dt;
        ru(source_stage, m, 1, k, j, i) = (recovered.ey - ey_star)/a_dt;
        ru(source_stage, m, 2, k, j, i) = (recovered.ez - ez_star)/a_dt;
        if (viscosity_data.enabled) {
          const Real p11_star = visc_u(m, srrmhd::IVP11, k, j, i);
          const Real p22_star = visc_u(m, srrmhd::IVP22, k, j, i);
          const Real p33_star = visc_u(m, srrmhd::IVP33, k, j, i);
          const Real p12_star = visc_u(m, srrmhd::IVP12, k, j, i);
          const Real p13_star = visc_u(m, srrmhd::IVP13, k, j, i);
          const Real p23_star = visc_u(m, srrmhd::IVP23, k, j, i);
          visc_u(m, srrmhd::IVP11, k, j, i) = state.d*recovered_pi.p11;
          visc_u(m, srrmhd::IVP22, k, j, i) = state.d*recovered_pi.p22;
          visc_u(m, srrmhd::IVP33, k, j, i) = state.d*recovered_pi.p33;
          visc_u(m, srrmhd::IVP12, k, j, i) = state.d*recovered_pi.p12;
          visc_u(m, srrmhd::IVP13, k, j, i) = state.d*recovered_pi.p13;
          visc_u(m, srrmhd::IVP23, k, j, i) = state.d*recovered_pi.p23;
          visc_w(m, srrmhd::IVP11, k, j, i) = recovered_pi.p11;
          visc_w(m, srrmhd::IVP22, k, j, i) = recovered_pi.p22;
          visc_w(m, srrmhd::IVP33, k, j, i) = recovered_pi.p33;
          visc_w(m, srrmhd::IVP12, k, j, i) = recovered_pi.p12;
          visc_w(m, srrmhd::IVP13, k, j, i) = recovered_pi.p13;
          visc_w(m, srrmhd::IVP23, k, j, i) = recovered_pi.p23;
          ru(source_stage, m, 3+srrmhd::IVP11, k, j, i) =
              (state.d*recovered_pi.p11 - p11_star)/a_dt;
          ru(source_stage, m, 3+srrmhd::IVP22, k, j, i) =
              (state.d*recovered_pi.p22 - p22_star)/a_dt;
          ru(source_stage, m, 3+srrmhd::IVP33, k, j, i) =
              (state.d*recovered_pi.p33 - p33_star)/a_dt;
          ru(source_stage, m, 3+srrmhd::IVP12, k, j, i) =
              (state.d*recovered_pi.p12 - p12_star)/a_dt;
          ru(source_stage, m, 3+srrmhd::IVP13, k, j, i) =
              (state.d*recovered_pi.p13 - p13_star)/a_dt;
          ru(source_stage, m, 3+srrmhd::IVP23, k, j, i) =
              (state.d*recovered_pi.p23 - p23_star)/a_dt;
        }
      }
      max_iter = (iterations > max_iter) ? iterations : max_iter;
    }, Kokkos::Sum<int>(failures), Kokkos::Max<int>(max_iterations));

    pmy_pack->pmesh->ecounter.neos_fail += failures;
    pmy_pack->pmesh->ecounter.maxit_c2p =
        std::max(pmy_pack->pmesh->ecounter.maxit_c2p, max_iterations);
    if (failures > 0) return TaskStatus::fail;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::Fluxes
//! \brief Wrapper task list function that calls everything necessary to compute fluxes
//! of conserved variables

TaskStatus MHD::Fluxes(Driver *pdrive, int stage) {
  // select which calculate_flux function to call based on rsolver_method
  if (rsolver_method == MHD_RSolver::advect) {
    CalculateFluxes<MHD_RSolver::advect>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf) {
    CalculateFluxes<MHD_RSolver::llf>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle) {
    CalculateFluxes<MHD_RSolver::hlle>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlld) {
    CalculateFluxes<MHD_RSolver::hlld>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_sr) {
    CalculateFluxes<MHD_RSolver::llf_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle_sr) {
    CalculateFluxes<MHD_RSolver::hlle_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_srr) {
    CalculateFluxes<MHD_RSolver::llf_srr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_gr) {
    CalculateFluxes<MHD_RSolver::llf_gr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle_gr) {
    CalculateFluxes<MHD_RSolver::hlle_gr>(pdrive, stage);
  }

  if (relativistic_viscosity_data.enabled) CalculateViscousFluxes();

  // Add viscous, resistive, heat-flux, etc fluxes
  if (pvisc != nullptr) {
    pvisc->IsotropicViscousFlux(w0, pvisc->nu_iso, peos->eos_data, uflx);
  }
  if ((presist != nullptr) && (peos->eos_data.is_ideal)) {
    presist->OhmicEnergyFlux(b0, uflx);
  }
  if ((pbier != nullptr) && (peos->eos_data.is_ideal)) {
    pbier->BiermannEnergyFlux(w0, peos->eos_data, b0, uflx);
  }
  if (pcond != nullptr) {
    pcond->AddHeatFlux(w0, peos->eos_data, uflx);
  }

  // call FOFC if necessary
  if (use_fofc) {
    FOFC(pdrive, stage);
  } else if (pmy_pack->pcoord->is_general_relativistic) {
    if (pmy_pack->pcoord->coord_data.bh_excise) {
      FOFC(pdrive, stage);
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus MHD::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel)  {
    tstat = pbval_u->PackAndSendFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus MHD::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::MHDSrcTerms
//! \brief Wrapper task list function to apply source terms to conservative vars
//! Note source terms must be computed using only primitives (w0), as the conserved
//! variables (u0) have already been partially updated when this fn called.

TaskStatus MHD::MHDSrcTerms(Driver *pdrive, int stage) {
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Add source terms for various physics
  if (relativistic_viscosity_data.enabled
      && !relativistic_viscosity_data.linearized_target_1d) {
    const int viscosity_failures = AddRelativisticViscousSource(beta_dt);
    pmy_pack->pmesh->ecounter.neos_fail += viscosity_failures;
    if (viscosity_failures > 0) return TaskStatus::fail;
  }
  if (is_resistive_rel && !use_electric_ct) AddResistiveChargeSource(beta_dt);
  if (psrc->const_accel)  psrc->ConstantAccel(w0, peos->eos_data, beta_dt, u0);
  if (psrc->ism_cooling)  psrc->ISMCooling(w0, peos->eos_data, beta_dt, u0);
  if (psrc->rel_cooling) {
    psrc->RelCooling(w0, peos->eos_data, beta_dt, u0, pdrive, stage);
  }
  if (psrc->sn_driving)   psrc->SupernovaDriving(w0, peos->eos_data, beta_dt, u0);
  if (psrc->shearing_box) psrc->ShearingBox(w0, bcc0, peos->eos_data, beta_dt, u0);

  // Add coordinate source terms in GR.  Again, must be computed with only primitives.
  if (pmy_pack->pcoord->is_general_relativistic &&
      !pmy_pack->pcoord->is_dynamical_relativistic) {
    pmy_pack->pcoord->CoordSrcTerms(w0, bcc0, peos->eos_data, beta_dt, u0);
  } else if (pmy_pack->pcoord->is_dynamical_relativistic) {
    pmy_pack->pdyngr->AddCoordTerms(w0, bcc0, beta_dt, u0, pmy_pack->pmesh->mb_indcs.ng);
  }

  // Add user source terms
  if (pmy_pack->pmesh->pgen->user_srcs) {
    (pmy_pack->pmesh->pgen->user_srcs_func)(pmy_pack->pmesh, beta_dt);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::AddResistiveChargeSource()
//! \brief Add the explicit advective current -q v to Ampere's equation.
//!
//! The conductive part of the relativistic current is handled by the local implicit
//! solve.  Here q=div(E) uses CellCenteredCharge(), and v^i=u^i/u^0 converts AthenaK's
//! primitive spatial four-velocity to the physical three-velocity.

void MHD::AddResistiveChargeSource(const Real beta_dt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  auto w = w0;
  auto u = u0;

  par_for("srrmhd_charge_current", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real idx1 = 1.0/size.d_view(m).dx1;
    const Real idx2 = multi_d ? 1.0/size.d_view(m).dx2 : 0.0;
    const Real idx3 = three_d ? 1.0/size.d_view(m).dx3 : 0.0;
    const Real q = srrmhd::CellCenteredCharge(
        w, m, k, j, i, idx1, idx2, idx3, multi_d, three_d);
    const Real ux = w(m, IVX, k, j, i);
    const Real uy = w(m, IVY, k, j, i);
    const Real uz = w(m, IVZ, k, j, i);
    const Real ilor = 1.0/sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
    u(m, srrmhd::IRE1, k, j, i) -= beta_dt*q*ux*ilor;
    u(m, srrmhd::IRE2, k, j, i) -= beta_dt*q*uy*ilor;
    u(m, srrmhd::IRE3, k, j, i) -= beta_dt*q*uz*ilor;
  });
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendU_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus MHD::SendU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_u->PackAndSendCC(u0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvU_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection

TaskStatus MHD::RecvU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    Real qom = (psrc->qshear)*(psrc->omega0);
    tstat = porb_u->RecvAndUnpackCC(u0, recon_method, qom);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RestrictU
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus MHD::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus MHD::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  if (tstat != TaskStatus::complete) return tstat;
  if (relativistic_viscosity_data.enabled) {
    tstat = pbval_visc->PackAndSendCC(visc_u0, coarse_u0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus MHD::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  if (tstat != TaskStatus::complete) return tstat;
  if (relativistic_viscosity_data.enabled) {
    tstat = pbval_visc->RecvAndUnpackCC(visc_u0, coarse_u0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendU_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus MHD::SendU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_u->PackAndSendCC(u0, recon_method);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvU_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus MHD::RecvU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_u->RecvAndUnpackCC(u0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::EFieldSrc
//! \brief Wrapper task list function to apply source terms to electric field

TaskStatus MHD::EFieldSrc(Driver *pdrive, int stage) {
  // only execute when (shearing box defined) AND (2D)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->two_d)) {
    psrc->SBoxEField(b0, efld);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendE
//! \brief Wrapper task list function to pack/send fluxes of magnetic fields
//! (i.e. edge-centered electric field E) at MeshBlock boundaries. This is performed both
//! at MeshBlock boundaries at the same level (to keep magnetic flux in-sync on different
//! MeshBlocks), and at fine/coarse boundaries with SMR/AMR using restricted values of E.

TaskStatus MHD::SendE(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  tstat = pbval_b->PackAndSendFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvE
//! \brief Wrapper task list function to recv/unpack fluxes of magnetic fields
//! (i.e. edge-centered electric field E) at MeshBlock boundaries

TaskStatus MHD::RecvE(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  tstat = pbval_b->RecvAndUnpackFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendBEdge
//! \brief Pack/send the edge-centered magnetic flux used by dual CT.

TaskStatus MHD::SendBEdge(Driver *pdrive, int stage) {
  if (!use_electric_ct) return TaskStatus::complete;
  return pbval_e->PackAndSendFluxFC(bfld);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvBEdge
//! \brief Receive and average the dual-CT edge magnetic flux at block interfaces.

TaskStatus MHD::RecvBEdge(Driver *pdrive, int stage) {
  if (!use_electric_ct) return TaskStatus::complete;
  TaskStatus tstat = pbval_e->RecvAndUnpackFluxFC(bfld);
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_e->ClearFluxSend();
  if (tstat != TaskStatus::complete) return tstat;
  return pbval_e->ClearFluxRecv();
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendB_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus MHD::SendB_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_b->PackAndSendFC(b0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvB_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection

TaskStatus MHD::RecvB_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    Real qom = (psrc->qshear)*(psrc->omega0);
    tstat = porb_b->RecvAndUnpackFC(b0, recon_method, qom);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendB
//! \brief Wrapper task list function to pack/send face-centered magnetic fields

TaskStatus MHD::SendB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->PackAndSendFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvB
//! \brief Wrapper task list function to recv/unpack face-centered magnetic fields

TaskStatus MHD::RecvB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->RecvAndUnpackFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ExchangeElectricFaces
//! \brief Blocking same-level face-E exchange used only during initialization.

TaskStatus MHD::ExchangeElectricFaces(DvceFaceFld4D<Real> &eface) {
  if (!use_electric_ct || pmy_pack->pmesh->nmb_total == 1) {
    return TaskStatus::complete;
  }
  TaskStatus tstat = pbval_ect_face->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_ect_face->PackAndSendFC(eface, coarse_e0);
  if (tstat != TaskStatus::complete) return tstat;
  do {
    tstat = pbval_ect_face->RecvAndUnpackFC(eface, coarse_e0);
  } while (tstat == TaskStatus::incomplete);
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_ect_face->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;
  return pbval_ect_face->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::StartElectricFaceExchange
//! \brief Post a nonblocking same-level exchange of a face-electric register.

TaskStatus MHD::StartElectricFaceExchange(DvceFaceFld4D<Real> &eface) {
  if (!use_electric_ct || pmy_pack->pmesh->nmb_total == 1) {
    return TaskStatus::complete;
  }
  if (ect_comm_phase != 0) return TaskStatus::fail;
  TaskStatus tstat = pbval_ect_face->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_ect_face->PackAndSendFC(eface, coarse_e0);
  if (tstat == TaskStatus::complete) ect_comm_phase = 1;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::FinishElectricFaceExchange
//! \brief Poll and finish a face-electric exchange without host-side spinning.

TaskStatus MHD::FinishElectricFaceExchange(DvceFaceFld4D<Real> &eface) {
  if (!use_electric_ct || pmy_pack->pmesh->nmb_total == 1) {
    return TaskStatus::complete;
  }
  TaskStatus tstat;
  if (ect_comm_phase == 1) {
    tstat = pbval_ect_face->RecvAndUnpackFC(eface, coarse_e0);
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 2;
  }
  if (ect_comm_phase == 2) {
    tstat = pbval_ect_face->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 3;
  }
  if (ect_comm_phase == 3) {
    tstat = pbval_ect_face->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 0;
    return TaskStatus::complete;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::StartSharedElectricAverage
//! \brief Save local normal E faces and post their same-level exchange.

TaskStatus MHD::StartSharedElectricAverage(DvceFaceFld4D<Real> &eface) {
  if (!use_electric_ct || pmy_pack->pmesh->nmb_total == 1) {
    return TaskStatus::complete;
  }
  if (ect_comm_phase != 0) return TaskStatus::fail;
  Kokkos::deep_copy(DevExeSpace(), ect_face_prev.x1f, eface.x1f);
  Kokkos::deep_copy(DevExeSpace(), ect_face_prev.x2f, eface.x2f);
  Kokkos::deep_copy(DevExeSpace(), ect_face_prev.x3f, eface.x3f);
  TaskStatus tstat = pbval_ect_face->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_ect_face->PackAndSendFC(eface, coarse_e0);
  if (tstat == TaskStatus::complete) ect_comm_phase = 1;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::FinishSharedElectricAverage
//! \brief Poll the exchange and average local/neighbor copies of normal E.

TaskStatus MHD::FinishSharedElectricAverage(DvceFaceFld4D<Real> &eface) {
  if (!use_electric_ct || pmy_pack->pmesh->nmb_total == 1) {
    return TaskStatus::complete;
  }
  TaskStatus tstat;
  if (ect_comm_phase == 1) {
    tstat = pbval_ect_face->RecvAndUnpackFC(eface, coarse_e0);
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 2;
  }
  if (ect_comm_phase == 2) {
    tstat = pbval_ect_face->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 3;
  }
  if (ect_comm_phase == 3) {
    tstat = pbval_ect_face->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 0;
  } else {
    return TaskStatus::fail;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto prev = ect_face_prev;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const int ix1m = NeighborIndex(-1, 0, 0, 0, 0);
  const int ix1p = NeighborIndex(1, 0, 0, 0, 0);
  const int ix2m = NeighborIndex(0, -1, 0, 0, 0);
  const int ix2p = NeighborIndex(0, 1, 0, 0, 0);
  const int ix3m = NeighborIndex(0, 0, -1, 0, 0);
  const int ix3p = NeighborIndex(0, 0, 1, 0, 0);
  par_for(
    "dual_ct_average_shared_e1", DevExeSpace(), 0, nmb1, ks, ke, js, je,
    is, ie + 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
     const bool shared = ((i == is) && (nghbr.d_view(m, ix1m).gid >= 0)) ||
                         ((i == ie + 1) && (nghbr.d_view(m, ix1p).gid >= 0));
     if (shared) eface.x1f(m, k, j, i) =
       0.5 * (eface.x1f(m, k, j, i) + prev.x1f(m, k, j, i));
    });
  par_for(
    "dual_ct_average_shared_e2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1,
    is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
     const bool shared = ((j == js) && (nghbr.d_view(m, ix2m).gid >= 0)) ||
                         ((j == je + 1) && (nghbr.d_view(m, ix2p).gid >= 0));
     if (shared) eface.x2f(m, k, j, i) =
       0.5 * (eface.x2f(m, k, j, i) + prev.x2f(m, k, j, i));
    });
  if (three_d) {
    par_for(
      "dual_ct_average_shared_e3", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je,
      is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
       const bool shared = ((k == ks) && (nghbr.d_view(m, ix3m).gid >= 0)) ||
                           ((k == ke + 1) && (nghbr.d_view(m, ix3p).gid >= 0));
       if (shared) eface.x3f(m, k, j, i) =
         0.5 * (eface.x3f(m, k, j, i) + prev.x3f(m, k, j, i));
      });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::StartElectricCellExchange
//! \brief Post the nonblocking Picard cell-state exchange.

TaskStatus MHD::StartElectricCellExchange() {
  if (!use_electric_ct || pmy_pack->pmesh->nmb_total == 1) {
    return TaskStatus::complete;
  }
  if (ect_comm_phase != 0) return TaskStatus::fail;
  const bool viscosity = relativistic_viscosity_data.enabled;
  DvceArray5D<Real> state = u0;
  const int nstate = nmhd + nscalars + (viscosity ? srrmhd::NVISC : 0);
  if (viscosity) {
    state = ect_cell_state;
    Kokkos::deep_copy(Kokkos::subview(state, Kokkos::ALL,
                      std::make_pair(0,nmhd+nscalars), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), u0);
    Kokkos::deep_copy(Kokkos::subview(state, Kokkos::ALL,
                      std::make_pair(nmhd+nscalars,nstate), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), visc_u0);
  }
  TaskStatus tstat = pbval_ect_u->InitRecv(nstate);
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_ect_u->PackAndSendCC(state, coarse_u0);
  if (tstat == TaskStatus::complete) ect_comm_phase = 1;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::FinishElectricCellExchange
//! \brief Poll the Picard cell exchange and recover communicated ghost primitives.

TaskStatus MHD::FinishElectricCellExchange() {
  if (!use_electric_ct || pmy_pack->pmesh->nmb_total == 1) {
    return TaskStatus::complete;
  }
  TaskStatus tstat;
  const bool viscosity = relativistic_viscosity_data.enabled;
  DvceArray5D<Real> state = viscosity ? ect_cell_state : u0;
  if (ect_comm_phase == 1) {
    tstat = pbval_ect_u->RecvAndUnpackCC(state, coarse_u0);
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 2;
  }
  if (ect_comm_phase == 2) {
    tstat = pbval_ect_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 3;
  }
  if (ect_comm_phase == 3) {
    tstat = pbval_ect_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    ect_comm_phase = 0;
  } else {
    return TaskStatus::fail;
  }

  if (viscosity) {
    const int nstate = nmhd + nscalars + srrmhd::NVISC;
    Kokkos::deep_copy(u0, Kokkos::subview(state, Kokkos::ALL,
                      std::make_pair(0,nmhd+nscalars), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(visc_u0, Kokkos::subview(state, Kokkos::ALL,
                      std::make_pair(nmhd+nscalars,nstate), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL));
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1m1 = indcs.nx1 + 2 * indcs.ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? indcs.nx2 + 2 * indcs.ng - 1 : 0;
  const int n3m1 = (indcs.nx3 > 1) ? indcs.nx3 + 2 * indcs.ng - 1 : 0;
  if (viscosity) {
    return RecoverViscousPrimitives(visc_u0, 0.0, true, false,
                                    0, n1m1, 0, n2m1, 0, n3m1);
  }
  peos->ConsToPrim(u0, b0, w0, bcc0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendElectricFaces
//! \brief Exchange the primary face-centered electric field.

TaskStatus MHD::SendElectricFaces(Driver *pdrive, int stage) {
  return ExchangeElectricFaces(e0);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvElectricFaces
//! \brief Receive primary face E and rebuild its cell-centered conserved mirror.

TaskStatus MHD::RecvElectricFaces(Driver *pdrive, int stage) {
  if (!use_electric_ct) return TaskStatus::complete;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1m1 = indcs.nx1 + 2 * indcs.ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? indcs.nx2 + 2 * indcs.ng - 1 : 0;
  const int n3m1 = (indcs.nx3 > 1) ? indcs.nx3 + 2 * indcs.ng - 1 : 0;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto e = e0;
  auto u = u0;
  par_for(
    "dual_ct_sync_received_e", DevExeSpace(), 0, nmb1, 0, n3m1, 0, n2m1, 0,
    n1m1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
     Real ec1, ec2, ec3;
     srrmhd::ElectricFaceToCell(e, m, k, j, i, ec1, ec2, ec3);
     u(m, srrmhd::IRE1, k, j, i) = ec1;
     u(m, srrmhd::IRE2, k, j, i) = ec2;
     u(m, srrmhd::IRE3, k, j, i) = ec3;
    });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendB_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus MHD::SendB_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_b->PackAndSendFC(b0, recon_method);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvB_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus MHD::RecvB_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // only execute when (shearing box defined) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_b->RecvAndUnpackFC(b0);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ApplyPhysicalBCs
//! \brief Wrapper task list function to call funtions that set physical and user BCs

TaskStatus MHD::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  auto *pgen = pmy_pack->pmesh->pgen.get();
  bool run_user_bcs = pgen->user_bcs;

  // do not apply BCs in strictly periodic domains unless a user BC callback is enrolled
  if (pmy_pack->pmesh->strictly_periodic && pgen->user_bcs_func == nullptr) {
    return TaskStatus::complete;
  }

  // physical BCs are skipped for strictly periodic meshes
  if (!pmy_pack->pmesh->strictly_periodic) {
    pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);
    if (relativistic_viscosity_data.enabled) {
      pbval_visc->HydroBCs((pmy_pack), (pbval_visc->u_in), visc_u0);
    }
    pbval_b->BFieldBCs((pmy_pack), (pbval_b->b_in), b0);
  } else {
    // allow explicitly enrolled user BC callbacks to override periodic ghosts
    run_user_bcs = true;
  }

  // user BCs
  if (run_user_bcs && pgen->user_bcs_func != nullptr) {
    (pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse bundaries with SMR/AMR

TaskStatus MHD::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    pbval_b->FillCoarseInBndryFC(b0, coarse_b0);
    if (pmy_pack->pmesh->pmr->prolong_prims) {
      pbval_u->ConsToPrimCoarseBndry(coarse_u0, coarse_b0, coarse_w0);
      pbval_u->ProlongateCC(w0, coarse_w0);
      pbval_b->ProlongateFC(b0, coarse_b0);
      pbval_u->PrimToConsFineBndry(w0, b0, u0);
    } else {
      pbval_u->ProlongateCC(u0, coarse_u0);
      pbval_b->ProlongateFC(b0, coarse_b0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Recover a viscous SRMHD state with either fixed or implicitly relaxed shear.

TaskStatus MHD::RecoverViscousPrimitives(
    const DvceArray5D<Real> &visc_star, const Real shear_dt_over_tau,
    const bool fixed_spatial_shear, const bool update_conserved_shear,
    const int il, const int iu, const int jl, const int ju,
    const int kl, const int ku) {
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  auto u = u0;
  auto w = w0;
  auto b = b0;
  auto bcc = bcc0;
  auto visc_u = visc_u0;
  auto visc_w = visc_w0;
  auto eos = peos->eos_data;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const auto viscosity_data = relativistic_viscosity_data;
  int failures = 0;
  int max_iterations = 0;
  Kokkos::parallel_reduce("viscous_srrmhd_c2p",
  Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0, kl, jl, il},
                                         {nmb1+1, ku+1, ju+1, iu+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, int &sum_fail, int &max_iter) {
    srrmhd::SRRMHDCons1D state;
    state.d = u(m, IDN, k, j, i);
    state.mx = u(m, IM1, k, j, i);
    state.my = u(m, IM2, k, j, i);
    state.mz = u(m, IM3, k, j, i);
    state.e = u(m, IEN, k, j, i);
    state.ex = u(m, srrmhd::IRE1, k, j, i);
    state.ey = u(m, srrmhd::IRE2, k, j, i);
    state.ez = u(m, srrmhd::IRE3, k, j, i);
    state.bx = 0.5*(b.x1f(m, k, j, i) + b.x1f(m, k, j, i+1));
    state.by = 0.5*(b.x2f(m, k, j, i) + b.x2f(m, k, j+1, i));
    state.bz = 0.5*(b.x3f(m, k, j, i) + b.x3f(m, k+1, j, i));
    if (!(state.d > 0.0)) {
      ++sum_fail;
      return;
    }

    srrmhd::ShearStress pi_star;
    pi_star.p11 = visc_star(m, srrmhd::IVP11, k, j, i)/state.d;
    pi_star.p22 = visc_star(m, srrmhd::IVP22, k, j, i)/state.d;
    pi_star.p33 = visc_star(m, srrmhd::IVP33, k, j, i)/state.d;
    pi_star.p12 = visc_star(m, srrmhd::IVP12, k, j, i)/state.d;
    pi_star.p13 = visc_star(m, srrmhd::IVP13, k, j, i)/state.d;
    pi_star.p23 = visc_star(m, srrmhd::IVP23, k, j, i)/state.d;

    srrmhd::ShearStress pi_ns;
    if (!fixed_spatial_shear && viscosity_data.linearized_target_1d) {
      const int im = (i > 0) ? i - 1 : i;
      const int ip = (i < n1 - 1) ? i + 1 : i;
      const Real inv_length = (ip > im)
          ? 1.0/(static_cast<Real>(ip - im)*mbsize.d_view(m).dx1) : 0.0;
      const Real du1_dx = (w(m, IVX, k, j, ip) - w(m, IVX, k, j, im))*inv_length;
      const Real du2_dx = (w(m, IVY, k, j, ip) - w(m, IVY, k, j, im))*inv_length;
      const Real du3_dx = (w(m, IVZ, k, j, ip) - w(m, IVZ, k, j, im))*inv_length;
      const Real wgas = w(m, IDN, k, j, i) + eos.gamma*w(m, IEN, k, j, i);
      const Real eta_shear = wgas*viscosity_data.nu;
      pi_ns.p11 = -(4.0/3.0)*eta_shear*du1_dx;
      pi_ns.p22 = (2.0/3.0)*eta_shear*du1_dx;
      pi_ns.p33 = (2.0/3.0)*eta_shear*du1_dx;
      pi_ns.p12 = -eta_shear*du2_dx;
      pi_ns.p13 = -eta_shear*du3_dx;
      pi_ns = srrmhd::ProjectShearStress(
          w(m, IVX, k, j, i), w(m, IVY, k, j, i), w(m, IVZ, k, j, i), pi_ns);
    }

    srrmhd::SRRMHDPrim1D guess;
    guess.d = w(m, IDN, k, j, i);
    guess.vx = w(m, IVX, k, j, i);
    guess.vy = w(m, IVY, k, j, i);
    guess.vz = w(m, IVZ, k, j, i);
    guess.e = w(m, IEN, k, j, i);
    guess.ex = state.ex;
    guess.ey = state.ey;
    guess.ez = state.ez;
    guess.bx = state.bx;
    guess.by = state.by;
    guess.bz = state.bz;
    if (!isfinite(guess.vx) || !isfinite(guess.vy) || !isfinite(guess.vz)) {
      guess.vx = 0.0;
      guess.vy = 0.0;
      guess.vz = 0.0;
    }

    srrmhd::SRRMHDPrim1D recovered = guess;
    srrmhd::ShearStress recovered_pi;
    int iterations = 0;
    const bool success = srrmhd::SingleC2P_IdealSRRMHDImplicitViscous(
        state, eos, state.ex, state.ey, state.ez, 0.0, pi_star, pi_ns,
        shear_dt_over_tau,
        fixed_spatial_shear ? 1.0e30 : viscosity_data.chi_max,
        fixed_spatial_shear, guess, recovered, recovered_pi, iterations);
    if (!success) {
      ++sum_fail;
    } else {
      w(m, IDN, k, j, i) = recovered.d;
      w(m, IVX, k, j, i) = recovered.vx;
      w(m, IVY, k, j, i) = recovered.vy;
      w(m, IVZ, k, j, i) = recovered.vz;
      w(m, IEN, k, j, i) = recovered.e;
      w(m, srrmhd::IRE1, k, j, i) = recovered.ex;
      w(m, srrmhd::IRE2, k, j, i) = recovered.ey;
      w(m, srrmhd::IRE3, k, j, i) = recovered.ez;
      bcc(m, IBX, k, j, i) = recovered.bx;
      bcc(m, IBY, k, j, i) = recovered.by;
      bcc(m, IBZ, k, j, i) = recovered.bz;
      visc_w(m, srrmhd::IVP11, k, j, i) = recovered_pi.p11;
      visc_w(m, srrmhd::IVP22, k, j, i) = recovered_pi.p22;
      visc_w(m, srrmhd::IVP33, k, j, i) = recovered_pi.p33;
      visc_w(m, srrmhd::IVP12, k, j, i) = recovered_pi.p12;
      visc_w(m, srrmhd::IVP13, k, j, i) = recovered_pi.p13;
      visc_w(m, srrmhd::IVP23, k, j, i) = recovered_pi.p23;
      if (update_conserved_shear) {
        visc_u(m, srrmhd::IVP11, k, j, i) = state.d*recovered_pi.p11;
        visc_u(m, srrmhd::IVP22, k, j, i) = state.d*recovered_pi.p22;
        visc_u(m, srrmhd::IVP33, k, j, i) = state.d*recovered_pi.p33;
        visc_u(m, srrmhd::IVP12, k, j, i) = state.d*recovered_pi.p12;
        visc_u(m, srrmhd::IVP13, k, j, i) = state.d*recovered_pi.p13;
        visc_u(m, srrmhd::IVP23, k, j, i) = state.d*recovered_pi.p23;
      }
    }
    max_iter = (iterations > max_iter) ? iterations : max_iter;
  }, Kokkos::Sum<int>(failures), Kokkos::Max<int>(max_iterations));

  pmy_pack->pmesh->ecounter.neos_fail += failures;
  pmy_pack->pmesh->ecounter.maxit_c2p =
      std::max(pmy_pack->pmesh->ecounter.maxit_c2p, max_iterations);
  if (failures > 0) return TaskStatus::fail;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ConToPrim
//! \brief Wrapper task list function to call ConsToPrim over entire mesh (including gz)

TaskStatus MHD::ConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  if (!relativistic_viscosity_data.enabled) {
    peos->ConsToPrim(u0, b0, w0, bcc0, false, 0, n1m1, 0, n2m1, 0, n3m1);
    return TaskStatus::complete;
  }
  return RecoverViscousPrimitives(visc_u0, 0.0, true, false,
                                  0, n1m1, 0, n2m1, 0, n3m1);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears U, B, Flx_U, Flx_B, U_OA, B_OA, U_Shr, BShr
//! If (last stage)>stage>=(0): clears U, B, Flx_U, Flx_B,             U_Shr, B_Shr
//! If stage=(-1):              clears U, B
//! If stage=(-4):              clears                                 U_Shr, B_Shr

TaskStatus MHD::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat;
  if ((stage >= 0) || (stage == -1)) {
    // check sends of U complete
    TaskStatus tstat = pbval_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    if (relativistic_viscosity_data.enabled) {
      tstat = pbval_visc->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // check sends of B complete
    tstat = pbval_b->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check sends for fluxes of U complete.  Always check sends of E complete
  // do not check flux send for ICs (stage < 0)
  if (stage >= 0) {
    // with SMR/AMR check sends of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->ClearFluxSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // check sends of restricted fluxes of B complete even for uniform grids
    tstat = pbval_b->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;
    if (use_electric_ct) {
      tstat = pbval_e->ClearFluxSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with orbital advection check sends for U and B complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = porb_b->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with shearing box boundaries check sends of U and B complete
  // only execute when (shearing box defined) AND (stage>=0 or -4) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && ((stage >= 0) || (stage == -4)) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = psbox_b->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears U, B, Flx_U, Flx_B, U_OA, B_OA, U_Shr, BShr
//! If (last stage)>stage>=(0): clears U, B, Flx_U, Flx_B,             U_Shr, B_Shr
//! If stage=(-1):              clears U, B
//! If stage=(-4):              clears                                 U_Shr, B_Shr

TaskStatus MHD::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat;
  if ((stage >= 0) || (stage == -1)) {
    // check receives of U complete
    tstat = pbval_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    if (relativistic_viscosity_data.enabled) {
      tstat = pbval_visc->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // check receives of B complete
    tstat = pbval_b->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check recvs for fluxes of U complete.  Always check recvs of E complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR check receives of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->ClearFluxRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // with SMR/AMR check receives of restricted fluxes of B complete
    tstat = pbval_b->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;
    if (use_electric_ct) {
      tstat = pbval_e->ClearFluxRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with orbital advection check receives of U and B are complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && (stage == pdrive->nexp_stages) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = porb_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = porb_b->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with shearing box boundaries check receives of U and B complete
  // only execute when (shearing box defined) AND (stage>=0 or -4) AND (3D OR 2d_r_phi)
  if ((psrc->shearing_box) && ((stage >= 0) || (stage == -4)) &&
      (pmy_pack->pmesh->three_d || psrc->shearing_box_r_phi)) {
    tstat = psbox_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = psbox_b->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RestrictB
//! \brief Wrapper function that restricts face-centered variables (magnetic field)

TaskStatus MHD::RestrictB(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictFC(b0, coarse_b0);
  }
  return TaskStatus::complete;
}

} // namespace mhd
