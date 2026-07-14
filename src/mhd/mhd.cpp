//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.cpp
//! \brief implementation of MHD class constructor and assorted functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/biermann.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "bvals/bvals.hpp"
#include "eos/resistive_srmhd.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHD::MHD(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    visc_u0("visc_cons",1,1,1,1,1),
    visc_w0("visc_prim",1,1,1,1,1),
    b0("B_fc",1,1,1,1),
    e0("E_fc",1,1,1,1),
    bcc0("B_cc",1,1,1,1,1),
    eta_cell("eta_cell",1,1,1,1),
    eta_face("eta_face",1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    coarse_b0("cB_fc",1,1,1,1),
    coarse_e0("cE_fc",1,1,1,1),
    u1("cons1",1,1,1,1,1),
    visc_u1("visc_cons1",1,1,1,1,1),
    visc_ustar("visc_cons_star",1,1,1,1,1),
    ect_cell_state("ect_cell_state",1,1,1,1,1),
    b1("B_fc1",1,1,1,1),
    e1("E_fc1",1,1,1,1),
    jfc("J_fc",1,1,1,1),
    estar("E_fc_star",1,1,1,1),
    ect_face_prev("E_fc_prev",1,1,1,1),
    ect_src("E_fc_src",1,1,1,1,1),
    ect_u_prev("ect_u_prev",1,1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    visc_flx("visc_flx",1,1,1,1,1),
    efld("efld",1,1,1,1),
    bfld("bfld",1,1,1,1),
    wsaved("wsaved",1,1,1,1,1),
    bccsaved("bccsaved",1,1,1,1,1),
    e3x1("e3x1",1,1,1,1),
    e2x1("e2x1",1,1,1,1),
    e1x2("e1x2",1,1,1,1),
    e3x2("e3x2",1,1,1,1),
    e2x3("e2x3",1,1,1,1),
    e1x3("e1x3",1,1,1,1),
    e1_cc("e1_cc",1,1,1,1),
    e2_cc("e2_cc",1,1,1,1),
    e3_cc("e3_cc",1,1,1,1),
    utest("utest",1,1,1,1,1),
    bcctest("bcctest",1,1,1,1,1),
    visc_utest("visc_utest",1,1,1,1,1),
    fofc("fofc",1,1,1,1),
    eta1("eta1",1,1,1,1),
    eta2("eta2",1,1,1,1),
    eta3("eta3",1,1,1,1) {
  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));

  // (1) construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("mhd","eos");
  is_resistive_rel = pin->GetOrAddBoolean("mhd", "resistive_rel", false);
  relativistic_viscosity_data.enabled =
      pin->GetOrAddBoolean("mhd", "relativistic_viscosity", false);
  use_electric_ct = pin->GetOrAddBoolean("mhd", "electric_ct", false);
  if (relativistic_viscosity_data.enabled && !is_resistive_rel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd>/relativistic_viscosity=true requires "
              << "<mhd>/resistive_rel=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (use_electric_ct && !is_resistive_rel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd>/electric_ct=true requires "
              << "<mhd>/resistive_rel=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (is_resistive_rel) {
    if (!(pmy_pack->pcoord->is_special_relativistic)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/resistive_rel=true requires "
                << "<coord>/special_rel=true" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (eqn_of_state.compare("ideal") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Resistive SRMHD currently requires <mhd>/eos=ideal"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pmy_pack->pmesh->multilevel) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Resistive SRMHD does not yet support SMR or AMR"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pin->DoesBlockExist("ion-neutral")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Resistive SRMHD cannot yet be combined with "
                << "ion-neutral coupling" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pin->DoesBlockExist("shearing_box")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Resistive SRMHD does not yet support shearing-box "
                << "coupling" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pin->DoesBlockExist("turb_driving") &&
        pin->GetOrAddString("turb_driving", "relativistic_forcing", "legacy")
            != "mechanical") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Resistive SRMHD turbulence driving requires "
                << "<turb_driving>/relativistic_forcing=mechanical" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pin->DoesParameterExist("mhd", "ohmic_resistivity") ||
        pin->DoesParameterExist("mhd", "viscosity") ||
        pin->DoesParameterExist("mhd", "biermann_coeff") ||
        pin->DoesParameterExist("mhd", "biermann_from_cgs") ||
        pin->DoesParameterExist("mhd", "conductivity") ||
        pin->DoesParameterExist("mhd", "tdep_conductivity")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Resistive SRMHD cannot yet be combined with legacy "
                << "MHD diffusion modules" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::string eta_model =
        pin->GetOrAddString("mhd", "resistivity_model", "uniform");
    if (eta_model.compare("uniform") == 0) {
      resistivity_data.model = srrmhd::ResistivityModel::uniform;
      resistivity = pin->GetReal("mhd", "resistivity");
      if (resistivity <= 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Uniform resistive SRMHD requires "
                  << "<mhd>/resistivity > 0" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      resistivity_data.eta_uniform = resistivity;
    } else if (eta_model.compare("charge_starvation") == 0) {
      resistivity_data.model = srrmhd::ResistivityModel::charge_starvation;
      resistivity_data.eta_floor = pin->GetOrAddReal("mhd", "eta_floor", 1.0e-8);
      resistivity_data.eta_scale = pin->GetReal("mhd", "eta_scale");
      resistivity_data.number_per_mass =
          pin->GetOrAddReal("mhd", "number_per_mass", 1.0);
      if (resistivity_data.eta_floor <= 0.0 ||
          resistivity_data.eta_scale <= 0.0 ||
          resistivity_data.number_per_mass <= 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Charge-starvation resistivity requires positive "
                  << "<mhd>/eta_floor, eta_scale, and number_per_mass" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Retain a positive scalar for legacy diagnostics.  Dynamic implicit paths use
      // only the frozen eta_cell/eta_face coefficients in this mode.
      resistivity = resistivity_data.eta_floor;
      resistivity_data.eta_uniform = resistivity;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unknown <mhd>/resistivity_model='" << eta_model
                << "' (expected uniform or charge_starvation)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (relativistic_viscosity_data.enabled) {
      relativistic_viscosity_data.nu = pin->GetReal("mhd", "shear_viscosity");
      relativistic_viscosity_data.tau =
          pin->GetReal("mhd", "shear_relaxation_time");
      relativistic_viscosity_data.linearized_target_1d = pin->GetOrAddBoolean(
          "mhd", "linearized_shear_target_1d", false);
      relativistic_viscosity_data.chi_max =
          pin->GetOrAddReal("mhd", "shear_chi_max", 2.0);
      if (relativistic_viscosity_data.nu < 0.0 ||
          relativistic_viscosity_data.tau <= 0.0 ||
          relativistic_viscosity_data.chi_max <= 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Relativistic viscosity requires "
                  << "<mhd>/shear_viscosity >= 0, shear_relaxation_time > 0, "
                  << "and shear_chi_max > 0"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const Real gamma = pin->GetReal("mhd", "gamma");
      const Real causal_margin = srrmhd::LinearViscosityCausalityMargin(
          gamma, relativistic_viscosity_data);
      if (causal_margin < -1.0e-12) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Relativistic viscosity violates the conservative "
                  << "linear causality gate: (gamma-1) + 4*shear_viscosity/"
                  << "(3*shear_relaxation_time) must be <= 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (is_resistive_rel) {
      peos = new IdealSRRMHD(ppack, pin);
      nmhd = srrmhd::NSRRMHD;
    } else if (pmy_pack->pcoord->is_special_relativistic) {
      peos = new IdealSRMHD(ppack, pin);
      nmhd = 5;
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      peos = new IdealGRMHD(ppack, pin);
      nmhd = 5;
    } else {
      peos = new IdealMHD(ppack, pin);
      nmhd = 5;
    }

  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
                <<"<mhd> eos = isothermal cannot be used with SR/GR"<< std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalMHD(ppack, pin);
      nmhd = 4;
    }

  // EOS string not recognized
  } else {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
              <<"<mhd> eos = '"<< eqn_of_state <<"' not implemented"<< std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("mhd","nscalars",0);
  if (is_resistive_rel && nscalars != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Resistive SRMHD does not yet support passive scalars"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Viscosity (only constructed if needed)
  if (pin->DoesParameterExist("mhd","viscosity")) {
    pvisc = new Viscosity("mhd", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Resistivity (only constructed if needed)
  if (pin->DoesParameterExist("mhd","ohmic_resistivity")) {
    presist = new Resistivity(ppack, pin);
  } else {
    presist = nullptr;
  }

  // Biermann battery (only constructed if needed)
  if (pin->DoesParameterExist("mhd","biermann_coeff") ||
      pin->DoesParameterExist("mhd","biermann_from_cgs")) {
    pbier = new BiermannBattery(ppack, pin);
  } else {
    pbier = nullptr;
  }

  // Thermal conduction (only constructed if needed)
  if (pin->DoesParameterExist("mhd","conductivity") ||
      pin->DoesParameterExist("mhd","tdep_conductivity")) {
    pcond = new Conduction("mhd", ppack, pin);
  } else {
    pcond = nullptr;
  }

  // Source terms (constructor parses input file to initialize only srcterms needed)
  psrc = new SourceTerms("mhd", ppack, pin);

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");
  if (is_resistive_rel && evolution_t.compare("static") != 0) {
    if (relativistic_viscosity_data.enabled && pmy_pack->pmesh->multilevel) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dynamic relativistic viscosity does not yet support "
                << "mesh refinement" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (relativistic_viscosity_data.enabled && pmy_pack->pmesh->multi_d
        && relativistic_viscosity_data.linearized_target_1d) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "linearized_shear_target_1d is only valid on "
                << "one-dimensional meshes" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::string integrator = pin->GetOrAddString("time", "integrator", "rk2");
    if (use_electric_ct) {
      if (pmy_pack->pmesh->one_d && pmy_pack->pmesh->nmb_total != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Dynamic dual CT currently supports a single-block "
                  << "one-dimensional mesh, or a uniform multidimensional mesh"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (integrator.compare("imex2") != 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Dynamic dual CT requires the IMEX-SSP2(3,2,2) "
                  << "integrator selected by <time>/integrator=imex2" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (integrator.compare("imex2") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dynamic resistive SRMHD requires "
                << "<time>/integrator=imex2" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // allocate memory for conserved and primitive variables
  // With AMR, maximum size of Views are limited by total device memory through an input
  // parameter, which in turn limits max number of MBs that can be created.
  {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(u0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(w0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    if (relativistic_viscosity_data.enabled) {
      Kokkos::realloc(visc_u0, nmb, srrmhd::NVISC, ncells3, ncells2, ncells1);
      Kokkos::realloc(visc_w0, nmb, srrmhd::NVISC, ncells3, ncells2, ncells1);
      Kokkos::deep_copy(visc_u0, 0.0);
      Kokkos::deep_copy(visc_w0, 0.0);
    }

    // allocate memory for face-centered and cell-centered magnetic fields
    Kokkos::realloc(bcc0,   nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(b0.x1f, nmb, ncells3, ncells2, ncells1+1);
    Kokkos::realloc(b0.x2f, nmb, ncells3, ncells2+1, ncells1);
    Kokkos::realloc(b0.x3f, nmb, ncells3+1, ncells2, ncells1);
    if (resistivity_data.model != srrmhd::ResistivityModel::uniform) {
      Kokkos::realloc(eta_cell, nmb, ncells3, ncells2, ncells1);
    }
    if (use_electric_ct) {
      Kokkos::realloc(e0.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(e0.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(e0.x3f, nmb, ncells3+1, ncells2, ncells1);
      Kokkos::realloc(e1.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(e1.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(e1.x3f, nmb, ncells3+1, ncells2, ncells1);
      Kokkos::realloc(jfc.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(jfc.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(jfc.x3f, nmb, ncells3+1, ncells2, ncells1);
      Kokkos::realloc(estar.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(estar.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(estar.x3f, nmb, ncells3+1, ncells2, ncells1);
      Kokkos::realloc(ect_face_prev.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(ect_face_prev.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(ect_face_prev.x3f, nmb, ncells3+1, ncells2, ncells1);
      Kokkos::realloc(ect_src.x1f, nmb, 3, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(ect_src.x2f, nmb, 3, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(ect_src.x3f, nmb, 3, ncells3+1, ncells2, ncells1);
      Kokkos::realloc(ect_u_prev, nmb, 3, ncells3, ncells2, ncells1);
      Kokkos::realloc(bfld.x1e, nmb, ncells3+1, ncells2+1, ncells1);
      Kokkos::realloc(bfld.x2e, nmb, ncells3+1, ncells2, ncells1+1);
      Kokkos::realloc(bfld.x3e, nmb, ncells3, ncells2+1, ncells1+1);
      if (resistivity_data.model != srrmhd::ResistivityModel::uniform) {
        Kokkos::realloc(eta_face.x1f, nmb, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(eta_face.x2f, nmb, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(eta_face.x3f, nmb, ncells3+1, ncells2, ncells1);
      }
    }
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nmhd+nscalars), n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_w0, nmb, (nmhd+nscalars), n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_b0.x1f, nmb, n_ccells3, n_ccells2, n_ccells1+1);
    Kokkos::realloc(coarse_b0.x2f, nmb, n_ccells3, n_ccells2+1, n_ccells1);
    Kokkos::realloc(coarse_b0.x3f, nmb, n_ccells3+1, n_ccells2, n_ccells1);
    if (use_electric_ct) {
      Kokkos::realloc(coarse_e0.x1f, nmb, n_ccells3, n_ccells2, n_ccells1+1);
      Kokkos::realloc(coarse_e0.x2f, nmb, n_ccells3, n_ccells2+1, n_ccells1);
      Kokkos::realloc(coarse_e0.x3f, nmb, n_ccells3+1, n_ccells2, n_ccells1);
    }
  }

  // allocate boundary buffers for conserved (cell-centered) and face-centered variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_u->InitializeBuffers((nmhd+nscalars));
  if (relativistic_viscosity_data.enabled) {
    pbval_visc = new MeshBoundaryValuesCC(ppack, pin, false);
    pbval_visc->InitializeBuffers(srrmhd::NVISC);
  }
  pbval_b = new MeshBoundaryValuesFC(ppack, pin);
  pbval_b->InitializeBuffers(3);
  if (use_electric_ct) {
    pbval_e = new MeshBoundaryValuesFC(ppack, pin);
    pbval_e->InitializeBuffers(3);
    pbval_ect_face = new MeshBoundaryValuesFC(ppack, pin);
    pbval_ect_face->InitializeBuffers(3);
    pbval_ect_u = new MeshBoundaryValuesCC(ppack, pin, false);
    const int nect = nmhd + nscalars
                     + (relativistic_viscosity_data.enabled ? srrmhd::NVISC : 0);
    pbval_ect_u->InitializeBuffers(nect);
  }

  // Orbital advection and shearing box BCs (if requested in input file)
  if (pin->DoesBlockExist("shearing_box")) {
    porb_u = new OrbitalAdvectionCC(ppack, pin, (nmhd+nscalars));
    porb_b = new OrbitalAdvectionFC(ppack, pin);
    psbox_u = new ShearingBoxBoundaryCC(ppack, pin, (nmhd+nscalars));
    psbox_b = new ShearingBoxBoundaryFC(ppack, pin);
  } else {
    porb_u = nullptr;
    porb_b = nullptr;
    psbox_u = nullptr;
    psbox_b = nullptr;
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("static") != 0) {
    // determine if FOFC is enabled
    use_fofc = pin->GetOrAddBoolean("mhd","fofc",false);
    force_fofc = pin->GetOrAddBoolean("mhd", "fofc_force", false);
    if (force_fofc && !use_fofc) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/fofc_force=true requires <mhd>/fofc=true"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // determine if h-correction is enabled (Sanders, Morano & Druguet 1998)
    use_hcorr = pin->GetOrAddBoolean("mhd","h_correction",false);

    // select reconstruction method (default PLM)
    std::string xorder = pin->GetOrAddString("mhd","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
      // check that nghost > 2 with PLM+FOFC
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (use_fofc && indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 3 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // check that nghost > 3 with PPM4(or PPMX or WENOZ)+FOFC
      if (use_fofc && indcs.ng < 4) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 4 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select Riemann solver (no default).  Test for compatibility of options
    std::string rsolver = pin->GetString("mhd","rsolver");
    // Special relativistic solvers
    if (pmy_pack->pcoord->is_special_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (is_resistive_rel && rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_srr;
        } else if (is_resistive_rel) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver
                    << "' not implemented for resistive SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        } else if (rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_sr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = MHD_RSolver::hlle_sr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                    << " for SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for SR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic solvers
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_gr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = MHD_RSolver::hlle_gr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                    << " for GR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for GR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic dynamic solvers
    } else if (evolution_t.compare("dynamic") == 0) {
      // LLF solver
      if (rsolver.compare("llf") == 0) {
        rsolver_method = MHD_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = MHD_RSolver::hlle;
      // HLLD solver
      } else if (rsolver.compare("hlld") == 0) {
        rsolver_method = MHD_RSolver::hlld;
      // Roe solver
      // } else if (rsolver.compare("roe") == 0) {
      //   rsolver_method = MHD_RSolver::roe;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << " for dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic kinematic solver
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        rsolver_method = MHD_RSolver::advect;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << " for kinematic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    // check h-correction is only used with HLLD
    if (use_hcorr && rsolver_method != MHD_RSolver::hlld) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "h_correction requires rsolver=hlld" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Final memory allocations
    {
      // allocate second registers
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(u1,     nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
      if (relativistic_viscosity_data.enabled) {
        Kokkos::realloc(visc_u1, nmb, srrmhd::NVISC, ncells3, ncells2, ncells1);
        if (use_electric_ct) {
          Kokkos::realloc(visc_ustar, nmb, srrmhd::NVISC,
                          ncells3, ncells2, ncells1);
          Kokkos::realloc(ect_cell_state, nmb, nmhd+nscalars+srrmhd::NVISC,
                          ncells3, ncells2, ncells1);
        }
      }
      Kokkos::realloc(b1.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(b1.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(b1.x3f, nmb, ncells3+1, ncells2, ncells1);

      // allocate fluxes, electric fields
      Kokkos::realloc(uflx.x1f, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1+1);
      Kokkos::realloc(uflx.x2f, nmb, (nmhd+nscalars), ncells3, ncells2+1, ncells1);
      Kokkos::realloc(uflx.x3f, nmb, (nmhd+nscalars), ncells3+1, ncells2, ncells1);
      if (relativistic_viscosity_data.enabled) {
        Kokkos::realloc(visc_flx.x1f, nmb, srrmhd::NVISC, ncells3, ncells2,
                        ncells1+1);
        Kokkos::realloc(visc_flx.x2f, nmb, srrmhd::NVISC, ncells3, ncells2+1,
                        ncells1);
        Kokkos::realloc(visc_flx.x3f, nmb, srrmhd::NVISC, ncells3+1, ncells2,
                        ncells1);
      }
      Kokkos::realloc(efld.x1e, nmb, ncells3+1, ncells2+1, ncells1);
      Kokkos::realloc(efld.x2e, nmb, ncells3+1, ncells2, ncells1+1);
      Kokkos::realloc(efld.x3e, nmb, ncells3, ncells2+1, ncells1+1);

      // allocate scratch arrays for face- and cell-centered E used in CornerE
      Kokkos::realloc(e3x1, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2x1, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1x2, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e3x2, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2x3, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1x3, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1_cc, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2_cc, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e3_cc, nmb, ncells3, ncells2, ncells1);

      // allocate array of flags used with FOFC
      if (use_fofc) {
        int nvars = (pmy_pack->pcoord->is_dynamical_relativistic) ? nmhd+nscalars : nmhd;
        Kokkos::realloc(fofc,    nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(utest,   nmb, nvars, ncells3, ncells2, ncells1);
        Kokkos::realloc(bcctest, nmb, 3,    ncells3, ncells2, ncells1);
        if (relativistic_viscosity_data.enabled) {
          Kokkos::realloc(visc_utest, nmb, srrmhd::NVISC,
                          ncells3, ncells2, ncells1);
        }
        Kokkos::deep_copy(fofc, false);
      }

      // allocate cell-centered max eigenvalue arrays for h-correction
      if (use_hcorr) {
        Kokkos::realloc(eta1, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(eta2, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(eta3, nmb, ncells3, ncells2, ncells1);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

MHD::~MHD() {
  delete peos;
  delete pbval_u;
  if (pbval_visc != nullptr) {
    delete pbval_visc;
  }
  delete pbval_b;
  if (pbval_e != nullptr) {delete pbval_e;}
  if (pbval_ect_face != nullptr) {delete pbval_ect_face;}
  if (pbval_ect_u != nullptr) {delete pbval_ect_u;}
  if (pvisc != nullptr) {delete pvisc;}
  if (presist!= nullptr) {delete presist;}
  if (pbier  != nullptr) {delete pbier;}
  if (pcond != nullptr) {delete pcond;}
  if (psrc!= nullptr) {delete psrc;}
}

//----------------------------------------------------------------------------------------
// SetSaveWBcc:  set flag to save primitives and cell-centered B field, e.g., for jcon

void MHD::SetSaveWBcc() {
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  // allocated saved arrays for time derivatives
  Kokkos::realloc(wsaved,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(bccsaved, nmb, 3,               ncells3, ncells2, ncells1);

  wbcc_saved = true;
}

} // namespace mhd
