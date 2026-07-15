//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsrmhd_charged_vortex.cpp
//! \brief Exact charged-vortex equilibrium for two- and three-dimensional SRRMHD.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/resistive_srmhd.hpp"
#include "mhd/dual_ct.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {

enum class VortexPlane {xy = 0, yz = 1, zx = 2};

KOKKOS_INLINE_FUNCTION
void ChargedVortexState(const Real x, const Real y, const Real q0, const Real p0,
                        const Real rho, const Real gamma, Real &ex, Real &ey,
                        Real &bz, Real &ux, Real &uy, Real &pressure, Real &charge) {
  const Real r2 = x*x + y*y;
  const Real s = r2 + 1.0;
  const Real root = sqrt(s*s - 0.25*q0*q0);
  ex = 0.5*q0*x/s;
  ey = 0.5*q0*y/s;
  bz = root/s;

  // The analytic solution specifies the azimuthal three-velocity.  AthenaK stores
  // spatial four-velocity, so convert v^i to u^i=gamma_L v^i.
  const Real vx = 0.5*q0*y/root;
  const Real vy = -0.5*q0*x/root;
  const Real lor = 1.0/sqrt(1.0 - vx*vx - vy*vy);
  ux = lor*vx;
  uy = lor*vy;

  const Real gm1 = gamma - 1.0;
  const Real pressure_factor = (4.0*r2 + 4.0 - q0*q0)/(s*(4.0 - q0*q0));
  pressure = -rho*gm1/gamma
      +(p0 + rho*gm1/gamma)*pow(pressure_factor, gamma/(2.0*gm1));
  charge = q0/(s*s);
}

KOKKOS_INLINE_FUNCTION
void ChargedVortexPlaneState(const Real x1, const Real x2, const Real x3,
                             const int plane, const Real q0, const Real p0,
                             const Real rho, const Real gamma, Real &e1, Real &e2,
                             Real &e3, Real &b1, Real &b2, Real &b3, Real &u1,
                             Real &u2, Real &u3, Real &pressure, Real &charge) {
  Real a, b;
  if (plane == static_cast<int>(VortexPlane::xy)) {
    a = x1;
    b = x2;
  } else if (plane == static_cast<int>(VortexPlane::yz)) {
    a = x2;
    b = x3;
  } else {
    a = x3;
    b = x1;
  }

  Real ea, eb, bc, ua, ub;
  ChargedVortexState(a, b, q0, p0, rho, gamma, ea, eb, bc, ua, ub,
                     pressure, charge);
  e1 = e2 = e3 = 0.0;
  b1 = b2 = b3 = 0.0;
  u1 = u2 = u3 = 0.0;
  if (plane == static_cast<int>(VortexPlane::xy)) {
    e1 = ea;
    e2 = eb;
    b3 = bc;
    u1 = ua;
    u2 = ub;
  } else if (plane == static_cast<int>(VortexPlane::yz)) {
    e2 = ea;
    e3 = eb;
    b1 = bc;
    u2 = ua;
    u3 = ub;
  } else {
    e3 = ea;
    e1 = eb;
    b2 = bc;
    u3 = ua;
    u1 = ub;
  }
}

int GetVortexPlane(ParameterInput *pin, const bool three_d) {
  const std::string name = pin->GetOrAddString("problem", "plane", "xy");
  if (name == "xy") return static_cast<int>(VortexPlane::xy);
  if (three_d && name == "yz") return static_cast<int>(VortexPlane::yz);
  if (three_d && name == "zx") return static_cast<int>(VortexPlane::zx);
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "<problem>/plane must be xy in 2D or one of xy, yz, zx in 3D"
            << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

void SRRMHDChargedVortexErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ResistiveSRMHDChargedVortex()
//! \brief Initialize the exact stationary charged-vortex equilibrium.
//!
//! In 3D, <problem>/plane selects xy, yz, or zx.  The solution is constant along the
//! remaining direction, so the three cyclic choices exercise all three CT edge fields.

void ProblemGenerator::ResistiveSRMHDChargedVortex(ParameterInput *pin,
                                                    const bool restart) {
  pgen_final_func = SRRMHDChargedVortexErrors;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  if (pmhd == nullptr || !(pmhd->is_resistive_rel) ||
      !(pmy_mesh_->two_d || pmy_mesh_->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "rsrmhd_charged_vortex requires two- or three-dimensional "
              << "resistive SRMHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real q0 = pin->GetOrAddReal("problem", "q0", 0.7);
  const Real p0 = pin->GetOrAddReal("problem", "p0", 0.1);
  const Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real nonideal_e_scale =
      pin->GetOrAddReal("problem", "nonideal_e_scale", 1.0);
  const Real gamma = pmhd->peos->eos_data.gamma;
  const bool use_ect = pmhd->use_electric_ct;
  const int plane = GetVortexPlane(pin, pmy_mesh_->three_d);
  auto &mbsize = pmbp->pmb->mb_size;
  auto w = pmhd->w0;
  auto u = pmhd->u0;
  auto b = pmhd->b0;
  auto ef = pmhd->e0;
  auto bcc = pmhd->bcc0;

  Kokkos::deep_copy(b.x1f, 0.0);
  Kokkos::deep_copy(b.x2f, 0.0);
  Kokkos::deep_copy(b.x3f, 0.0);
  if (use_ect) {
    Kokkos::deep_copy(ef.x1f, 0.0);
    Kokkos::deep_copy(ef.x2f, 0.0);
    Kokkos::deep_copy(ef.x3f, 0.0);
    par_for("pgen_srr_vortex_e1f", DevExeSpace(), 0, nmb-1, ks, ke,
            js, je, is, ie+1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x1 = LeftEdgeX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                                 mbsize.d_view(m).x1max);
      const Real x2 = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                                   mbsize.d_view(m).x2max);
      const Real x3 = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                                   mbsize.d_view(m).x3max);
      Real e1, e2, e3, b1, b2, b3, u1, u2, u3, pressure, charge;
      ChargedVortexPlaneState(x1, x2, x3, plane, q0, p0, rho, gamma,
                              e1, e2, e3, b1, b2, b3, u1, u2, u3,
                              pressure, charge);
      ef.x1f(m,k,j,i) = nonideal_e_scale*e1;
    });
    par_for("pgen_srr_vortex_e2f", DevExeSpace(), 0, nmb-1, ks, ke,
            js, je+1, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x1 = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                                   mbsize.d_view(m).x1max);
      const Real x2 = LeftEdgeX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                                 mbsize.d_view(m).x2max);
      const Real x3 = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                                   mbsize.d_view(m).x3max);
      Real e1, e2, e3, b1, b2, b3, u1, u2, u3, pressure, charge;
      ChargedVortexPlaneState(x1, x2, x3, plane, q0, p0, rho, gamma,
                              e1, e2, e3, b1, b2, b3, u1, u2, u3,
                              pressure, charge);
      ef.x2f(m,k,j,i) = nonideal_e_scale*e2;
    });
    par_for("pgen_srr_vortex_e3f", DevExeSpace(), 0, nmb-1, ks, ke+1,
            js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x1 = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                                   mbsize.d_view(m).x1max);
      const Real x2 = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                                   mbsize.d_view(m).x2max);
      const Real x3 = LeftEdgeX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                                 mbsize.d_view(m).x3max);
      Real e1, e2, e3, b1, b2, b3, u1, u2, u3, pressure, charge;
      ChargedVortexPlaneState(x1, x2, x3, plane, q0, p0, rho, gamma,
                              e1, e2, e3, b1, b2, b3, u1, u2, u3,
                              pressure, charge);
      ef.x3f(m,k,j,i) = nonideal_e_scale*e3;
    });
  }
  if (plane == static_cast<int>(VortexPlane::xy)) {
    par_for("pgen_srr_charged_vortex_b3", DevExeSpace(), 0, nmb-1, ks, ke+1,
            js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x1 = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                                  mbsize.d_view(m).x1max);
      const Real x2 = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                                  mbsize.d_view(m).x2max);
      Real e1, e2, e3, b1, b2, b3, u1, u2, u3, pressure, charge;
      ChargedVortexPlaneState(x1, x2, 0.0, plane, q0, p0, rho, gamma,
                              e1, e2, e3, b1, b2, b3, u1, u2, u3,
                              pressure, charge);
      b.x3f(m,k,j,i) = b3;
    });
  } else if (plane == static_cast<int>(VortexPlane::yz)) {
    par_for("pgen_srr_charged_vortex_b1", DevExeSpace(), 0, nmb-1, ks, ke,
            js, je, is, ie+1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x2 = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                                  mbsize.d_view(m).x2max);
      const Real x3 = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                                  mbsize.d_view(m).x3max);
      Real e1, e2, e3, b1, b2, b3, u1, u2, u3, pressure, charge;
      ChargedVortexPlaneState(0.0, x2, x3, plane, q0, p0, rho, gamma,
                              e1, e2, e3, b1, b2, b3, u1, u2, u3,
                              pressure, charge);
      b.x1f(m,k,j,i) = b1;
    });
  } else {
    par_for("pgen_srr_charged_vortex_b2", DevExeSpace(), 0, nmb-1, ks, ke,
            js, je+1, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x1 = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                                  mbsize.d_view(m).x1max);
      const Real x3 = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                                  mbsize.d_view(m).x3max);
      Real e1, e2, e3, b1, b2, b3, u1, u2, u3, pressure, charge;
      ChargedVortexPlaneState(x1, 0.0, x3, plane, q0, p0, rho, gamma,
                              e1, e2, e3, b1, b2, b3, u1, u2, u3,
                              pressure, charge);
      b.x2f(m,k,j,i) = b2;
    });
  }

  par_for("pgen_srr_charged_vortex", DevExeSpace(), 0, nmb-1, ks, ke,
          js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, mbsize.d_view(m).x1min,
                                mbsize.d_view(m).x1max);
    const Real x2 = CellCenterX(j-js, indcs.nx2, mbsize.d_view(m).x2min,
                                mbsize.d_view(m).x2max);
    const Real x3 = CellCenterX(k-ks, indcs.nx3, mbsize.d_view(m).x3min,
                                mbsize.d_view(m).x3max);
    Real e1, e2, e3, b1, b2, b3, u1, u2, u3, pressure, charge;
    ChargedVortexPlaneState(x1, x2, x3, plane, q0, p0, rho, gamma,
                            e1, e2, e3, b1, b2, b3, u1, u2, u3,
                            pressure, charge);
    if (use_ect) {
      srrmhd::ElectricFaceToCell(ef, m, k, j, i, e1, e2, e3);
    } else {
      e1 *= nonideal_e_scale;
      e2 *= nonideal_e_scale;
      e3 *= nonideal_e_scale;
    }
    w(m, IDN, k, j, i) = rho;
    w(m, IVX, k, j, i) = u1;
    w(m, IVY, k, j, i) = u2;
    w(m, IVZ, k, j, i) = u3;
    w(m, IEN, k, j, i) = pressure/(gamma - 1.0);
    w(m, srrmhd::IRE1, k, j, i) = e1;
    w(m, srrmhd::IRE2, k, j, i) = e2;
    w(m, srrmhd::IRE3, k, j, i) = e3;
    bcc(m, IBX, k, j, i) = b1;
    bcc(m, IBY, k, j, i) = b2;
    bcc(m, IBZ, k, j, i) = b3;
  });
  pmhd->peos->PrimToCons(w, bcc, u, is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn void SRRMHDChargedVortexErrors()
//! \brief Measure charge, equilibrium, symmetry, and magnetic-divergence errors.

void SRRMHDChargedVortexErrors(ParameterInput *pin, Mesh *pm) {
  auto *pmhd = pm->pmb_pack->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const Real q0 = pin->GetOrAddReal("problem", "q0", 0.7);
  const Real p0 = pin->GetOrAddReal("problem", "p0", 0.1);
  const Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real interior_radius = pin->GetOrAddReal("problem", "interior_radius", 5.0);
  const Real gamma = pmhd->peos->eos_data.gamma;
  const bool three_d = pm->three_d;
  const int plane = GetVortexPlane(pin, three_d);
  const int ea_index = (plane == static_cast<int>(VortexPlane::xy)) ? srrmhd::IRE1
      : ((plane == static_cast<int>(VortexPlane::yz)) ? srrmhd::IRE2
                                                       : srrmhd::IRE3);
  const int eb_index = (plane == static_cast<int>(VortexPlane::xy)) ? srrmhd::IRE2
      : ((plane == static_cast<int>(VortexPlane::yz)) ? srrmhd::IRE3
                                                       : srrmhd::IRE1);
  auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->w0);
  auto bx = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x1f);
  auto by = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x2f);
  auto bz = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->b0.x3f);
  auto exf = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e0.x1f);
  auto eyf = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e0.x2f);
  auto ezf = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->e0.x3f);
  auto mbsize = pm->pmb_pack->pmb->mb_size.h_view;

  Real l1_q = 0.0, linf_q = 0.0;
  Real l1_p = 0.0, linf_p = 0.0;
  Real charge_integral = 0.0, boundary_flux = 0.0;
  Real max_divb = 0.0, max_symmetry = 0.0;
  int ninterior = 0;
  const int nx1_global = pm->mesh_indcs.nx1;
  const int nx2_global = pm->mesh_indcs.nx2;
  const int nx3_global = pm->mesh_indcs.nx3;
  const int ncells_global = nx1_global*nx2_global*nx3_global;
  std::vector<Real> parity_data(3*ncells_global, 0.0);
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    const Real dx = mbsize(m).dx1;
    const Real dy = mbsize(m).dx2;
    const Real dz = three_d ? mbsize(m).dx3 : 1.0;
    const Real idx1 = 1.0/dx;
    const Real idx2 = 1.0/dy;
    const Real idx3 = three_d ? 1.0/dz : 0.0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real x1 = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                      mbsize(m).x1max);
          const Real x2 = CellCenterX(j-js, indcs.nx2, mbsize(m).x2min,
                                      mbsize(m).x2max);
          const Real x3 = CellCenterX(k-ks, indcs.nx3, mbsize(m).x3min,
                                      mbsize(m).x3max);
          int gi = static_cast<int>((x1 - pm->mesh_size.x1min)/pm->mesh_size.dx1);
          int gj = static_cast<int>((x2 - pm->mesh_size.x2min)/pm->mesh_size.dx2);
          int gk = three_d
              ? static_cast<int>((x3 - pm->mesh_size.x3min)/pm->mesh_size.dx3) : 0;
          gi = std::max(0, std::min(nx1_global-1, gi));
          gj = std::max(0, std::min(nx2_global-1, gj));
          gk = std::max(0, std::min(nx3_global-1, gk));
          const int global_index = (gk*nx2_global + gj)*nx1_global + gi;
          parity_data[global_index] = w(m, ea_index, k, j, i);
          parity_data[ncells_global + global_index] = w(m, eb_index, k, j, i);
          parity_data[2*ncells_global + global_index] = w(m, IEN, k, j, i);
          Real e10, e20, e30, b10, b20, b30, u10, u20, u30, pressure0, charge0;
          ChargedVortexPlaneState(x1, x2, x3, plane, q0, p0, rho, gamma,
                                  e10, e20, e30, b10, b20, b30, u10, u20, u30,
                                  pressure0, charge0);
          Real charge = srrmhd::CellCenteredCharge(
              w, m, k, j, i, idx1, idx2, idx3, true, three_d);
          if (pmhd->use_electric_ct) {
            charge = (exf(m,k,j,i+1)-exf(m,k,j,i))*idx1
                +(eyf(m,k,j+1,i)-eyf(m,k,j,i))*idx2;
            if (three_d) {
              charge += (ezf(m,k+1,j,i)-ezf(m,k,j,i))*idx3;
            }
          }
          charge_integral += charge*dx*dy*dz;

          Real a, b;
          bool away_from_plane_boundary;
          if (plane == static_cast<int>(VortexPlane::xy)) {
            a = x1;
            b = x2;
            away_from_plane_boundary =
                (gi > 0 && gi < nx1_global-1 && gj > 0 && gj < nx2_global-1);
          } else if (plane == static_cast<int>(VortexPlane::yz)) {
            a = x2;
            b = x3;
            away_from_plane_boundary =
                (gj > 0 && gj < nx2_global-1 && gk > 0 && gk < nx3_global-1);
          } else {
            a = x3;
            b = x1;
            away_from_plane_boundary =
                (gk > 0 && gk < nx3_global-1 && gi > 0 && gi < nx1_global-1);
          }
          if (away_from_plane_boundary &&
              a*a + b*b < interior_radius*interior_radius) {
            const Real q_error = fabs(charge - charge0);
            const Real pressure = (gamma - 1.0)*w(m, IEN, k, j, i);
            const Real p_error = fabs(pressure - pressure0);
            l1_q += q_error;
            l1_p += p_error;
            linf_q = std::max(linf_q, q_error);
            linf_p = std::max(linf_p, p_error);
            ++ninterior;
          }
          Real divb = (bx(m,k,j,i+1) - bx(m,k,j,i))/dx
              +(by(m,k,j+1,i) - by(m,k,j,i))/dy;
          if (three_d) divb += (bz(m,k+1,j,i) - bz(m,k,j,i))/dz;
          max_divb = std::max(max_divb, fabs(divb));
        }
      }
    }
    if (pmhd->use_electric_ct) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          boundary_flux += dy*dz*(exf(m,k,j,ie+1)-exf(m,k,j,is));
        }
      }
      for (int k = ks; k <= ke; ++k) {
        for (int i = is; i <= ie; ++i) {
          boundary_flux += dx*dz*(eyf(m,k,je+1,i)-eyf(m,k,js,i));
        }
      }
      if (three_d) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            boundary_flux += dx*dy*(ezf(m,ke+1,j,i)-ezf(m,ks,j,i));
          }
        }
      }
    } else {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          boundary_flux += 0.5*dy*dz*(w(m,srrmhd::IRE1,k,j,ie+1)
                                      +w(m,srrmhd::IRE1,k,j,ie));
          boundary_flux -= 0.5*dy*dz*(w(m,srrmhd::IRE1,k,j,is)
                                      +w(m,srrmhd::IRE1,k,j,is-1));
        }
      }
      for (int k = ks; k <= ke; ++k) {
        for (int i = is; i <= ie; ++i) {
          boundary_flux += 0.5*dx*dz*(w(m,srrmhd::IRE2,k,je+1,i)
                                      +w(m,srrmhd::IRE2,k,je,i));
          boundary_flux -= 0.5*dx*dz*(w(m,srrmhd::IRE2,k,js,i)
                                      +w(m,srrmhd::IRE2,k,js-1,i));
        }
      }
    }
    if (three_d && !(pmhd->use_electric_ct)) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          boundary_flux += 0.5*dx*dy*(w(m,srrmhd::IRE3,ke+1,j,i)
                                      +w(m,srrmhd::IRE3,ke,j,i));
          boundary_flux -= 0.5*dx*dy*(w(m,srrmhd::IRE3,ks,j,i)
                                      +w(m,srrmhd::IRE3,ks-1,j,i));
        }
      }
    }
  }

  Real sum_diagnostics[4] = {l1_q, l1_p, charge_integral, boundary_flux};
  Real max_diagnostics[3] = {linf_q, linf_p, max_divb};
  int failures = pm->ecounter.neos_fail;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sum_diagnostics, 4, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, max_diagnostics, 3, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ninterior, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, parity_data.data(), 3*ncells_global,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  l1_q = sum_diagnostics[0];
  l1_p = sum_diagnostics[1];
  charge_integral = sum_diagnostics[2];
  boundary_flux = sum_diagnostics[3];
  linf_q = max_diagnostics[0];
  linf_p = max_diagnostics[1];
  max_divb = max_diagnostics[2];
  l1_q /= ninterior;
  l1_p /= ninterior;
  const Real gauss_residual = fabs(charge_integral - boundary_flux);

  // Compute half-turn parity against global opposite cells.  A local-index reflection
  // is incorrect once an active plane spans more than one MeshBlock.
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real x1 = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                      mbsize(m).x1max);
          const Real x2 = CellCenterX(j-js, indcs.nx2, mbsize(m).x2min,
                                      mbsize(m).x2max);
          const Real x3 = CellCenterX(k-ks, indcs.nx3, mbsize(m).x3min,
                                      mbsize(m).x3max);
          int gi = static_cast<int>((x1 - pm->mesh_size.x1min)/pm->mesh_size.dx1);
          int gj = static_cast<int>((x2 - pm->mesh_size.x2min)/pm->mesh_size.dx2);
          int gk = three_d
              ? static_cast<int>((x3 - pm->mesh_size.x3min)/pm->mesh_size.dx3) : 0;
          gi = std::max(0, std::min(nx1_global-1, gi));
          gj = std::max(0, std::min(nx2_global-1, gj));
          gk = std::max(0, std::min(nx3_global-1, gk));
          int giopp = gi, gjopp = gj, gkopp = gk;
          if (plane == static_cast<int>(VortexPlane::xy)) {
            giopp = nx1_global - 1 - gi;
            gjopp = nx2_global - 1 - gj;
          } else if (plane == static_cast<int>(VortexPlane::yz)) {
            gjopp = nx2_global - 1 - gj;
            gkopp = nx3_global - 1 - gk;
          } else {
            gkopp = nx3_global - 1 - gk;
            giopp = nx1_global - 1 - gi;
          }
          const int opposite = (gkopp*nx2_global + gjopp)*nx1_global + giopp;
          max_symmetry = std::max(max_symmetry,
              fabs(w(m,ea_index,k,j,i) + parity_data[opposite]));
          max_symmetry = std::max(max_symmetry,
              fabs(w(m,eb_index,k,j,i) + parity_data[ncells_global + opposite]));
          max_symmetry = std::max(max_symmetry,
              fabs(w(m,IEN,k,j,i) - parity_data[2*ncells_global + opposite]));
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &max_symmetry, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif

  if (global_variable::my_rank == 0) {
    const std::string basename = pin->GetString("job", "basename");
    const std::string error_name = basename + "-errs.dat";
    std::ifstream old_error_file(error_name);
    const bool error_file_exists = old_error_file.good();
    old_error_file.close();
    std::ofstream file(error_name, std::ios::app);
    if (!error_file_exists) {
      file << "# Nx Ncycle L1_q Linf_q L1_p Linf_p Gauss max_divB max_symmetry "
           << "recovery_failures time\n";
    }
    const int resolution = (plane == static_cast<int>(VortexPlane::xy)) ? nx1_global
        : ((plane == static_cast<int>(VortexPlane::yz)) ? nx2_global : nx3_global);
    file << std::setprecision(17) << resolution << " " << pm->ncycle << " "
         << l1_q << " " << linf_q << " " << l1_p << " " << linf_p << " "
         << gauss_residual << " " << max_divb << " " << max_symmetry << " "
         << failures << " " << pm->time << std::endl;
  }

  if (pin->DoesParameterExist("problem", "viscous_state_name")) {
    const std::string name = pin->GetString("problem", "viscous_state_name");
    if (name.compare("none") != 0) {
      auto visc_u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->visc_u0);
      std::string filename = name;
      if (global_variable::nranks > 1) {
        filename += "-rank" + std::to_string(global_variable::my_rank);
      }
      std::ofstream state_file(filename + "-state.dat");
      state_file << "# x1 x2 x3 rho eint u1 u2 u3 E1 E2 E3 "
                 << "pi11 pi22 pi33 pi12 pi13 pi23\n" << std::setprecision(17);
      for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) {
              const Real x1 = CellCenterX(i-is, indcs.nx1, mbsize(m).x1min,
                                          mbsize(m).x1max);
              const Real x2 = CellCenterX(j-js, indcs.nx2, mbsize(m).x2min,
                                          mbsize(m).x2max);
              const Real x3 = CellCenterX(k-ks, indcs.nx3, mbsize(m).x3min,
                                          mbsize(m).x3max);
              const Real lor = sqrt(1.0 + SQR(w(m, IVX, k, j, i))
                  + SQR(w(m, IVY, k, j, i)) + SQR(w(m, IVZ, k, j, i)));
              const Real d = w(m, IDN, k, j, i)*lor;
              state_file << x1 << " " << x2 << " " << x3 << " "
                         << w(m, IDN, k, j, i) << " " << w(m, IEN, k, j, i)
                         << " " << w(m, IVX, k, j, i) << " "
                         << w(m, IVY, k, j, i) << " " << w(m, IVZ, k, j, i)
                         << " " << w(m, srrmhd::IRE1, k, j, i) << " "
                         << w(m, srrmhd::IRE2, k, j, i) << " "
                         << w(m, srrmhd::IRE3, k, j, i);
              for (int n = 0; n < srrmhd::NVISC; ++n) {
                state_file << " " << visc_u(m, n, k, j, i)/d;
              }
              state_file << "\n";
            }
          }
        }
      }
    }
  }

  if (pmhd->resistivity_data.model != srrmhd::ResistivityModel::uniform) {
    Real eta_min = std::numeric_limits<Real>::max();
    Real eta_max = 0.0;
    if (pmhd->use_electric_ct) {
      auto eta1 =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->eta_face.x1f);
      auto eta2 =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->eta_face.x2f);
      auto eta3 =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->eta_face.x3f);
      for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie + 1; ++i) {
              eta_min = std::min(eta_min, eta1(m, k, j, i));
              eta_max = std::max(eta_max, eta1(m, k, j, i));
            }
          }
        }
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je + 1; ++j) {
            for (int i = is; i <= ie; ++i) {
              eta_min = std::min(eta_min, eta2(m, k, j, i));
              eta_max = std::max(eta_max, eta2(m, k, j, i));
            }
          }
        }
        const int e3ke = three_d ? ke + 1 : ks;
        for (int k = ks; k <= e3ke; ++k) {
          for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) {
              eta_min = std::min(eta_min, eta3(m, k, j, i));
              eta_max = std::max(eta_max, eta3(m, k, j, i));
            }
          }
        }
      }
    } else {
      auto eta = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmhd->eta_cell);
      for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) {
              eta_min = std::min(eta_min, eta(m, k, j, i));
              eta_max = std::max(eta_max, eta(m, k, j, i));
            }
          }
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &eta_min, 1, MPI_ATHENA_REAL, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &eta_max, 1, MPI_ATHENA_REAL, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      const std::string basename = pin->GetString("job", "basename");
      std::ofstream eta_file(basename + "-eta.dat");
      eta_file << std::setprecision(17) << eta_min << " " << eta_max << std::endl;
    }
  }
}
