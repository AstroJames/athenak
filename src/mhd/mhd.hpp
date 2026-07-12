#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.hpp
//  \brief definitions for MHD class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "mhd/resistivity_model.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Viscosity;
class Resistivity;
class BiermannBattery;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class OrbitalAdvectionFC;
class ShearingBoxBoundaryCC;
class ShearingBoxBoundaryFC;
class Driver;

// function ptr for user-defined MHD boundary functions enrolled in problem generator
namespace mhd {
using MHDBoundaryFnPtr = void (*)(int m, Mesh* pm, MHD* pmhd, DvceArray5D<Real> &u);
}

// constants that enumerate MHD Riemann Solver options
enum class MHD_RSolver {advect, llf, hlle, hlld, roe,   // non-relativistic
                        llf_sr, hlle_sr, llf_srr,       // ideal SR and resistive SR
                        llf_gr, hlle_gr};                       // GR

// Progress states for the re-entrant multidimensional dual-CT implicit task.
// Communication
// completion is polled on successive task-list passes rather than on the host.
enum class ECTImplicitState {idle, lagged_cell_recv, star_face_recv, picard_compute,
                             picard_face_recv, picard_cell_recv, picard_reduce,
                             final_average_recv, final_face_recv};

//----------------------------------------------------------------------------------------
//! \struct MHDTaskIDs
//  \brief container to hold TaskIDs of all mhd tasks

struct MHDTaskIDs {
  TaskID savest;
  TaskID irecv;
  TaskID copyu;
  TaskID impl;
  TaskID impl_bcs;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID ectprep;
  TaskID sendbedge;
  TaskID recvbedge;
  TaskID ect;
  TaskID srctrms;
  TaskID sendu_oa;
  TaskID recvu_oa;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID sendu_shr;
  TaskID recvu_shr;
  TaskID efld;
  TaskID efldsrc;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID sendb_oa;
  TaskID recvb_oa;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID sendb_shr;
  TaskID recvb_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD {
 public:
  MHD(MeshBlockPack *ppack, ParameterInput *pin);
  ~MHD();

  // data
  ReconstructionMethod recon_method;
  MHD_RSolver rsolver_method;
  EquationOfState *peos;   // chosen EOS

  int nmhd;                // number of mhd variables (8/5/4 for resistive/ideal/iso)
  int nscalars;            // number of passive scalars
  bool is_resistive_rel = false;  // true for full resistive SRMHD with evolved E
  bool use_electric_ct = false;   // true for charge-conserving face-centered E prototype
  Real resistivity = 0.0;         // scalar relativistic resistivity eta
  srrmhd::ResistivityData resistivity_data;
  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  DvceFaceFld4D<Real> b0;  // face-centered magnetic fields
  DvceFaceFld4D<Real> e0;  // primary face-centered electric field for dual CT
  DvceArray5D<Real> bcc0;  // cell-centered magnetic fields
  DvceArray4D<Real> eta_cell;     // frozen cell-local eta for one IMEX stage
  DvceFaceFld4D<Real> eta_face;   // frozen face-local eta for one IMEX stage

  DvceArray5D<Real> coarse_u0;    // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;    // primitive variables on 2x coarser grid (for SMR/AMR)
  DvceFaceFld4D<Real> coarse_b0;  // face-centered B-field on 2x coarser grid
  DvceFaceFld4D<Real> coarse_e0;  // face-centered E-field on 2x coarser grid

  // Objects containing boundary communication buffers and routines for u and b
  MeshBoundaryValuesCC *pbval_u;
  MeshBoundaryValuesFC *pbval_b;
  MeshBoundaryValuesFC *pbval_e = nullptr;         // explicit edge-B synchronization
  MeshBoundaryValuesFC *pbval_ect_face = nullptr;  // implicit face/star exchange
  MeshBoundaryValuesCC *pbval_ect_u = nullptr;     // implicit Picard-state exchange
  MHDBoundaryFnPtr MHDBoundaryFunc[6];

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  OrbitalAdvectionFC *porb_b = nullptr;
  ShearingBoxBoundaryCC *psbox_u = nullptr;
  ShearingBoxBoundaryFC *psbox_b = nullptr;

  // Object(s) for extra physics (viscosity, resistivity, Biermann, conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Resistivity *presist = nullptr;
  BiermannBattery *pbier = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables, second register
  DvceFaceFld4D<Real> b1;     // face-centered magnetic fields, second register
  DvceFaceFld4D<Real> e1;     // face-centered electric fields, second register
  DvceFaceFld4D<Real> jfc;    // face-centered current used by dual CT
  DvceFaceFld4D<Real> estar;  // face-centered right-hand side of an implicit stage
  DvceFaceFld4D<Real> ect_face_prev;  // local face values before interface averaging
  DvceFaceFld5D<Real> ect_src;  // face-centered conductive IMEX source history
  DvceArray5D<Real> ect_u_prev; // previous four-velocity for face/cell iteration
  ECTImplicitState ect_impl_state = ECTImplicitState::idle;
  int ect_impl_estage = -1;
  int ect_leading_stage = -2;
  int ect_picard_iteration = 0;
  int ect_comm_phase = 0;
  Real ect_local_residual = 0.0;
  Real ect_global_residual = 0.0;
#if MPI_PARALLEL_ENABLED
  MPI_Request ect_reduce_request = MPI_REQUEST_NULL;
#endif
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  DvceEdgeFld4D<Real> bfld;   // edge-centered magnetic fields (fluxes of E)
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  Real dtnew;

  // following used for time derivatives in computation of jcon
  bool wbcc_saved = false;
  DvceArray5D<Real> wsaved;
  DvceArray5D<Real> bccsaved;

  // following used for FOFC algorithm
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC

  // following used for h-correction (Sanders, Morano & Druguet 1998)
  DvceArray4D<Real> eta1, eta2, eta3;  // max |eigenvalue| in x1, x2, x3 per cell
  bool use_hcorr = false;              // flag to enable h-correction

  // container to hold names of TaskIDs
  MHDTaskIDs id;

  // functions...
  void SetSaveWBcc();
  void AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_timeintegrator" task list
  TaskStatus SaveMHDState(Driver *d, int stage);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus FirstTwoImpRK(Driver *d, int stage);
  TaskStatus ImpRKUpdate(Driver *d, int stage);
  TaskStatus FaceImpRKUpdate(Driver *d, int stage);
  void FreezeFaceResistivity();
  TaskStatus ExchangeElectricFaces(DvceFaceFld4D<Real> &e);
  TaskStatus StartElectricFaceExchange(DvceFaceFld4D<Real> &e);
  TaskStatus FinishElectricFaceExchange(DvceFaceFld4D<Real> &e);
  TaskStatus StartSharedElectricAverage(DvceFaceFld4D<Real> &e);
  TaskStatus FinishSharedElectricAverage(DvceFaceFld4D<Real> &e);
  TaskStatus StartElectricCellExchange();
  TaskStatus FinishElectricCellExchange();
  TaskStatus SendElectricFaces(Driver *d, int stage);
  TaskStatus RecvElectricFaces(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus DualCTPrepare(Driver *d, int stage);
  TaskStatus SendBEdge(Driver *d, int stage);
  TaskStatus RecvBEdge(Driver *d, int stage);
  TaskStatus DualCTUpdate(Driver *d, int stage);
  TaskStatus MHDSrcTerms(Driver *d, int stage);
  void AddResistiveChargeSource(const Real beta_dt);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus CornerE(Driver *d, int stage);
  TaskStatus EFieldSrc(Driver *d, int stage);
  TaskStatus SendE(Driver *d, int stage);
  TaskStatus RecvE(Driver *d, int stage);
  TaskStatus CT(Driver *d, int stage);
  TaskStatus SendB_OA(Driver *d, int stage);
  TaskStatus RecvB_OA(Driver *d, int stage);
  TaskStatus RestrictB(Driver *d, int stage);
  TaskStatus SendB(Driver *d, int stage);
  TaskStatus RecvB(Driver *d, int stage);
  TaskStatus SendB_Shr(Driver *d, int stage);
  TaskStatus RecvB_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <MHD_RSolver T>
  void CalculateFluxes(Driver *d, int stage);

  // first-order flux correction
  void FOFC(Driver *d, int stage);

  DvceArray5D<Real> utest, bcctest;  // scratch arrays for FOFC

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
