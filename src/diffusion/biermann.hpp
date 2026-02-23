#ifndef DIFFUSION_BIERMANN_HPP_
#define DIFFUSION_BIERMANN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file biermann.hpp
//  \brief Contains data and functions implementing the Biermann battery term in Ohm's
//  law for MHD.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"

struct EOS_Data;

//----------------------------------------------------------------------------------------
//! \class BiermannBattery
//  \brief data and functions that implement Biermann battery physics

class BiermannBattery {
 public:
  BiermannBattery(MeshBlockPack *pp, ParameterInput *pin);
  ~BiermannBattery();

  // data
  Real coeff;
  bool enabled;

  // function to add Biermann E-field term to edge-centered electric fields
  void BiermannEField(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                      DvceEdgeFld4D<Real> &efld);

 private:
  MeshBlockPack* pmy_pack;

  // coefficient options
  bool coeff_from_closure;
  Real pe_fraction;
  Real mu_e;
};

#endif // DIFFUSION_BIERMANN_HPP_
