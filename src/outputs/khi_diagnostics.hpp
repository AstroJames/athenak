#ifndef OUTPUTS_KHI_DIAGNOSTICS_HPP_
#define OUTPUTS_KHI_DIAGNOSTICS_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file khi_diagnostics.hpp
//! \brief Newtonian KHI cell diagnostics and distributed planar reductions.

#include <vector>
#include "athena.hpp"
#include "outputs.hpp"

namespace khi {
constexpr int nvar = 45;
// Shared by the histories, profiles and slices; all derivatives are cell-centered.
// du[a][b] = partial_b u^a. Magnetic permeability is unity.
KOKKOS_INLINE_FUNCTION
void Cell(const DvceArray5D<Real> &w, const DvceArray5D<Real> &u,
          const DvceArray5D<Real> &b, int m, int k, int j, int i,
          Real dx, Real dy, Real dz, Real nu, Real eta, Real gm1, Real *q) {
  for (int n = 0; n < nvar; ++n) q[n] = 0.0;
  Real v[3], mag[3], du[3][3], db[3][3];
  for (int a = 0; a < 3; ++a) {
    v[a] = w(m,IVX+a,k,j,i);
    mag[a] = b(m,a,k,j,i);
    du[a][0] = (w(m,IVX+a,k,j,i+1)-w(m,IVX+a,k,j,i-1))/(2*dx);
    du[a][1] = (w(m,IVX+a,k,j+1,i)-w(m,IVX+a,k,j-1,i))/(2*dy);
    du[a][2] = (w(m,IVX+a,k+1,j,i)-w(m,IVX+a,k-1,j,i))/(2*dz);
    db[a][0] = (b(m,a,k,j,i+1)-b(m,a,k,j,i-1))/(2*dx);
    db[a][1] = (b(m,a,k,j+1,i)-b(m,a,k,j-1,i))/(2*dy);
    db[a][2] = (b(m,a,k+1,j,i)-b(m,a,k-1,j,i))/(2*dz);
  }
  Real omega[3] = {du[2][1]-du[1][2], du[0][2]-du[2][0], du[1][0]-du[0][1]};
  Real current[3] = {db[2][1]-db[1][2], db[0][2]-db[2][0], db[1][0]-db[0][1]};
  Real rho = u(m,IDN,k,j,i), ke = 0.0, me = 0.0;
  q[0] = 1.0;
  q[1] = rho;
  q[5] = u(m,IEN,k,j,i);
  q[43] = du[0][0]+du[1][1]+du[2][2];
  for (int a = 0; a < 3; ++a) {
    q[2+a] = u(m,IM1+a,k,j,i);
    q[7+a] = v[a];
    q[10+a] = mag[a];
    q[13+a] = SQR(v[a]);
    q[16+a] = SQR(mag[a]);
    q[19+a] = 0.5*rho*SQR(v[a]);
    ke += q[19+a];
    me += 0.5*q[16+a];
    q[28+a] = omega[a];
    q[31+a] = current[a];
    q[34] += SQR(omega[a]);
    q[35] += SQR(current[a]);
    q[36] += v[a]*omega[a];
    q[37] += mag[a]*current[a];
    q[38] += v[a]*mag[a];
    for (int c = 0; c < 3; ++c) {
      q[39] += mag[a]*mag[c]*du[a][c];
      Real strain = 0.5*(du[a][c]+du[c][a]) - ((a == c) ? q[43]/3.0 : 0.0);
      q[42] += 2.0*rho*nu*SQR(strain);
    }
  }
  q[6] = q[5]-ke-me;
  q[22] = v[1]*mag[2]; q[23] = v[2]*mag[1];
  q[24] = v[2]*mag[0]; q[25] = v[0]*mag[2];
  q[26] = v[0]*mag[1]; q[27] = v[1]*mag[0];
  q[40] = -me*q[43];  // compression contribution to the volume magnetic-energy budget
  q[41] = eta*q[35];
  q[44] = gm1*q[6];
}
}  // namespace khi

class KhiDiagnosticsOutput : public BaseTypeOutput {
 public:
  KhiDiagnosticsOutput(ParameterInput *pin, Mesh *pm, OutputParameters op);
  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;

 private:
  bool profiles_;
  Real nu_, eta_, gm1_;
  int cells_[3], offset_[3], nplanes_;
  Real lower_[3], spacing_[3];
  std::vector<Real> sums_;
  std::vector<Real> Plane(int region, int axis, int bin) const;
};
#endif  // OUTPUTS_KHI_DIAGNOSTICS_HPP_
