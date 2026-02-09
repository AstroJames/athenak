#ifndef OUTPUTS_POWER_SPECTRUM_HPP_
#define OUTPUTS_POWER_SPECTRUM_HPP_

#include "outputs.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <Kokkos_Complex.hpp>
#include <memory>

// ── view & plan aliases ────────────────────────────────────────────────────
using real_view_t =
    Kokkos::View<Real***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using complex_view_t =
    Kokkos::View<Kokkos::complex<Real>***, Kokkos::LayoutRight,
                 Kokkos::DefaultExecutionSpace>;

using PlanType =
    KokkosFFT::Plan<Kokkos::DefaultExecutionSpace,
                    real_view_t,
                    complex_view_t,
                    3>; // 3-D r2c

// ── PowerSpectrumOutput class ──────────────────────────────────────────────
class PowerSpectrumOutput : public BaseTypeOutput {
public:
  PowerSpectrumOutput(ParameterInput *pin, Mesh *pm, OutputParameters &opar);

  void LoadOutputData (Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;

private:
  // FFT buffers and spectrum
  real_view_t         fft_in_;   // real-valued input cube
  complex_view_t      fft_out_;  // complex output cube
  Kokkos::View<Real*> spectrum_; // shell-integrated power

  std::unique_ptr<PlanType> plan_; // non-copyable FFT plan

  // geometry & bookkeeping
  int  nbins_ = 0;
  int  nx_ = 0, ny_ = 0, nz_ = 0;
  Real Lx_ = 1.0, Ly_ = 1.0, Lz_ = 1.0;
};

#endif  // OUTPUTS_POWER_SPECTRUM_HPP_