#ifndef OUTPUTS_POWER_SPECTRUM_HPP_
#define OUTPUTS_POWER_SPECTRUM_HPP_

#include <memory>

#include <Kokkos_Core.hpp>

#include "mesh/mesh.hpp"
#include "outputs.hpp"
#include "parameter_input.hpp"
#include "power_spectrum_backend.hpp"

class PowerSpectrumOutput : public BaseTypeOutput {
 public:
  PowerSpectrumOutput(ParameterInput *pin, Mesh *pm, OutputParameters &opar);

  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
  // Peak curl shell for history_curl_peak, in inverse code length. Evaluated at this
  // output's own cadence: hst rows between spectrum outputs repeat the latest value
  // (piecewise constant); the hst output is ordered after curl-peak spectra so rows
  // in a spectrum cycle use that cycle's spectrum. Computes a spectrum only if none
  // exists yet.
  Real CurlPeak(Mesh *pm);

 private:
  Kokkos::View<Real*> spectrum_;
  Kokkos::View<Real*> curl_spectrum_;
  Kokkos::View<Real*> helicity_spectrum_;  // optional, key helicity_spectrum = true
  std::unique_ptr<PowerSpectrumBackend> backend_;
  int nbins_ = 0;
  int loaded_cycle_ = -1;
  Real loaded_time_ = -1.0;
};

#endif  // OUTPUTS_POWER_SPECTRUM_HPP_
