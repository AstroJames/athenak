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
  Real CurlPeak(Mesh *pm);

 private:
  Kokkos::View<Real*> spectrum_;
  Kokkos::View<Real*> curl_spectrum_;
  std::unique_ptr<PowerSpectrumBackend> backend_;
  int nbins_ = 0;
  int loaded_cycle_ = -1;
  Real loaded_time_ = -1.0;
};

#endif  // OUTPUTS_POWER_SPECTRUM_HPP_
