#ifndef OUTPUTS_POWER_SPECTRUM_BACKEND_HPP_
#define OUTPUTS_POWER_SPECTRUM_BACKEND_HPP_

#include <memory>

#include <Kokkos_Core.hpp>

#include "mesh/mesh.hpp"
#include "outputs.hpp"

class PowerSpectrumBackend {
 public:
  virtual ~PowerSpectrumBackend() = default;
  virtual int GetNumBins() const = 0;
  virtual void Compute(Mesh *pm, const OutputParameters &out_params,
                       Kokkos::View<Real*> spectrum) = 0;
};

std::unique_ptr<PowerSpectrumBackend> BuildPowerSpectrumBackend(
    Mesh *pm, const OutputParameters &out_params);

#endif  // OUTPUTS_POWER_SPECTRUM_BACKEND_HPP_
