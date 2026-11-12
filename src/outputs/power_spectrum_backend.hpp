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
  // Empty curl/helicity views (extent 0) disable those optional shell sums.
  virtual void Compute(Mesh *pm, const OutputParameters &out_params,
                       Kokkos::View<Real*> spectrum,
                       Kokkos::View<Real*> curl_spectrum,
                       Kokkos::View<Real*> helicity_spectrum) = 0;
};

std::unique_ptr<PowerSpectrumBackend> BuildPowerSpectrumBackend(
    Mesh *pm, const OutputParameters &out_params, bool want_helicity = false,
    bool allow_device = true);

#endif  // OUTPUTS_POWER_SPECTRUM_BACKEND_HPP_
