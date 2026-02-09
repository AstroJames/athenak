//=== power_spectrum.cpp ===================================================
//  On-the-fly isotropic power spectrum output
//===========================================================================

#include "power_spectrum.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

PowerSpectrumOutput::PowerSpectrumOutput(ParameterInput *pin,
                                         Mesh *pm,
                                         OutputParameters &op)
    : BaseTypeOutput(pin, pm, op) {
  backend_ = BuildPowerSpectrumBackend(pm, out_params);
  nbins_ = backend_->GetNumBins();
  spectrum_ = Kokkos::View<Real*>("spectrum", nbins_);
}

void PowerSpectrumOutput::LoadOutputData(Mesh *pm) {
  Kokkos::deep_copy(spectrum_, Real(0));
  backend_->Compute(pm, out_params, spectrum_);
}

void PowerSpectrumOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
#if MPI_PARALLEL_ENABLED
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;
#endif

  auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), spectrum_);

  std::ostringstream number;
  number << std::setw(5) << std::setfill('0') << out_params.file_number;
  std::string fname = out_params.file_basename + "." + out_params.file_id +
                      "." + number.str() + ".spec";

  std::ofstream ofs(fname);
  ofs << std::scientific << std::setprecision(6);
  for (int s = 1; s <= nbins_; ++s) {
    ofs << s << ' ' << host(s - 1) << '\n';
  }

  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
