//=== power_spectrum.cpp ===================================================
//  On-the-fly isotropic power spectrum output
//===========================================================================

#include "power_spectrum.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

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
  if (!op.history_curl_peak.empty()) {
    auto &size = pm->mesh_size;
    Real lx = size.x1max-size.x1min, ly = size.x2max-size.x2min;
    Real lz = size.x3max-size.x3min;
    if (op.history_curl_peak.size() > 10 || !pm->strictly_periodic || !pm->three_d ||
        std::abs(lx-ly) > 1.0e-12*lx || std::abs(lx-lz) > 1.0e-12*lx) {
      std::cerr << "history_curl_peak requires a label of at most 10 characters "
                   "and a periodic cubic 3-D domain.\n";
      std::exit(EXIT_FAILURE);
    }
    curl_spectrum_ = Kokkos::View<Real*>("curl_spectrum", nbins_);
  }
}

void PowerSpectrumOutput::LoadOutputData(Mesh *pm) {
  if (loaded_cycle_ == pm->ncycle && loaded_time_ == pm->time) return;
  Kokkos::deep_copy(spectrum_, Real(0));
  if (curl_spectrum_.extent(0) > 0) Kokkos::deep_copy(curl_spectrum_, Real(0));
  backend_->Compute(pm, out_params, spectrum_, curl_spectrum_);
  loaded_cycle_ = pm->ncycle;
  loaded_time_ = pm->time;
}

Real PowerSpectrumOutput::CurlPeak(Mesh *pm) {
  LoadOutputData(pm);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), curl_spectrum_);
  Real maximum = 0.0;
  int peak = 0;  // zero placeholder when all curl power vanishes
  for (int n = 0; n < nbins_; ++n) {
    if (host(n) > maximum) {
      maximum = host(n);
      peak = n+1;  // first (lowest) shell wins exact ties
    }
  }
  return peak*(2.0*std::acos(-1.0))/(pm->mesh_size.x1max-pm->mesh_size.x1min);
}

void PowerSpectrumOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  int rank = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  if (rank == 0) {
    auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), spectrum_);
    auto curl_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), curl_spectrum_);

    std::ostringstream number;
    number << std::setw(5) << std::setfill('0') << out_params.file_number;
    std::string fname = out_params.file_basename + "." + out_params.file_id +
                        "." + number.str() + ".spec";

    std::ofstream ofs(fname);
    ofs << std::scientific << std::setprecision(17);
    ofs << "# time=" << pm->time << " cycle=" << pm->ncycle << '\n'
        << "# shell_sum |FFT(field)/Ncells|^2, summed over vector components; no 1/2\n"
        << "# integer shell s <= |n| < s+1; zero mode excluded; s=1.." << nbins_
        << "; corners beyond the last shell omitted\n"
        << "# Cubic box: k=2*pi*|n|/L; velocity is unweighted (not kinetic energy).\n";
    if (curl_spectrum_.extent(0) > 0) {
      ofs << "# columns: shell field_power curl_power; curl=i*k x FFT(field), "
             "k in inverse code length; Nyquist derivatives set to zero.\n";
    }
    for (int s = 1; s <= nbins_; ++s) {
      ofs << s << ' ' << host(s - 1);
      if (curl_spectrum_.extent(0) > 0) ofs << ' ' << curl_host(s - 1);
      ofs << '\n';
    }
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
