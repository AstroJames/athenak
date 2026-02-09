//=== power_spectrum.cpp ===================================================
//  On-the-fly isotropic velocity-power spectrum (dk = 1)
//===========================================================================

#include "power_spectrum.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------
PowerSpectrumOutput::PowerSpectrumOutput(ParameterInput *pin,
                                         Mesh          *pm,
                                         OutputParameters &op)
  : BaseTypeOutput(pin, pm, op)
{
  // --- global zone counts ------------------------------------------------
  const auto &g = pm->mesh_indcs;  // nx1,nx2,nx3 (global)
  nx_ = g.nx1;
  ny_ = g.nx2;
  nz_ = g.nx3;

  // --- integer shells up to scalar Nyquist -------------------------------
  nbins_ = std::min({nx_/2, ny_/2, nz_/2});

  spectrum_ = Kokkos::View<Real*>("spectrum", nbins_);

  if (pm->multilevel) {
    std::cout << "### FATAL ERROR in PowerSpectrumOutput::PowerSpectrumOutput\n"
              << "power_spectrum currently supports only single-level meshes.\n";
    std::exit(EXIT_FAILURE);
  }

  // --- FFT buffers & plan ------------------------------------------------
#if MPI_PARALLEL_ENABLED
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
#endif
    fft_in_ = real_view_t("fft_in", nx_, ny_, nz_);
    fft_out_ = complex_view_t("fft_out", nx_, ny_, nz_/2 + 1);

    plan_ = std::make_unique<PlanType>(
        Kokkos::DefaultExecutionSpace{}, fft_in_, fft_out_,
        KokkosFFT::Direction::forward, std::array<int,3>{0,1,2});
#if MPI_PARALLEL_ENABLED
  }
#endif
}

//------------------------------------------------------------------------
// LoadOutputData: gather vx,vy,vz ➜ FFT ➜ bin power
//------------------------------------------------------------------------
void PowerSpectrumOutput::LoadOutputData(Mesh *pm)
{
  Kokkos::deep_copy(spectrum_, Real(0));

  MeshBlockPack *pack = pm->pmb_pack;
  const auto &indcs = pm->mb_indcs;
  const int nmb = pack->nmb_thispack;
  const int nxB = indcs.nx1;
  const int nyB = indcs.nx2;
  const int nzB = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const Real inv_ntot = 1.0/static_cast<Real>(nx_*ny_*nz_);
  const Real inv_ntot_sq = inv_ntot*inv_ntot;

  // The primitive 5-D view (var,m,k,j,i) — identical to derived_variables.cpp
  const auto &w0_ = (pm->pmb_pack->phydro != nullptr) ?
                     pm->pmb_pack->phydro->w0 :
                     pm->pmb_pack->pmhd  ->w0;
  bool spectrum_of_magnetic = (out_params.variable.compare("magnetic_field") == 0 ||
                               out_params.variable.compare("magnetic") == 0 ||
                               out_params.variable.compare("bfield") == 0 ||
                               out_params.variable.compare("mhd_bcc1") == 0);
  DvceArray5D<Real> bcc0_;
  if (spectrum_of_magnetic) {
    bcc0_ = pm->pmb_pack->pmhd->bcc0;
  }

  bool spectrum_of_density = (out_params.variable.compare("density") == 0 ||
                              out_params.variable.compare("hydro_w_d") == 0 ||
                              out_params.variable.compare("mhd_w_d") == 0);
  int nfields = (spectrum_of_density) ? 1 : 3;

  for (int comp = 0; comp < nfields; ++comp) {
    int iv = IDN;
    if (!spectrum_of_density && !spectrum_of_magnetic) {
      iv = (comp==0 ? IVX : comp==1 ? IVY : IVZ);
    }

    // ---- pack local active-zone field data into contiguous send buffer ---
    const int64_t ncell_mb = static_cast<int64_t>(nxB) * nyB * nzB;
    const int64_t ncell_local = static_cast<int64_t>(nmb) * ncell_mb;
    Kokkos::View<Real*> send_dev("spec_send_dev", ncell_local);
    Kokkos::parallel_for(
      "pack_local_field",
      Kokkos::RangePolicy<>(0, ncell_local),
      KOKKOS_LAMBDA(const int64_t idx) {
        const int m = idx / ncell_mb;
        const int64_t rem0 = idx - static_cast<int64_t>(m) * ncell_mb;
        const int kB = rem0 / (static_cast<int64_t>(nyB) * nxB);
        const int64_t rem1 = rem0 - static_cast<int64_t>(kB) * nyB * nxB;
        const int jB = rem1 / nxB;
        const int iB = rem1 - static_cast<int64_t>(jB) * nxB;
        if (spectrum_of_magnetic) {
          send_dev(idx) = bcc0_(m, comp, kB + ks, jB + js, iB + is);
        } else {
          send_dev(idx) = w0_(m, iv, kB + ks, jB + js, iB + is);
        }
      });
    auto send_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_dev);
    std::vector<Real> local_field(ncell_local);
    for (int64_t q = 0; q < ncell_local; ++q) local_field[q] = send_host(q);

    std::vector<int> local_gids(nmb);
    for (int m = 0; m < nmb; ++m) {
      local_gids[m] = pack->pmb->mb_gid.h_view(m);
    }

    bool do_fft_and_bin = true;

#if MPI_PARALLEL_ENABLED
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int local_nmb = nmb;
    int local_ncell = static_cast<int>(ncell_local);
    std::vector<int> nmb_eachrank;
    std::vector<int> ncell_eachrank;
    if (rank == 0) {
      nmb_eachrank.resize(nranks);
      ncell_eachrank.resize(nranks);
    }

    MPI_Gather(&local_nmb, 1, MPI_INT,
               (rank == 0 ? nmb_eachrank.data() : nullptr), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_ncell, 1, MPI_INT,
               (rank == 0 ? ncell_eachrank.data() : nullptr), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> mb_displs, cell_displs;
    std::vector<int> recv_gids;
    std::vector<Real> recv_field;
    if (rank == 0) {
      mb_displs.assign(nranks, 0);
      cell_displs.assign(nranks, 0);
      for (int r = 1; r < nranks; ++r) {
        mb_displs[r] = mb_displs[r-1] + nmb_eachrank[r-1];
        cell_displs[r] = cell_displs[r-1] + ncell_eachrank[r-1];
      }
      const int total_nmb = mb_displs[nranks-1] + nmb_eachrank[nranks-1];
      const int total_ncell = cell_displs[nranks-1] + ncell_eachrank[nranks-1];
      recv_gids.resize(total_nmb);
      recv_field.resize(total_ncell);
    }

    MPI_Gatherv(local_gids.data(), local_nmb, MPI_INT,
                (rank == 0 ? recv_gids.data() : nullptr),
                (rank == 0 ? nmb_eachrank.data() : nullptr),
                (rank == 0 ? mb_displs.data() : nullptr),
                MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_field.data(), local_ncell, MPI_ATHENA_REAL,
                (rank == 0 ? recv_field.data() : nullptr),
                (rank == 0 ? ncell_eachrank.data() : nullptr),
                (rank == 0 ? cell_displs.data() : nullptr),
                MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);

    do_fft_and_bin = (rank == 0);
    if (rank == 0) {
      Kokkos::deep_copy(fft_in_, Real(0));
      for (int r = 0; r < nranks; ++r) {
        for (int bm = 0; bm < nmb_eachrank[r]; ++bm) {
          const int gid = recv_gids[mb_displs[r] + bm];
          const auto &lloc = pm->lloc_eachmb[gid];
          const int bx = static_cast<int>(lloc.lx1);
          const int by = static_cast<int>(lloc.lx2);
          const int bz = static_cast<int>(lloc.lx3);
          const int x0 = bx * nxB;
          const int y0 = by * nyB;
          const int z0 = bz * nzB;
          const int64_t off = static_cast<int64_t>(cell_displs[r]) + bm * ncell_mb;
          for (int kB = 0; kB < nzB; ++kB) {
            for (int jB = 0; jB < nyB; ++jB) {
              for (int iB = 0; iB < nxB; ++iB) {
                const int64_t loc = off + (static_cast<int64_t>(kB) * nyB + jB) * nxB + iB;
                fft_in_(x0 + iB, y0 + jB, z0 + kB) = recv_field[loc];
              }
            }
          }
        }
      }
    }
#else
    Kokkos::deep_copy(fft_in_, Real(0));
    for (int m = 0; m < nmb; ++m) {
      const int gid = local_gids[m];
      const auto &lloc = pm->lloc_eachmb[gid];
      const int x0 = static_cast<int>(lloc.lx1) * nxB;
      const int y0 = static_cast<int>(lloc.lx2) * nyB;
      const int z0 = static_cast<int>(lloc.lx3) * nzB;
      const int64_t off = static_cast<int64_t>(m) * ncell_mb;
      for (int kB = 0; kB < nzB; ++kB) {
        for (int jB = 0; jB < nyB; ++jB) {
          for (int iB = 0; iB < nxB; ++iB) {
            const int64_t loc = off + (static_cast<int64_t>(kB) * nyB + jB) * nxB + iB;
            fft_in_(x0 + iB, y0 + jB, z0 + kB) = local_field[loc];
          }
        }
      }
    }
#endif

    if (!do_fft_and_bin) continue;

    // ---- forward FFT on global field (rank 0 in MPI) --------------------
    plan_->execute_impl(fft_in_, fft_out_);

    const int64_t nmodes = int64_t(nx_) * ny_ * (nz_/2 + 1);

    // ---- bin |F|² -------------------------------------------------------
    Kokkos::parallel_for(
      "bin_power",
      Kokkos::RangePolicy<>(0,nmodes),
      KOKKOS_LAMBDA(int64_t lin){
        int iz =  lin / (nx_*ny_);
        int rem = lin % (nx_*ny_);
        int iy =  rem / nx_;
        int ix =  rem % nx_;

        int kx = (ix <= nx_/2 ? ix : ix-nx_);
        int ky = (iy <= ny_/2 ? iy : iy-ny_);
        int kz =  iz;                           // r2c: kz ∈ [0,nz/2]

        Real km = sqrt(double(kx*kx + ky*ky + kz*kz));
        int  s  = int(floor(km));

        if (s>=1 && s<=nbins_) {
          Real half_weight = 2.0;
          if (iz == 0 || ((nz_%2 == 0) && (iz == nz_/2))) {
            half_weight = 1.0;
          }
          auto z = fft_out_(ix,iy,iz);
          Kokkos::atomic_add(&spectrum_(s-1),
                             half_weight*(z.real()*z.real() + z.imag()*z.imag())*inv_ntot_sq);
        }
      });
  }
}

//------------------------------------------------------------------------
// WriteOutputFile (rank 0)  — k (integer), P(k)
//------------------------------------------------------------------------
void PowerSpectrumOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin)
{
#if MPI_PARALLEL_ENABLED
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
  for (int s = 1; s <= nbins_; ++s)
      ofs << s << ' ' << host(s-1) << '\n';

  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
