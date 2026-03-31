//=== power_spectrum_backend.cpp ===========================================
//  Backend implementations for power spectrum output
//===========================================================================

#include "power_spectrum_backend.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if FFT_ENABLED
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>
#endif

#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

#if HEFFTE_ENABLED
#include <heffte.h>
#endif

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

#if FFT_ENABLED
using real_view_t =
    Kokkos::View<Real***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using complex_view_t =
    Kokkos::View<Kokkos::complex<Real>***, Kokkos::LayoutRight,
                 Kokkos::DefaultExecutionSpace>;
using PlanType =
    KokkosFFT::Plan<Kokkos::DefaultExecutionSpace,
                    real_view_t,
                    complex_view_t,
                    3>;

enum class SpectrumFieldType {
  kDensity,
  kVelocity,
  kMagnetic
};

struct SpectrumFieldAlias {
  const char *alias;
  SpectrumFieldType type;
};

constexpr std::array<SpectrumFieldAlias, 8> kSpectrumFieldAliases{{
    {"density", SpectrumFieldType::kDensity},
    {"hydro_w_d", SpectrumFieldType::kDensity},
    {"mhd_w_d", SpectrumFieldType::kDensity},
    {"velocity", SpectrumFieldType::kVelocity},
    {"magnetic_field", SpectrumFieldType::kMagnetic},
    {"magnetic", SpectrumFieldType::kMagnetic},
    {"bfield", SpectrumFieldType::kMagnetic},
    {"mhd_bcc1", SpectrumFieldType::kMagnetic},
}};

SpectrumFieldType ResolveSpectrumFieldType(const std::string &name) {
  for (const auto &entry : kSpectrumFieldAliases) {
    if (name.compare(entry.alias) == 0) return entry.type;
  }
  std::cout << "### FATAL ERROR in ResolveSpectrumFieldType\n"
            << "Unknown power_spectrum variable='" << name << "'.\n";
  std::exit(EXIT_FAILURE);
}

int NumFieldComponents(const SpectrumFieldType type) {
  return (type == SpectrumFieldType::kDensity ? 1 : 3);
}

bool FieldUsesMagnetic(const SpectrumFieldType type) {
  return (type == SpectrumFieldType::kMagnetic);
}

int VelocityComponentIndex(const int comp) {
  return (comp == 0 ? IVX : (comp == 1 ? IVY : IVZ));
}

void ValidateFieldAvailability(const SpectrumFieldType type, Mesh *pm) {
  if (type == SpectrumFieldType::kMagnetic && pm->pmb_pack->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in PowerSpectrumBackend\n"
              << "variable='magnetic' requested, but MHD data is unavailable.\n";
    std::exit(EXIT_FAILURE);
  }
}

class LegacyPowerSpectrumBackend final : public PowerSpectrumBackend {
 public:
  explicit LegacyPowerSpectrumBackend(Mesh *pm) {
    const auto &g = pm->mesh_indcs;
    nx_ = g.nx1;
    ny_ = g.nx2;
    nz_ = g.nx3;
    nbins_ = std::min({nx_/2, ny_/2, nz_/2});

    if (pm->multilevel) {
      std::cout << "### FATAL ERROR in LegacyPowerSpectrumBackend\n"
                << "power_spectrum currently supports only single-level meshes.\n";
      std::exit(EXIT_FAILURE);
    }

#if MPI_PARALLEL_ENABLED
    int rank = 0;
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

  int GetNumBins() const override { return nbins_; }

  void Compute(Mesh *pm, const OutputParameters &out_params,
               Kokkos::View<Real*> spectrum) override {
    MeshBlockPack *pack = pm->pmb_pack;
    const auto &indcs = pm->mb_indcs;
    const int nmb = pack->nmb_thispack;
    const int nxB = indcs.nx1;
    const int nyB = indcs.nx2;
    const int nzB = indcs.nx3;
    const int is = indcs.is;
    const int js = indcs.js;
    const int ks = indcs.ks;
    const Real inv_ntot = 1.0/static_cast<Real>(
        static_cast<int64_t>(nx_)*ny_*nz_);
    const Real inv_ntot_sq = inv_ntot*inv_ntot;
    const SpectrumFieldType field_type = ResolveSpectrumFieldType(out_params.variable);
    ValidateFieldAvailability(field_type, pm);
    const bool spectrum_of_magnetic = FieldUsesMagnetic(field_type);
    const int nfields = NumFieldComponents(field_type);

    const auto &w0_ = (pm->pmb_pack->phydro != nullptr) ?
                      pm->pmb_pack->phydro->w0 :
                      pm->pmb_pack->pmhd->w0;
    DvceArray5D<Real> bcc0_;
    if (spectrum_of_magnetic) {
      bcc0_ = pm->pmb_pack->pmhd->bcc0;
    }

    for (int comp = 0; comp < nfields; ++comp) {
      int iv = IDN;
      if (field_type == SpectrumFieldType::kVelocity) {
        iv = VelocityComponentIndex(comp);
      }

      const int64_t ncell_mb = static_cast<int64_t>(nxB) * nyB * nzB;
      const int64_t ncell_local = static_cast<int64_t>(nmb) * ncell_mb;
      Kokkos::View<Real*> send_dev("spec_send_dev", ncell_local);
      Kokkos::parallel_for(
          "pack_local_field", Kokkos::RangePolicy<>(0, ncell_local),
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
      auto send_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_dev);
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
                 (rank == 0 ? nmb_eachrank.data() : nullptr), 1, MPI_INT, 0,
                 MPI_COMM_WORLD);
      MPI_Gather(&local_ncell, 1, MPI_INT,
                 (rank == 0 ? ncell_eachrank.data() : nullptr), 1, MPI_INT, 0,
                 MPI_COMM_WORLD);

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
                  (rank == 0 ? mb_displs.data() : nullptr), MPI_INT, 0,
                  MPI_COMM_WORLD);
      MPI_Gatherv(local_field.data(), local_ncell, MPI_ATHENA_REAL,
                  (rank == 0 ? recv_field.data() : nullptr),
                  (rank == 0 ? ncell_eachrank.data() : nullptr),
                  (rank == 0 ? cell_displs.data() : nullptr),
                  MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);

      do_fft_and_bin = (rank == 0);
      if (rank == 0) {
        auto fft_in_host = Kokkos::create_mirror_view(fft_in_);
        Kokkos::deep_copy(fft_in_host, Real(0));
        for (int r = 0; r < nranks; ++r) {
          for (int bm = 0; bm < nmb_eachrank[r]; ++bm) {
            const int gid = recv_gids[mb_displs[r] + bm];
            const auto &lloc = pm->lloc_eachmb[gid];
            const int x0 = static_cast<int>(lloc.lx1) * nxB;
            const int y0 = static_cast<int>(lloc.lx2) * nyB;
            const int z0 = static_cast<int>(lloc.lx3) * nzB;
            const int64_t off = static_cast<int64_t>(cell_displs[r]) + bm * ncell_mb;
            for (int kB = 0; kB < nzB; ++kB) {
              for (int jB = 0; jB < nyB; ++jB) {
                for (int iB = 0; iB < nxB; ++iB) {
                  const int64_t loc = off + (static_cast<int64_t>(kB) * nyB + jB) * nxB + iB;
                  fft_in_host(x0 + iB, y0 + jB, z0 + kB) = recv_field[loc];
                }
              }
            }
          }
        }
        Kokkos::deep_copy(fft_in_, fft_in_host);
      }
#else
      {
        auto fft_in_host = Kokkos::create_mirror_view(fft_in_);
        Kokkos::deep_copy(fft_in_host, Real(0));
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
                fft_in_host(x0 + iB, y0 + jB, z0 + kB) = local_field[loc];
              }
            }
          }
        }
        Kokkos::deep_copy(fft_in_, fft_in_host);
      }
#endif

      if (!do_fft_and_bin) continue;

      plan_->execute_impl(fft_in_, fft_out_);

      const int64_t nmodes = int64_t(nx_) * ny_ * (nz_/2 + 1);
      Kokkos::parallel_for(
          "bin_power", Kokkos::RangePolicy<>(0, nmodes), KOKKOS_LAMBDA(int64_t lin) {
            int iz = lin / (nx_ * ny_);
            int rem = lin % (nx_ * ny_);
            int iy = rem / nx_;
            int ix = rem % nx_;

            int kx = (ix <= nx_/2 ? ix : ix - nx_);
            int ky = (iy <= ny_/2 ? iy : iy - ny_);
            int kz = iz;

            Real km = sqrt(double(kx*kx + ky*ky + kz*kz));
            int s = int(floor(km));
            if (s >= 1 && s <= nbins_) {
              Real half_weight = 2.0;
              if (iz == 0 || ((nz_ % 2 == 0) && (iz == nz_/2))) {
                half_weight = 1.0;
              }
              auto z = fft_out_(ix, iy, iz);
              Kokkos::atomic_add(&spectrum(s - 1),
                                 half_weight * (z.real()*z.real() + z.imag()*z.imag()) *
                                     inv_ntot_sq);
            }
          });
    }
  }

 private:
  int nbins_ = 0;
  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;
  real_view_t fft_in_;
  complex_view_t fft_out_;
  std::unique_ptr<PlanType> plan_;
};
#endif  // FFT_ENABLED

#if HEFFTE_ENABLED
struct HeffteCellSample {
  int gx;
  int gy;
  int gz;
  Real val;
};

class HefftePowerSpectrumBackend final : public PowerSpectrumBackend {
 public:
  explicit HefftePowerSpectrumBackend(Mesh *pm) {
    if (pm->multilevel) {
      std::cout << "### FATAL ERROR in HefftePowerSpectrumBackend\n"
                << "heFFTe power_spectrum currently supports only single-level meshes.\n";
      std::exit(EXIT_FAILURE);
    }
    const auto &g = pm->mesh_indcs;
    nx_ = g.nx1;
    ny_ = g.nx2;
    nz_ = g.nx3;
    nbins_ = std::min({nx_/2, ny_/2, nz_/2});

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_nranks_);
    fft_nranks_ = std::min(world_nranks_, nx_);
    participates_fft_ = (world_rank_ < fft_nranks_);
    if (world_rank_ == 0) {
      std::cout << "PowerSpectrum(heffte): using " << fft_nranks_
                << " MPI rank(s) for FFT out of " << world_nranks_
                << " total rank(s), capped by nx=" << nx_ << ".\n";
    }

    // Precompute gx-to-owner lookup table consistent with slab boundaries
    gx_to_owner_.resize(nx_);
    for (int r = 0; r < fft_nranks_; ++r) {
      const int x0 = (r * nx_) / fft_nranks_;
      const int x1 = ((r + 1) * nx_) / fft_nranks_ - 1;
      for (int gx = x0; gx <= x1; ++gx) {
        gx_to_owner_[gx] = r;
      }
    }

    int color = (participates_fft_ ? 0 : MPI_UNDEFINED);
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank_, &fft_comm_);
    if (!participates_fft_) return;

    MPI_Comm_rank(fft_comm_, &fft_rank_);
    slab_x0_ = (fft_rank_ * nx_) / fft_nranks_;
    slab_x1_ = ((fft_rank_ + 1) * nx_) / fft_nranks_ - 1;
    local_nx_ = slab_x1_ - slab_x0_ + 1;

    using box3d = heffte::box3d<int>;
    // heFFTe box order is set to row-major [x][y][z] layout where z is fastest.
    const std::array<int, 3> mem_order = {2, 1, 0};
    const box3d in_box({slab_x0_, 0, 0}, {slab_x1_, ny_ - 1, nz_ - 1}, mem_order);
    const box3d out_box({slab_x0_, 0, 0}, {slab_x1_, ny_ - 1, nz_/2}, mem_order);
    plan_ = std::make_unique<heffte::fft3d_r2c<heffte::backend::stock>>(
        in_box, out_box, 2, fft_comm_);
  }

  ~HefftePowerSpectrumBackend() override {
    if (participates_fft_ && fft_comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&fft_comm_);
    }
  }

  int GetNumBins() const override { return nbins_; }

  void Compute(Mesh *pm, const OutputParameters &out_params,
               Kokkos::View<Real*> spectrum) override {
    std::vector<Real> global_bins(nbins_, Real(0));

    const auto &indcs = pm->mb_indcs;
    const int nmb = pm->pmb_pack->nmb_thispack;
    const int nxB = indcs.nx1;
    const int nyB = indcs.nx2;
    const int nzB = indcs.nx3;
    const int is = indcs.is;
    const int js = indcs.js;
    const int ks = indcs.ks;
    const Real inv_ntot = 1.0/static_cast<Real>(
        static_cast<int64_t>(nx_)*ny_*nz_);
    const Real inv_ntot_sq = inv_ntot*inv_ntot;
    const SpectrumFieldType field_type = ResolveSpectrumFieldType(out_params.variable);
    ValidateFieldAvailability(field_type, pm);
    const bool spectrum_of_magnetic = FieldUsesMagnetic(field_type);
    const int nfields = NumFieldComponents(field_type);

    const auto &w0_ = (pm->pmb_pack->phydro != nullptr) ?
                      pm->pmb_pack->phydro->w0 :
                      pm->pmb_pack->pmhd->w0;
    DvceArray5D<Real> bcc0_;
    if (spectrum_of_magnetic) {
      bcc0_ = pm->pmb_pack->pmhd->bcc0;
    }

    std::vector<int> local_gids(nmb);
    for (int m = 0; m < nmb; ++m) {
      local_gids[m] = pm->pmb_pack->pmb->mb_gid.h_view(m);
    }

    for (int comp = 0; comp < nfields; ++comp) {
      int iv = IDN;
      if (field_type == SpectrumFieldType::kVelocity) {
        iv = VelocityComponentIndex(comp);
      }

      const int64_t ncell_mb = static_cast<int64_t>(nxB) * nyB * nzB;
      const int64_t ncell_local = static_cast<int64_t>(nmb) * ncell_mb;
      Kokkos::View<Real*> send_dev("heffte_send_dev", ncell_local);
      Kokkos::parallel_for(
          "pack_local_field_heffte", Kokkos::RangePolicy<>(0, ncell_local),
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
      auto send_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_dev);

      std::vector<std::vector<HeffteCellSample>> send_lists(world_nranks_);
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
              const int gx = x0 + iB;
              const int gy = y0 + jB;
              const int gz = z0 + kB;
              const int owner = gx_to_owner_[gx];
              const int64_t loc =
                  off + (static_cast<int64_t>(kB) * nyB + jB) * nxB + iB;
              send_lists[owner].push_back({gx, gy, gz, send_host(loc)});
            }
          }
        }
      }

      std::vector<int> send_counts(world_nranks_, 0), recv_counts(world_nranks_, 0);
      for (int r = 0; r < world_nranks_; ++r) {
        send_counts[r] = static_cast<int>(send_lists[r].size());
      }
      MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                   MPI_COMM_WORLD);

      std::vector<int> send_displs(world_nranks_, 0), recv_displs(world_nranks_, 0);
      for (int r = 1; r < world_nranks_; ++r) {
        send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
      }
      const int total_send = send_displs[world_nranks_ - 1] + send_counts[world_nranks_ - 1];
      const int total_recv = recv_displs[world_nranks_ - 1] + recv_counts[world_nranks_ - 1];

      std::vector<HeffteCellSample> send_flat(total_send);
      for (int r = 0; r < world_nranks_; ++r) {
        std::copy(send_lists[r].begin(), send_lists[r].end(),
                  send_flat.begin() + send_displs[r]);
      }
      std::vector<HeffteCellSample> recv_flat(total_recv);

      std::vector<int> send_counts_b(world_nranks_, 0), recv_counts_b(world_nranks_, 0);
      std::vector<int> send_displs_b(world_nranks_, 0), recv_displs_b(world_nranks_, 0);
      for (int r = 0; r < world_nranks_; ++r) {
        send_counts_b[r] = send_counts[r] * static_cast<int>(sizeof(HeffteCellSample));
        recv_counts_b[r] = recv_counts[r] * static_cast<int>(sizeof(HeffteCellSample));
        send_displs_b[r] = send_displs[r] * static_cast<int>(sizeof(HeffteCellSample));
        recv_displs_b[r] = recv_displs[r] * static_cast<int>(sizeof(HeffteCellSample));
      }

      MPI_Alltoallv(reinterpret_cast<char*>(send_flat.data()), send_counts_b.data(),
                    send_displs_b.data(), MPI_BYTE,
                    reinterpret_cast<char*>(recv_flat.data()), recv_counts_b.data(),
                    recv_displs_b.data(), MPI_BYTE, MPI_COMM_WORLD);

      std::vector<Real> local_bins(nbins_, Real(0));
      if (participates_fft_) {
        std::vector<Real> in_data(plan_->size_inbox(), Real(0));
        for (const auto &s : recv_flat) {
          const int ix = s.gx - slab_x0_;
          if (ix < 0 || ix >= local_nx_) continue;
          if (s.gy < 0 || s.gy >= ny_ || s.gz < 0 || s.gz >= nz_) continue;
          const int64_t id = (static_cast<int64_t>(ix) * ny_ + s.gy) * nz_ + s.gz;
          in_data[id] = s.val;
        }

        std::vector<std::complex<Real>> out_data(plan_->size_outbox());
        plan_->forward(in_data.data(), out_data.data(), heffte::scale::none);

        const int nz_out = nz_/2 + 1;
        for (int ix = 0; ix < local_nx_; ++ix) {
          const int gx = slab_x0_ + ix;
          const int kx = (gx <= nx_/2 ? gx : gx - nx_);
          for (int gy = 0; gy < ny_; ++gy) {
            const int ky = (gy <= ny_/2 ? gy : gy - ny_);
            for (int gz = 0; gz < nz_out; ++gz) {
              const int kz = gz;
              const Real km = sqrt(static_cast<Real>(kx*kx + ky*ky + kz*kz));
              const int sbin = static_cast<int>(floor(km));
              if (sbin < 1 || sbin > nbins_) continue;
              Real half_weight = 2.0;
              if (gz == 0 || ((nz_ % 2 == 0) && (gz == nz_/2))) {
                half_weight = 1.0;
              }
              const int64_t id = (static_cast<int64_t>(ix) * ny_ + gy) * nz_out + gz;
              const auto &z = out_data[id];
              local_bins[sbin - 1] +=
                  half_weight * (z.real()*z.real() + z.imag()*z.imag()) * inv_ntot_sq;
            }
          }
        }
      }

      if (participates_fft_) {
        MPI_Allreduce(MPI_IN_PLACE, local_bins.data(), nbins_, MPI_ATHENA_REAL, MPI_SUM,
                      fft_comm_);
        if (fft_rank_ == 0) {
          for (int n = 0; n < nbins_; ++n) global_bins[n] += local_bins[n];
        }
      }
      MPI_Bcast(global_bins.data(), nbins_, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    }

    auto host = Kokkos::create_mirror_view(spectrum);
    for (int n = 0; n < nbins_; ++n) host(n) = global_bins[n];
    Kokkos::deep_copy(spectrum, host);
  }

 private:
  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;
  int nbins_ = 0;
  int world_rank_ = 0;
  int world_nranks_ = 1;
  int fft_nranks_ = 1;
  int fft_rank_ = -1;
  bool participates_fft_ = false;
  MPI_Comm fft_comm_ = MPI_COMM_NULL;
  int slab_x0_ = 0;
  int slab_x1_ = -1;
  int local_nx_ = 0;
  std::vector<int> gx_to_owner_;
  std::unique_ptr<heffte::fft3d_r2c<heffte::backend::stock>> plan_;
};
#endif  // HEFFTE_ENABLED

}  // namespace

std::unique_ptr<PowerSpectrumBackend> BuildPowerSpectrumBackend(
    Mesh *pm, const OutputParameters &out_params) {
#if FFT_ENABLED
  if (out_params.fft_backend.compare("legacy") == 0) {
    return std::make_unique<LegacyPowerSpectrumBackend>(pm);
  }
  if (out_params.fft_backend.compare("heffte") == 0) {
#if HEFFTE_ENABLED
    return std::make_unique<HefftePowerSpectrumBackend>(pm);
#else
    std::cout << "### FATAL ERROR in BuildPowerSpectrumBackend\n"
              << "fft_backend='heffte' requested, but this binary was compiled without "
              << "Athena_ENABLE_HEFFTE.\n";
    std::exit(EXIT_FAILURE);
#endif
  }
  std::cout << "### FATAL ERROR in BuildPowerSpectrumBackend\n"
            << "Unknown fft_backend='" << out_params.fft_backend << "' requested.\n";
  std::exit(EXIT_FAILURE);
#else
  std::cout << "### FATAL ERROR in BuildPowerSpectrumBackend\n"
            << "power_spectrum output requested, but this binary was compiled with "
            << "Athena_ENABLE_FFT=OFF.\n";
  std::exit(EXIT_FAILURE);
#endif
}
