# heFFTe Build Notes

AthenaK compiles exactly one FFT backend into each binary. Select the
MPI-distributed heFFTe implementation with
`Athena_FFT_BACKEND=HEFFTE`. This configuration does not add KokkosFFT.

The selected backend is used by `file_type = power_spectrum` and by spectral
initial-condition generation. The other choices are `KOKKOS` (KokkosFFT,
which uses FFTW on CPU builds) and `NONE`.

## Configure with heFFTe

heFFTe support requires MPI:

```bash
cmake -S . -B build-heffte \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_FFT_BACKEND=HEFFTE
```

### Preferred (portable): CMake package discovery

Install heFFTe so it provides `HeffteConfig.cmake`, then point CMake at the
install prefix:

```bash
cmake -S . -B build-heffte \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_FFT_BACKEND=HEFFTE \
  -DCMAKE_PREFIX_PATH=/path/to/heffte/prefix
```

### Fallback: direct include/library paths

If no package config is available:

```bash
cmake -S . -B build-heffte \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_FFT_BACKEND=HEFFTE \
  -DHEFFTE_INCLUDE_DIR=/path/to/heffte/include \
  -DHEFFTE_LIBRARY=/path/to/heffte/lib/libheffte.so
```

You can also pass `-DHEFFTE_ROOT=/path/to/heffte/prefix`.

For a HeFFTe installation that is independent of FFTW, configure HeFFTe with
its stock backend and `Heffte_ENABLE_FFTW=OFF`. HeFFTe still requires MPI for
distributed transforms.

## Runtime selection

The compiled backend is the runtime default, so no `fft_backend` parameter is
required:

```text
<outputX>
file_type = power_spectrum
```

An explicit `fft_backend = heffte` is accepted for a HeFFTe binary. If an
input explicitly requests a backend different from the one compiled into
AthenaK, startup stops with a clear error. The historical `legacy` name
remains an alias for the `KOKKOS` backend.
