# heFFTe Build Notes

AthenaK supports heFFTe as an optional backend for `file_type = power_spectrum`.

## Configure with heFFTe

heFFTe support requires MPI:

```bash
cmake .. -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_HEFFTE=ON
```

### Preferred (portable): CMake package discovery

Install heFFTe so it provides `HeffteConfig.cmake`, then point CMake at the
install prefix:

```bash
cmake .. \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_HEFFTE=ON \
  -DCMAKE_PREFIX_PATH=/path/to/heffte/prefix
```

### Fallback: direct include/library paths

If no package config is available:

```bash
cmake .. \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_HEFFTE=ON \
  -DHEFFTE_INCLUDE_DIR=/path/to/heffte/include \
  -DHEFFTE_LIBRARY=/path/to/heffte/lib/libheffte.so
```

You can also pass `-DHEFFTE_ROOT=/path/to/heffte/prefix`.

## Runtime selection

Enable heFFTe in an input file by setting:

```text
<outputX>
file_type = power_spectrum
fft_backend = heffte
```

If `fft_backend = heffte` is requested in runtime inputs but AthenaK was built
without `Athena_ENABLE_HEFFTE=ON`, AthenaK will stop with a clear error.
