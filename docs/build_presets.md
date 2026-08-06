# CMake build presets

AthenaK provides shared configure and build presets in `CMakePresets.json`.
Each preset uses a separate `build/<preset-name>` directory so incompatible
compiler, accelerator, and FFT configurations do not share a CMake cache.

## Local overrides

Put personal problem selections, experimental flags, and nonstandard paths in
`CMakeUserPresets.json`. CMake loads that file automatically, but Git ignores
it because it is specific to one user and machine.

## Trillium CPU

Load the AthenaK environment before configuring or building:

```bash
source ~/.env/athenak_env
```

The loader must provide the MPI compiler wrappers and FFTW modules. For the
HeFFTe preset it must also export `HEFFTE_PREFIX`, the installation prefix of
the local HeFFTe package.

Configure and build a turbulence problem with HeFFTe:

```bash
cmake --preset trillium-cpu-heffte -DPROBLEM=turb
cmake --build --preset trillium-cpu-heffte --parallel 48
```

Use `trillium-cpu-fftw` for the KokkosFFT/FFTW configuration.

## Other systems

The shared presets also include Apple Silicon CPU, BEE Volta GPU, and Trillium
Hopper GPU configurations. List the presets supported by the installed CMake:

```bash
