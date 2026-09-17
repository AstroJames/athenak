# Iterative first-order flux correction

Newtonian ideal MHD can opt into iterative FOFC:

```text
<mhd>
fofc = true
fofc_max_iterations = 8
fofc_diagnostics = false
```

The default cap is 1 (the original single-pass correction path). The cap is a
positive integer counting total correction passes per RK stage, including the
first pass, not an additional retry count. WENOZ with FOFC requires four ghost
cells. The primary reconstruction/Riemann solver remain on uncorrected faces;
corrected faces use the existing first-order LLF fallback.

Each pass exchanges accumulated owning-cell flags with neighboring blocks,
corrects flagged faces and their face electric fields, and recomputes the trial
update for flagged cells and their face neighbors. Newly failing neighbors are
flagged for the next pass. Every prediction uses the original stage registers;
iteration does not advance physical time or repeatedly modify those registers.
Explicit nonfinite checks supplement the ordinary EOS floor tests.

There is no global early exit. All ranks perform the configured number of
neighbor-mask exchanges, but skip correction/recheck kernels when neither local
nor incoming flags changed. The mask traversal of an active recheck remains a
dense scan; this is not a zero-overhead or entirely sparse implementation.

At the cap, unprocessed newly flagged cells or remaining nonfinite/nonpositive
density trial states produce a diagnostic abort. Already corrected, finite
states with positive density can still use the normal configured floors.
Actual primitive recovery also rejects nonfinite or nonpositive-density
conserved states and nonfinite recovered primitives in iterative mode.

## Supported scope and limitations

Iterative mode requires a uniform mesh, one MeshBlockPack per rank, no passive
scalars, no explicit diffusion, and no shearing box. Isothermal and relativistic
physics are not supported by this new path. Unsupported configurations fail at
startup; their existing non-iterative paths are not subject to these restrictions.

The magnetic trial state is the existing cell-centered estimate, not the exact
later constrained-transport update using corner electric fields. Source terms
also act later. This feature is therefore not a mathematical positivity
guarantee for the final multidimensional/source-coupled update. It does not add
a mass source, change the floors, or implement timestep rejection/retry.

Use a conservative multidimensional CFL: the CGM example uses 0.3. Repeating
an already fully corrected update cannot repair a timestep that is too large.
The earlier CGM replay completed 600 Myr at CFL 0.3; its CFL 0.6 counterpart
failed with already-corrected negative-density cells. This is validation of
one configuration, not an unconditional stability guarantee.

Verbose per-cell/per-pass diagnostics are opt-in. They can generate large logs
and should be disabled in production timing measurements. Failure guards remain
enabled independently of verbose diagnostics. An exhausted iterative correction
also writes a bounded failure snapshot (up to 32 hard-invalid cells per failing
rank), including the saved stage states and corrected face fluxes. This snapshot
is produced only on failure; it does not change the correction or timestep.

## Regression tests

The normal regression runner discovers `mhd/mhd_fofc_iterative.py`. Its helper
links the test against already-built production objects without replacing the
simulation executable. Like the current regression build harness, the helper
expects a Makefile-generator build and paths without whitespace.

Tests cover density and energy cascades, nonfinite fluxes, unavoidable finite
floors, multiple correction layers, cap exhaustion, distinct immutable RK3
stage registers, and 3D face/edge/corner mask exchange. The legacy test confirms
that a single pass leaves two negative-density neighbors in the fixture.

For an MPI build, set `ATHENA_FOFC_TEST_RANKS=1,2,8` to include MPI variants.
`ATHENA_FOFC_MPI_LAUNCHER` supplies launcher options if required.
`ATHENA_FOFC_TEST_EXECUTABLE` selects a prebuilt test executable, useful on
systems with read-only source/build mounts on compute nodes.

On Trillium, compile on a login node but run every test under Slurm, with all
runtime output in scratch. See the dated investigation notes for the original
CGM comparison and the integration-validation record for the merged revision.


## CGM example

[`cgm_turbulence_mhd_iterative_fofc.input`](../inputs/turbulence/cgm_turbulence_mhd_iterative_fofc.input)
is a small 32 x 32 x 64 demonstration with eight 16 x 16 x 32 MeshBlocks.
Build `PROBLEM=turb_cgm` with an FFT backend:

```sh
cmake -S . -B build_cgm -DPROBLEM=turb_cgm -DAthena_ENABLE_MPI=ON \
  -DAthena_FFT_BACKEND=KOKKOS -DCMAKE_BUILD_TYPE=Release
cmake --build build_cgm -j 8
```

Run from a writable simulation directory using one, two, four, or eight MPI
ranks, with paths adjusted to the executable and input:

```sh
mpirun -np 8 /path/to/build_cgm/src/athena \
  -i /path/to/inputs/turbulence/cgm_turbulence_mhd_iterative_fofc.input
```

On a cluster this command belongs inside an allocation. On Trillium, source
`~/.env/athenak_env` and the VAST preload script there, use the wrapped `mpirun`,
and write all runtime output under scratch; never run on a login node.

The box spans 100 x 100 x 200 kpc, with periodic x/y and diode (outflow-only)
z boundaries. It uses Newtonian ideal MHD, RK3, WENOZ/HLLD, four ghost cells,
CFL 0.3, eight FOFC passes, and disabled verbose FOFC logging.

The atmosphere starts at 10^7 K with cooling, gravity, and the plane-averaged
thermostat. The source temperature floor is 10^6 K; the retained EOS floor is
`tfloor=0.014494`, with `dfloor=1e-12` and `pfloor=1e-14`. A spectral field is
normalized to `beta_z0=100`. Disk forcing is compressive with `dedt=2.5e-4`
and a 25 Myr correlation time; the central engine is off. These are the
parameters of the high-resolution CFL comparison. They differ from the older
`cgm_turbulence_mhd.input`, which has reflecting z boundaries and weaker forcing.
See [CGM setup](cgm_turbulence_setup.md) for the atmosphere and source formulas.

The example runs 100 Myr, with history every 1 Myr, MHD and forcing fields every
25 Myr, and restarts every 50 Myr, plus initial/final outputs. It is a coarse
usage example. To reproduce the previous 600 Myr validation configuration,
use 264 x 264 x 528 cells, 22 x 22 x 44 MeshBlocks, `tlim=600`, and CFL 0.3;
that layout has 1728 blocks and used nine full 192-rank Trillium nodes.
The earlier completion is recorded in the dated validation notes; it does not
establish convergence or stability for arbitrary longer runs.

The latest integration checks and measured overhead are recorded in
[the release validation note](../notes/2026-09-16-fofc-main-release.md).
