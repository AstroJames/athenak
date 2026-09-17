# Iterative FOFC and WENOZ CGM example: main release validation

## Change and provenance

Iterative Newtonian ideal-MHD FOFC rechecks the trial update after shared-face
flux replacements and propagates new correction flags across MeshBlock/MPI
boundaries. The opt-in `fofc_max_iterations` includes the initial correction;
its default remains one. Failure guards and bounded failure-only snapshots
report exhausted or inadmissible corrections. This is not timestep retry or
a positivity guarantee for the later CT/source update.

The existing `integration/iterative_FOFC` checkout was reused. Merge `03bcf03`
incorporates remote main `47ca932` (including the upstream reconstruction and
3D MHD symmetry changes) into FOFC development `5f3df61`. The merge was clean.
No supernova diagnostic/source edits, other worktree changes, generated builds,
or local submodule symlinks are included in the release.

The new input is
`inputs/turbulence/cgm_turbulence_mhd_iterative_fofc.input`: a 32 x 32 x 64
example with RK3/WENOZ/HLLD, CFL 0.3, four ghost cells, eight FOFC passes, and
verbose diagnostics disabled. Its physical parameters follow the previous
high-resolution CFL comparison, including diode vertical boundaries,
compressive `dedt=2.5e-4` forcing, gravity, cooling, thermostat, and the original
floors. Its 100 Myr duration and modest output schedule make it a usage example,
not a spatial-convergence result. The documentation specifies the earlier
264 x 264 x 528 production layout separately.

## Current merged-code validation

Slurm debug allocation **2333410** on **tri0210** completed with exit zero and
was released. All executions and runtime files were on scratch. Release builds
used the canonical AthenaK environment, the repository's pinned Kokkos and
KokkosFFT sources, and separate serial/no-FFT and MPI/CGM/KokkosFFT builds.
MPI runs used the site VAST-wrapped `mpirun`.

| Check | Cases | Result |
| --- | ---: | --- |
| FOFC production-kernel fixtures, serial | 12 | Pass |
| Same fixtures, MPI at 1/2/8 ranks | 36 | Pass |
| Smooth 3D WENOZ MHD accuracy, caps 1/8 | 8 | Pass |
| MHD point symmetry | 8 | Pass |
| Hydro Rayleigh-Taylor symmetry | 12 | Pass |
| Strict 1D hydro FOFC versus constant 2D | 12 | Pass |
| 3D WENOZ MHD point symmetry | 2 | Pass |
| WENOZ CGM example, one MPI rank | 4 steps | Pass |
| WENOZ CGM example, eight MPI ranks | 100 Myr | Pass |

The 90 fixture/regression cases include expected diagnostic failures for
unrepairable states and cap exhaustion. Smooth-wave error tables remain
bitwise identical between caps 1 and 8. The full small CGM example completed
129 steps, and all 101 saved history rows are finite. These are selected CPU
checks; full repository/device CI is not claimed.

Reports, scripts, and provenance are retained under
`/scratch/jbeattie/cgm_fofc_main_integration/release_20260916`.

## Resolution of the earlier CFL validation gate

The previously submitted high-resolution CFL-0.3 job **2273136** completed
600 Myr at cycle 13861, exit zero. Its 6001 history rows were checked here and
are finite. This is evidence from the earlier solver revision, not a fresh
high-resolution replay of merge `03bcf03`.

The paired CFL-0.6 replay **2273100** failed at 312.866927 Myr, cycle 5222,
RK3 stage 2. Its retained snapshot identifies two already-flagged cells with
trial densities -1.3865890e-4 and -1.1647016e-3. Independent production-LLF
reconstruction agrees with the saved flux divergence to round-off. Repeating
identical first-order corrections cannot repair those frozen stage states.
This supports using CFL 0.3 for the example, not assuming a larger pass cap
can compensate for an excessive multidimensional timestep.

The retained six measured pairs from timing job **2272611** give a median
cap-8 overhead of **2.54%** and mean **2.83%** relative to cap 1. Individual
pairs range from -2.79% to 8.08%, showing timing scatter. These were matched
100-step segments at the same CFL, with verbose diagnostics and outputs
disabled; they do not measure the cost of changing CFL from 0.6 to 0.3 or
promise an overhead for arbitrary problems/resolutions.


## Style and footprint

Focused Python lint passed. Focused C++ lint passed with the compact-if
`whitespace/newline` category excluded; all 31 reported statements were
verified to already exist on remote main. No unrelated formatting edits were
made. This does not claim full repository CI has passed.

After validation, 148 disposable runtime files (92.5 MiB), including staged
executables and raw test outputs, were removed. Reports, scripts, input,
provenance and completion excerpts remain; see `cleanup.json` in the validation
directory. Rebuild/restage executables before rerunning the retained harness.
The existing reusable compilation trees remain available. Temporary lint
packages under `/tmp` were removed after use.
