# Iterative FOFC: main integration and release validation

## Provenance and scope

The isolated `integration/iterative_FOFC` worktree integrates development commit
`1a7d19f` with upstream `main` at `1339082362a12a5eb52b90a1c82d8fc1fa3b8b21`
(merge `cab8644`). The shared `/home/jbeattie/athenak` checkout and its unrelated
uncommitted changes were left untouched. No submodule changes are intended.

The upstream merge includes TENO5, strict 1D hydro FOFC bounds, and exact numerical
symmetry changes. The HLLD investigation-only print blocks were removed; the
HLLD numerical formulas match upstream main. Useful opt-in FOFC diagnostics and
the iterative-mode failure guards remain.

Release builds use the repository's exact pinned dependencies:

- Kokkos `08ceff92bcf3a828844480bc1e6137eb74028517` (4.4.0).
- KokkosFFT `8ccad3adfb31fc4df56b33ce966d11d4924394fb` (0.3.0).
- GCC 12.3.1, OpenMPI 4.1.5, double precision, Kokkos Serial.
- `build_cgm`: `PROBLEM=turb_cgm`, MPI enabled, Kokkos FFT backend.
- `build_regression`: built-in problem generators, MPI/FFT disabled.

Existing clean dependency sources are linked locally to avoid duplicate trees.
Those local symlinks and build directories must never be committed.

All simulations and fixture executions run under Slurm in
`/scratch/jbeattie/cgm_fofc_main_integration`, account `rrg-essick`.
Compilation alone runs on the login node.

The staged CGM binary (solver source at `da705e4`) has SHA256
`3b61e094c845890a3aa57aa78e2509e33749f6b145280b13343d3b736f96fbde`.
The staged serial regression binary has SHA256
`0b39983000046c10aa25e169960a33e2ca8f4e23323578abb57a3d8aa76ee79a`.

## Completed regression validation

Job `2272589` executed these standard regression modules and analyzers using
prebuilt binaries and independent per-case scratch directories:

| Module | Cases | Result |
| --- | ---: | --- |
| MHD iterative FOFC smooth 3D accuracy | 8 | Pass |
| MHD point symmetry | 8 | Pass |
| Hydro Rayleigh-Taylor symmetry | 12 | Pass |
| Hydro strict 1D FOFC | 8 | Pass |
| MHD linear waves | 382 | Pass |

The new smooth-wave test requires bitwise-identical error tables for caps 1 and
8 at each mesh layout, finite results, and the expected convergence/accuracy.
The other modules retain their upstream analyzers and thresholds. This is a
418-case selected CPU regression suite, not the entire repository CPU/device CI.

The job initially failed only in the serial production-kernel fixture: its
standalone main omitted the non-MPI rank initialization used by normal AthenaK
main. Adding `my_rank=0; nranks=1` fixes that test-harness error without changing
the solver. A mistakenly submitted rerun, `2272603`, was canceled promptly when
its argument-forwarding harness edit failed to apply.

The corrected fixture job `2272607` completed with exit 0 in 30 seconds. All ten
modes passed in serial and under MPI at 1, 2, and 8 ranks (40 cases total):
`legacy`, `cascade`, `energy`, `nan`, `soft_floor`, `mask3d`, `rk3`, `deep`,
`limit`, and `invalid`. The final two require a nonzero exit and the specific
`FOFC_EXHAUSTED` diagnostic. The legacy mode confirms the old single-pass defect,
and iterative modes repair the constructed cascades while preserving mass and
both saved RK stage registers.

Python lint passed for both new regression modules. Focused C++ lint passed
with the existing compact one-line-if style category excluded; this is not a
claim that full upstream CI lint or device tests have passed.

## Integration CGM and performance gates (submitted)

Three independent nine-node jobs were submitted after successful fixture job
`2272607`. Each uses 192 MPI ranks per node, one thread per rank, the canonical
AthenaK environment, and the site VAST preload with wrapped `mpirun`.

| Job | Configuration | Target |
| --- | --- | --- |
| `2272609` | Fresh single-pass, verbose diagnostics on | 600 Myr or identified failure |
| `2272610` | Fresh eight-pass, verbose diagnostics off | 3000 Myr; inspect 600 Myr milestone |
| `2272611` | Paired caps 1/8, diagnostics/output off | Fixed 100-step timing comparison |

CGM physics and numerics are unchanged: Newtonian MHD, 264 x 264 x 528 cells,
22 x 22 x 44 MeshBlocks, RK3/WENOZ/HLLD, FOFC, four ghost cells, CFL 0.6,
`tfloor=0.014494`, and the existing compressive forcing and other floors.
History, field/forcing, and restart output intervals remain 0.1, 25, and 500 Myr.
The long run starts fresh rather than restarting because the old restart does
not preserve the per-component OU forcing memory.

The benchmark restarts both caps from the same 500 Myr checkpoint, executes
exactly 100 steps per case, and disables verbose diagnostics and all four output
streams. It uses one warm-up pair followed by six measured pairs, alternating
execution order. Each case must report 172800 MeshBlock-cycles and normal cycle-
limit termination. Compare paired simulation timers, not allocation duration.
Its restart behavior is suitable for a matched cost comparison, not exact
reproduction of the fresh CGM trajectory.

Do not merge on the basis of submission alone. Record actual completion,
failure diagnostics, finite history checks, measured overhead, and GitHub CI/
review status before treating this as a completed release validation.

## Interpretation limits

The original pre-integration 600 Myr success is documented in
`2026-09-06-iterative-fofc.md`. It demonstrates avoidance of the original crash
on the iterative trajectory, not unconditional stability for every CGM state.
Finite positive-density states can still require floors. The magnetic predictor
is not the exact later corner-EMF CT update, and source terms act later.

All ranks still execute the configured number of neighbor communication rounds,
even when locally idle; only correction/recheck work is skipped when unchanged.
Any production-overhead statement must come from diagnostics-disabled timing.
