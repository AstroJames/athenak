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

Verbose per-cell/per-pass diagnostics are opt-in. They can generate large logs
and should be disabled in production timing measurements. Failure guards remain
enabled independently of verbose diagnostics.

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
