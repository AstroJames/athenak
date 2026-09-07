# Iterative FOFC development — 2026-09-06

Branch: `iterative_FOFC`, based on the diagnosed checkout `0b420af`.
The remote main history was fetched for inspection, but was not merged into this branch.
The user confirmed that no iterative implementation had previously been started.

## Reproduced Newtonian failure

The decisive log is
`/scratch/jbeattie/cgm_mhd_crash_fofc_postdiag_fresh/shell_2194403.out`.
At cycle 6661, time 526.133867756735867, dt 0.0417351350575759852,
RK stage 3, rank/gid 391, local k=11, j=13:

| Cell i | High-order predicted density | Density after one FOFC pass | Original flag |
| --- | --- | --- | --- |
| 6 | 2.16267932701789345 | -0.101725270754894126 | false |
| 8 | 2.12098499036782462 | -0.122475504883282849 | false |

The central cell i=7 was flagged. Replacing its two shared faces with first-order
LLF fluxes made the neighbors negative. The original algorithm cleared its mask
without testing the modified updates. Primitive recovery subsequently floors density
while retaining momentum, permitting enormous velocities.

This establishes a failure of the single-pass correction logic. It does not establish
that an HLLD formula is wrong. The reason initially flagging i=7 remains unresolved.
Previous checks of unselected HLLD intermediate states were not causal evidence.

## Implementation

Set the following in the input to enable the experimental path:

```text
<mhd>
fofc = true
fofc_max_iterations = 4
fofc_diagnostics = false
```

The default iteration limit is 1, retaining the original single-pass algorithm.
The existing explicit nonfinite trial-state detection is retained for Newtonian
ideal MHD. HLLD diagnostic aborts are retained behind the compile-time macro
`ATHENAK_HLLD_DIAGNOSTICS`, and are not enabled in the test executable.

For limits greater than one:

1. Run the original high-order predictor and floor/nonfinite test.
2. Exchange the owning cells' correction flags into one ghost layer, including
   faces, edges, and corners.
3. Apply the existing first-order LLF replacements, including their face EMFs.
4. Reconstruct the modified trial state for flagged cells and their face neighbors.
5. Mark previously unflagged cells that fail this check; retain every earlier flag.
6. Repeat for a fixed maximum number of rounds, then clear the stage's mask.

The mask is immutable while neighboring flags are being read. A predictor scratch
component holds newly discovered flags until a separate merge kernel. No new
volume-sized state array is required.

Each round exchanges unsigned-byte masks with neighboring blocks on a dedicated
MPI communicator. Same-rank neighbors are copied on device. Remote masks are staged
through host buffers, so this implementation does not require GPU-aware MPI.
There is no per-round global convergence reduction. There are fixed neighbor
exchanges even on idle ranks: dropping an idle rank early would lose corrections
propagating into that rank in a later round. Cell work is skipped if neither local
nor incoming flags changed. Within active rechecks, state reconstruction and
primitive conversion are conditional on a corrected face touching the cell;
the current mask traversal itself is still a dense scan.

A cell already fully corrected can continue to require a configured floor.
The normal EOS floor policy remains for finite states with positive density.
Nonfinite states or nonpositive density still present at the iteration limit
cause a diagnostic abort. Newly discovered unprocessed flags at the limit also
cause an abort, including floor-only flags. The implementation never silently
declares such a truncated correction sequence converged.

An additional guard in actual primitive recovery stops on nonfinite conserved
states, nonpositive density, or nonfinite recovered primitives when iterative
FOFC is enabled.

## Scope and remaining limitations

The initial implementation accepts Newtonian ideal MHD on a uniform mesh with one
MeshBlockPack per rank. It rejects iterative mode with passive scalars, explicit
diffusion, shearing boxes, or relativistic physics. These restrictions concern the
new iterative path only.

The recheck deliberately uses the existing cell-centered magnetic predictor with
corrected face EMFs. This is not identical to the later CT update, which uses
corner EMFs. Source terms also occur after FOFC. The new loop is therefore not a
proof of positivity of the final multidimensional MHD update or of source terms.
The accepted-state guard diagnoses remaining failures; it does not repair them.
There is no timestep retry or mass source in this patch.

The iteration limit and communication cost need measurement at the target
resolution. Do not claim a production overhead percentage from the unit tests.
The previously discussed 1–2% target is a target, not a measured result.

## Validation so far

Build: `build_turb_cgm_kokkos`, PROBLEM=turb_cgm, MPI, Kokkos Serial,
KokkosFFT/FFTW, double precision, Release. Source
`/home/jbeattie/.env/athenak_env` before compilation.

The standalone regression links against the actual production objects:

```bash
cmake --build build_turb_cgm_kokkos -j 8
bash tst/unit/build_fofc_test.sh build_turb_cgm_kokkos
```

The build helper uses the existing Makefile-generator build and assumes source
and build paths do not contain whitespace. Execute the resulting
`build_turb_cgm_kokkos/src/fofc_cascade` only under Slurm on Trillium.

Six modes passed on 1, 2, and 8 MPI ranks (18 combinations):

- `legacy`: exactly two negative neighbors survive a single pass.
- `cascade`: iterative correction repairs the density cascade and conserves mass.
- `energy`: iterative correction repairs an energy-only cascade.
- `nan`: a deliberately nonfinite input flux is flagged and replaced.
- `soft_floor`: a temperature floor that first-order transport cannot remove
  does not cause an endless correction loop.
- `mask3d`: all face, edge, and corner ghost flags match the owning periodic cells.

For multi-rank cascade tests the initial flag lies directly left of a rank
boundary. The tests also verify that neither conserved stage register nor the
primitive state is modified by FOFC. An `invalid` mode correctly aborts with
`FOFC_EXHAUSTED` if even first-order fluxes cannot repair a negative density.

The full reviewed suite passed in debug allocation **2265589**; the earlier
pre-optimization version passed in **2265528**. Test runtime files are in
`/scratch/jbeattie/cgm_iterative_fofc_development`.

## High-resolution comparison

Job **2265607** compared single-pass and four-pass runs from the finite t=500
restart to t=510, using the same executable and the original 264x264x528 mesh.
It uses 9 full CPU nodes, 192 ranks per node, account rrg-essick,
the normal AthenaK loader, and the VAST-wrapped mpirun.

Numerics: RK3, WENOZ, HLLD, CFL 0.6, nghost=4, blocks 22x22x44,
dfloor=1e-12, pfloor=1e-14, tfloor=0.014494. The original forcing, gravity,
cooling, thermostat, and boundary conditions are retained. Only history output
is enabled (dt=0.1); field, force, and restart writes are disabled for this
short comparison.

Directories:
- `/scratch/jbeattie/cgm_iterative_fofc_development/baseline_t500_v2`
- `/scratch/jbeattie/cgm_iterative_fofc_development/iterative4_t500_v2`

AthenaK's command-line parser cannot introduce a new parameter absent from a
restart header. The job therefore loads the small supplemental
`iterative_options.input` using `-i`, alongside `-r`.
The corrected invocation passed a parameter-only check in debug job **2265605**.
The initial comparison job **2265536** stopped at parameter parsing and produced
no evolution; it is not a failed numerical test.

The t=500 restart does not reproduce the original forcing trajectory exactly:
per-component OU forcing memory was not saved. A fresh high-resolution run
through the original failure time is still needed after the short comparison.

### Completed comparison and cap scan

Job **2265607** finished with a diagnostic abort in the four-pass case.
The baseline completed t=510 (cycle 6487). The four-pass case stopped at cycle
6408, t=501.222301503750032, RK stage 3, rank 391:

```text
FOFC_EXHAUSTED ... rounds=4 new=1 hard=0
```

This is a newly discovered, unprocessed floor flag at the cap, not a reported
nonfinite/nonpositive-density state on the aborting rank. Four passes are
insufficient even for this short interval under the current floor policy.

Diagnostic cap-scan job **2265889** repeated t=500 to t=510 with caps 4, 8,
and 16, on 9 full nodes. The four-pass abort reproduced. Both 8 and 16 completed
t=510 at cycle 6487. Their per-pass logs contained no hard-bad events; new flags
appeared as late as pass 4, requiring at least a fifth correction pass.
The logs do not distinguish which configured floor produced each soft flag.
This interval therefore tests convergence of floor-triggered correction,
not repair of the original fresh-run negative-density event.

The cap has no additional hard-coded upper bound: any positive input integer is
accepted. It counts total correction passes per RK stage, including the first.
Larger caps still incur fixed neighbor-exchange rounds after local cell work
stops. Eight is the smallest successful cap tested, not a proven minimum or a
guarantee for the full evolution.

### Fresh validation submitted

Job **2265910** was submitted with an afterok dependency on the successful cap
scan 2265889. It runs a fresh single-pass baseline followed by a fresh eight-pass
case, each from t=0 to t=600, beyond the original failure at t=526.13.
A baseline diagnostic abort does not prevent launching the iterative case.
Both use the same staged executable and identical physics/forcing inputs.
The binary contains the code committed as **1a7d19f**, but was staged before
the commit was made, so embedded build metadata need not show that commit.

Both retain the original Newtonian CGM setup, 264x264x528 mesh, 22x22x44 blocks,
RK3/WENOZ/HLLD, CFL 0.6, four ghost cells, original floors, gravity, cooling,
thermostat, magnetic initialization, and two-component forcing. No mass source
or additional floor was introduced. Diagnostics are enabled. History cadence
is 0.1 Myr, MHD/forcing binary cadence is 25 Myr, and restart cadence is 500 Myr
(plus the code's initial/final outputs).

Resources: account rrg-essick, compute partition, 9 nodes, 192 MPI ranks/node,
one thread/rank, walltime 02:30:00; normal AthenaK and VAST loaders and wrapped
mpirun. Runtime files are under the existing development scratch directory:

- `cap_scan_job.sh`, `iterative{4,8,16}_t500_diagnostics/`
- `fresh_validation_job.sh`, `fresh_validation.input`
- `fresh_baseline_t600/`, `fresh_iterative8_t600/` (created when launched)

### Fresh validation completed

At the user's request, the two cases were split to run concurrently.
The baseline's batch shell alone was sent SIGSTOP, leaving its MPI children
integrating; this prevented the original sequential second launch.
Independent iterative job **2265958** used nine other full nodes, the same
staged executable/input, cap 8, and directory
`fresh_iterative8_parallel_t600/`. Its walltime limit was 01:30:00.

The fresh single-pass baseline reproduced the original failure EXACTLY:
cycle 6661, t=526.133867756735867, dt=0.0417351350575759852, RK stage 3,
rank/gid 391, k=11, j=13, i=6 and i=8. Both high-order and post-correction
densities match the earlier diagnosed failure to all printed digits.
Its MPI step ended on 2026-09-06 at 16:22:01. The paused batch allocation was
then intentionally cancelled and released at 16:26:49. The scheduler's
CANCELLED status records that cleanup, not the numerical failure classification.
No duplicate iterative simulation was launched.

The fresh eight-pass case **COMPLETED**, exit 0, at t=600, cycle 9575.
Job 2265958 ran 2026-09-06 16:01:29 to 17:04:14 (01:02:45).
Its stderr is empty. Reported simulation CPU timer: 3739.419 s; throughput:
9.422724e7 zone-cycles/s. These are not a clean overhead comparison: the
baseline failed earlier, the trajectories/step counts differ, and diagnostics
were enabled. No production overhead percentage is inferred from these totals.

This validates removal of the original failure for this high-resolution CGM
case through t=600, retaining Newtonian RK3/WENOZ/HLLD and the original floors
and source terms. It is not a stability guarantee through the original t=3000
target or for every configuration.

The 141 MB iterative log was aggregated successfully in compute-node debug
allocation **2270848** (exit 0, 13 seconds); the login node only read the small
summary. Analysis script: `analyze_validation.sh`; output:
`validation_summary.out`.

Aggregate results:

- 9575 completed timesteps, 28725 RK stages, final time 600.
- 3230544 newly flagged physical cell-stage events, all floor-only.
- Zero FOFC_ITER_BAD records, zero hard-bad pass reports, zero exhaustion or
  accepted-state abort markers.
- New flags appeared as late as pass 5, implying a sixth correction pass was
  required. The cap of 8 was sufficient. A separate cap-6 replay was not run.
- Inferred minimum correction depths by RK stage: 1 pass: 190 stages;
  2: 11434; 3: 16018; 4: 1032; 5: 46; 6: 5; 7/8: 0.
  These count correction depth, not MPI rounds: all 8 scheduled exchanges
  still occur in the current implementation.
- Iterative history: 6001 data rows, t=0 through 600, zero NaN/Inf fields,
  minimum recorded dt=0.00681843.
- Baseline history: 5244 rows, last saved time rounded to 526.134,
  zero NaN/Inf fields before its diagnostic abort.

Interpretation: the full CGM experiment demonstrates avoidance of the original
failure on the changed iterative trajectory. It did NOT log the same two
negative-density states and then repair them: no hard-bad trial state occurred
in its recheck logs. Direct repair of correction-induced nonpositive density
remains demonstrated by the controlled regression tests, including MPI
boundaries. Preserve this distinction when describing the evidence.

All three allocations (baseline, independent iterative run, and analysis)
have ended and been released. No further simulations were submitted.
The current source worktree contains unrelated concurrent user changes; these
were not touched. The validation used the already-staged executable containing
the code committed as 1a7d19f, not a rebuild of those later edits.
