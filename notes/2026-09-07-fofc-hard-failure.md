# Integrated CGM hard failure and multidimensional CFL investigation

## Confirmed integrated failures

Both full-resolution runs at CFL 0.6 failed:

- Iterative cap 8, job `2272610`: 312.866927021256629 Myr, cycle 5222,
  RK3 stage 2, rank 744; `FOFC_EXHAUSTED rounds=8 new=0 hard=2`.
- Single-pass baseline, job `2272609`: 433.228808796411954 Myr, cycle 6663,
  stage 1, rank/gid 939, k=13,j=15,i=7. High-order density
  `0.00388218117209932628` becomes `-0.00503456084720069891` after shared-face
  correction. The cell was unflagged; its positive-y neighbor was flagged.

The baseline therefore reproduces the original uncorrected-neighbor mechanism.
The iterative failure is different: its last local recheck found two already-
flagged cells still nonfinite or nonpositive-density. Verbose diagnostics were
disabled, so the initial log does not identify the cells or which hard criterion
failed. `rounds=8` records the configured cap, not eight active local rechecks.

These trajectories are not identical: saved histories first differ near
10.524 Myr, and by cycle 5000 their times are 326.002 versus 302.258 Myr.
Initial saved history values agree, and the driver initializes its RNG with a
fixed seed. Additional flux corrections and subsequent timestep/evolution
differences prevent comparing late-time failures as tests of an identical state.
The exact contribution of reduction round-off to early divergence is unmeasured.

The earlier pre-main-integration 600 Myr success remains valid evidence for that
revision and trajectory, but is not a general stability guarantee. Do not merge
the development branch based on it alone.

## Paper guidance and timestep hypothesis

Stone et al., AthenaK, section III, equation (3) and the following paragraph:
https://arxiv.org/html/2409.16053v1#S3
The authors generally use CFL <= 1/Ndim, i.e. <= 1/3 in three dimensions.
The user suggested CFL 0.3, which follows this guidance. CFL 0.6 does not.

The code currently selects the minimum directional crossing time, not the
inverse sum of directional rates. In contrast, the multidimensional LLF density
update sums flux divergences over every active direction. A locally corrected
cell can therefore remain inadmissible even when each directional Courant
number appears acceptable. Frozen stage registers, primitives, and LLF face
fluxes do not change when the same correction is repeated: enlarging the
iteration cap cannot repair an already fully corrected cell by itself.

This is a candidate mechanism for the CGM failure, not yet a confirmed diagnosis
of those two cells. Source-driven changes to wave speeds, primitive/conserved
consistency, face-centered versus cell-centered B, and implementation errors
must still be checked from the actual failing stage.

## Failure-only snapshot implementation

`src/mhd/fofc_diagnostics.cpp` is invoked only after the existing iterative abort
condition is reached. It does not alter fluxes, states, or the timestep, and uses
no MPI collective. It records the last active local recheck separately from the
cap, then writes a bounded snapshot of up to 32 hard-invalid cells per failing
rank. Host mirrors are created only on the fatal path.

Each cell includes a 3x3x3 stencil of U0/U1/primitives/Bcc and flags, six face
normal fields, corrected conservative fluxes/EMFs, independent production-LLF
recomputations and wave speeds, the high-order trial conserved state, and the
final trial state. A scalar density budget reports the summed LLF draining rate,
neighbor inflow, and independently reconstructed density.

No generic positivity guarantee is claimed. The magnetic trial remains the
existing estimated update rather than exact later CT. Snapshot host arithmetic
can differ from device arithmetic at round-off; any discrepancies require
checking tolerances and the stored face values, not assuming a solver failure.

## Controlled 3D RK3 density test: passed

New fixture modes `cfl3d_safe` and `cfl3d_unsafe` use a 32^3 periodic mesh with
16^3 blocks, zero B and velocity, unit sound speed, and an admissible density
pulse at a block/rank corner. The distinct saved RK3 stage-2 densities at the
center are rho(U0)=1 and rho(U1)=0.01; all neighboring densities are 0.01.
The weighted center density is 0.2575 before the flux update. All corrected
faces use the actual production LLF solver.

- dt=0.6: predicted density = 0.2575 - 0.25*0.6*3*0.99 = -0.188.
  Iterative FOFC aborts with a hard-invalid, already-flagged cell.
- dt=0.3: predicted density = 0.03475. The correction returns successfully,
  all checked densities are positive, and total mass is conserved.

The actual directional timestep routine returns 1 for these primitive states.
The snapshots independently reproduce the negative density and LLF fluxes.
Job `2273137` passed both modes in serial and MPI at 1/2/8 ranks: eight tests.
This demonstrates the mechanism on prescribed valid stage states, not a full
time-evolved CGM reproduction or proof that every stage at CFL 0.3 is safe.

## Fresh CGM replays submitted

All files are under
`/scratch/jbeattie/cgm_fofc_main_integration/hard_failure_debug`.

- Snapshot validation job `2273040`: all 40 existing serial/MPI fixtures passed;
  deliberate hard failures produced complete snapshots.
- Job `2273100`: fresh cap-8 CFL-0.6 diagnostic replay, target 600 Myr,
  two-hour allocation. Running at the last check.
- Job `2273136`: fresh cap-8 CFL-0.3 comparison, target 600 Myr,
  four-hour allocation. Submitted independently to run in parallel.

Both CGM runs use the same staged snapshot binary, SHA256
`08b0cd25523b137faf9a47f150e90fbec4067d10c79795eabc83076ac4125476`,
built from `a42788d` plus the failure-only snapshot additions. Original staged
binaries and results were not overwritten. The fixture-only CFL extensions
were built separately after staging the CGM binary.

Newtonian ideal MHD; 264x264x528; 22x22x44 blocks; four ghost zones;
RK3/WENOZ/HLLD; cap 8; dfloor=1e-12, pfloor=1e-14, tfloor=0.014494;
same spectral IC and two-component compressive forcing, cooling, gravity, and
thermostat. Both use nine nodes, 192 MPI ranks/node, one thread/rank,
`rrg-essick`, canonical AthenaK environment and VAST-wrapped mpirun.

History output remains at 0.1 Myr. Large field, forcing, and restart outputs
are disabled in both diagnostic replays. They start fresh because the old
restart cannot restore the full per-component OU forcing memory.

Do not submit duplicates. Next: inspect an actual CGM snapshot, compare stored
and recomputed LLF fluxes, distinguish negative density from nonfinite states,
and evaluate the frozen-stage density balance under a reduced timestep. Assess
the full CFL-0.3 trajectory before claiming the model is fixed.

## Existing timing result (not a CFL comparison)

Completed job `2272611` used caps 1/8 at the same CFL, six measured alternating
100-step pairs with diagnostics/outputs disabled. Raw timings remain in
`/scratch/jbeattie/cgm_fofc_main_integration/timings.txt`. This measures per-step
iteration overhead, not the total cost of halving CFL. Do not conflate them.

## Publication and filesystem status

`origin/iterative_FOFC` currently contains `a42788d`. GitHub PR creation previously
failed with HTTP 403 (`Resource not accessible by integration`); no PR was
created and main was not changed. These new diagnostic changes are local until
explicitly recorded/published. The shared dirty `/home/jbeattie/athenak` checkout
remains untouched.

Old inactive debugging runs have been moved under `/scratch/jbeattie/cgm_archive`;
consult its README and HOME_FILES.md for historical path mappings. No original
crash evidence or completed validation dataset was deleted.
