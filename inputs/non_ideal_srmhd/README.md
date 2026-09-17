# Non-ideal SRMHD campaign inputs

Reconciled on 2026-09-17 against `relativistic_viscosity` at `973840bc`.
This directory contains **42 committed input definitions**. Their presence does
not establish that a run completed, converged, or used the current solver.
The tables below describe the committed files, not overrides used on Trillium.
No input or solver settings were changed during this documentation reconciliation.

## Scope, provenance, and numerical method

| Family | Files | Committed mesh | MeshBlock cells | Number of MeshBlocks |
| --- | ---: | --- | --- | ---: |
| KHI | 2 | 4096 x 8192 | 128 x 128 | 2048 |
| Decaying turbulence | 4 | 8192 x 8192 | 256 x 256 | 1024 |
| Mechanical cooling scan | 12 | 256 cubed | 32 cubed | 512 |
| Uncooled antenna benchmark | 1 | 256 cubed | 32 cubed | 512 |
| Reconnection | 7 | three levels below | 256 x 256 | 512, 2048, or 8192 |
| Native driving regimes | 12 | 256 cubed | 32 cubed | 512 |
| Small-scale dynamo | 4 | 256 cubed | 32 cubed | 512 |

All files explicitly select special relativity, ideal-gas EOS with `gamma=4/3`,
WENO-Z, LLF, FOFC, CFL 0.4, and four ghost cells. The ideal KHI control uses
SSPRK(3,3), `integrator=rk3`. All other files use IMEX-SSP3(4,3,3),
`integrator=imex3`, with evolved cell-centred electric fields
(`electric_ct=false`). Reconnection has no viscosity; the other non-ideal
families enable relativistic viscosity. Its relaxation time is 0.2 for KHI
and 0.02 for the other viscous families. These relaxation terms are evolved
by IMEX, not explicit RK3. The 3D boxes and KHI/decay domains are periodic;
reconnection uses outflow in its two active directions.

For every executed run, retain its solver commit, effective input including
command-line overrides, mesh/decomposition, restart lineage, and output headers.
A filename containing `256`, `8192`, or `4096x8192` is not proof of the actual
mesh. Ordinary primitive binaries do not contain the complete evolved viscous
state; preserve restart files when that state is needed.

The audited baseline includes the IMEX3 stiff-history correction `8c91655a`
and moving-equilibrium regression `973840bc`. Earlier results must be identified
by the solver that produced them; updating an input's documentation cannot
retroactively validate them. This reconciliation did not rerun production cases
or certify GPU/MPI accuracy or statistical convergence.

## 1. Long-time KHI

The committed ideal and viscous `khi_paper_*_4096x8192_t12.athinput` files use
`[-0.5,0.5] x [-1,1]`, `tlim=12`, `shear_four_velocity=1`,
`shear_width=0.05`, `density_contrast=1`, `pressure=10`, perturbation 0.01,
and perturbation width 0.2. Both start with zero electromagnetic fields.
The viscous file selects `shear_viscosity=1e-4`,
`shear_relaxation_time=0.2`, and **uniform resistivity 1.0**. Its additional
`eta_floor` parameter does not replace that uniform value. This is an
unmagnetized viscous-fluid comparison carried through the resistive MHD state,
not the FLASH magnetic KHI dynamo setup.

Histories have cadence 0.01, `mhd_w` binaries 0.2, and restarts 1.0 in code
time. At this mesh, a five-field single-precision payload is 640 MiB per
snapshot, or 38.125 GiB for 61 snapshots including the initial state, excluding
headers, restarts, and final profiles. The final profile is controlled by
`problem/viscous_profile_name`; use `none` for intermediate wall-time segments
if that profile is unwanted, and restore the desired name for the final segment.

**Executed-run distinction.** Local archived input files for the later low-shear
pair instead specify `4032 x 8064`, `168 x 168` MeshBlocks (1152 blocks),
`shear_four_velocity=0.2`, and the same RK3/IMEX3 split. Their dataset identifiers
are `khi_paper_ideal_4032x8064_t12_u0p2_20260825` and
`khi_paper_viscous_4032x8064_t12_u0p2_20260826`. These external run inputs were
inspected during reconciliation; they are not distributed here. The branch's
unit-four-velocity files do not reproduce those replacements. Keep the older
high-shear and replacement low-shear datasets separate even where a remote
run directory was reused. No choice between those physical setups is made here.

## 2. Decaying-turbulence magnetic-Prandtl scan

The four `decaying_turbulence_re50_pm*_8192.athinput` files use a unit square,
`rho0=p0=1`, initial velocity and magnetic RMS parameters 0.15, and
`nu_sh=0.0012`. Both spectral initializers use the `parabolic` option over
`nlow=1`, `nhigh=4`, with velocity seed 1837 and magnetic seed 918273.
There is no driving or cooling. Using the nominal reference
`L0=1/2.5=0.4` and `v0=0.15` gives `Re=L0*v0/nu_sh=50` and
`t0=L0/v0=8/3`; `tlim=2` is `0.75 t0`.

| Pm tag | Uniform eta | Nominal Rm |
| ---: | ---: | ---: |
| 10 | 1.2e-4 | 500 |
| 100 | 1.2e-5 | 5000 |
| 1000 | 1.2e-6 | 50000 |
| 10000 | 1.2e-7 | 500000 |

History cadence is 0.02; `mhd_w_bcc` binaries and restarts have cadence 0.5.
An eight-field payload is 2 GiB per snapshot, or 10 GiB for five snapshots,
excluding headers, restarts, and profiles. Actual Reynolds numbers based on
later measured velocities need not equal these nominal labels.

## 3. Mechanical driving and cooling scan

The twelve `mechanical_sigma*_cooling_*_256.athinput` files combine
`sigma0=(0.1,1,10)` with cooling suffixes `none`, `tcool0p1`, `tcool1`, and
`tcool10`. They start at rest with `rho0=1`, `p0=1/16`, uniform `B^z`,
and `beta0=(1,0.1,0.01)`, respectively. For the initially stationary fluid,
`w0=rho0+4p0=1.25`, so `B0^2=(0.125,1.25,12.5)`.

The fully solenoidal mechanical driver uses the isotropic parabola with
`nlow=1`, `nhigh=3`, peak 2, width 0.5, `accel_rms=0.09`, and
`tcorr=3.872983346207417`. Both transport coefficients are
`nu_sh=eta=6.454972243679028e-5`. With nominal `L0=0.5` and
`v0=sqrt(15)/30`, this is `Re=Rm=1000`. The nominal time is
`t0=3.872983346207417`, and the duration is `10 t0`.

Cooling is disabled by `relativistic_cooling=none` in the `none` files even
though a `cooling_time` value is present. The other files use entropy cooling
to `K0=1/16` with `t_cool/t0=(0.1,1,10)`. History, primitive/source, and restart
cadences are `(0.05,0.5,1) t0`. The nominal Reynolds number is a normalization,
not evidence that the physical dissipation range is resolved.

## 4. Uncooled antenna benchmark

[antenna_zhdankin_f0p65_256.athinput](antenna_zhdankin_f0p65_256.athinput)
uses `rho0=1`, `p0=100`, `beta0=1`, and hence
`sigma0=200/401`, `vA0=sqrt(200/601)`, `tA0=1/vA0` in the unit cube.
Its **committed nominal Re=Rm is 1000**, with
`nu_sh=eta=9.181167136334355e-5`, normalized using `Ldrive=1/(2*pi)`.
This differs from the older Re=50 calibration referred to in the original notes.

The input selects `zhdankin8`, `frequency_model=zhdankin2018`,
`frequency_factor=0.6`, `decorrelation_factor=0.5`, balanced amplitude fractions
0.65, stationary initialization, and seed 210989. Cooling is absent.
The run duration is `6 tA0`; history, primitive/current, and restart cadences
are `(0.02,0.1,1) tA0`. The input's source amplitude alone does not establish
an achieved fluctuation amplitude.

## 5. Uniform and charge-starvation reconnection

The seven `reconnection_*.athinput` files use `rsrmhd_reconnection` on
`[-2,2] x [-1,1]`, with outflow in both active directions and `tlim=2`.
They specify field 1, guide field 0, sheet width 0.02, upstream density 0.1,
sheet density contrast 3, upstream pressure 0.001, and localized pressure
perturbation 0.15. Pinch steepness and along/across offsets are `(200,10,2)`.
The exact initial profile is implemented in
[rsrmhd_reconnection.cpp](../../src/pgen/tests/rsrmhd_reconnection.cpp).
The perturbed initial state should not be described as an exact equilibrium.

Three resolutions pair uniform `S=1e5` with charge starvation. A fourth uniform
file has `S=2e5` on the intermediate grid. The filename normalization is
`S=2*sqrt(10/11)/eta`, based on cold magnetization 10 and half-length 2;
it is not a measurement of the evolved outflow speed or enthalpy magnetization.

| Grid | Cells per unit length | MeshBlocks | Cells per width 0.02 | Cells per scale 0.001 |
| --- | ---: | ---: | ---: | ---: |
| 8192 x 4096 | 2048 | 512 | 40.96 | 2.048 |
| 16384 x 8192 | 4096 | 2048 | 81.92 | 4.096 |
| 32768 x 16384 | 8192 | 8192 | 163.84 | 8.192 |

Uniform `S=1e5` uses `eta=1.9069251784911845e-5`; `S=2e5` uses half that value.
Charge starvation selects `eta_scale=1e-4`, `eta_floor=1.9069251784911845e-8`,
and `number_per_mass=1`. The implementation in
[resistivity_model.hpp](../../src/mhd/resistivity_model.hpp) uses

```text
E_star = Gamma [E + v cross B - (E dot v) v],
n_lab = number_per_mass * Gamma * rho,
eta = max(eta_floor, eta_scale * norm(E_star) / n_lab).
```

Here E and B denote spatial vectors and the norm is the Euclidean spatial
norm used by that routine. The floor is a maximum, not an additive term.
The historical `delta1e3` label is not a separate runtime parameter.

History cadence is 0.01. Binary outputs at cadence 0.1 contain `mhd_w_bcc`,
`mhd_e3`, `mhd_eta`, and `mhd_jz` (the last is the code identifier for `J^z`).
Restarts have cadence 0.25. The eleven-field payload totals 5.5 GiB per output
at `16384 x 8192` and 22 GiB at `32768 x 16384`, excluding headers and restarts.
For 21 outputs these are 115.5 and 462 GiB per case.

Use the grid ladder to measure changes in reconnection rate, outflow, density,
and electric/current diagnostics. Resolution counts alone do not establish
convergence. The older notes' assumed factor-of-two WENO-Z numerical-resistivity
reduction and literature convergence thresholds were not verified for this
implementation in this audit and are not acceptance criteria here. In particular,
`eta*abs(J^z)/B0` and `abs(E^z)/B0` should not be assumed identical throughout a
moving relativistic plasma; specify where and under which approximations the
comparison is made.

## 6. Native antenna and mechanical driving regimes

The twelve `antenna_sigma*_{weak,strong}_256.athinput` and
`mechanical_sigma*_mach{0p1,10}_256.athinput` files have the same three initial
magnetizations, but different fluid states, forcing processes, and clocks.
They are not a controlled change of coupling alone. See
[DRIVING_COMPARISON_PLAN.md](DRIVING_COMPARISON_PLAN.md) for the audited
amplitudes, times, cooling, and proposed controlled comparison.

In particular, the committed weak antenna cases run for **100 tA0**, not 10;
strong cases run for 10. Mechanical weak/strong correlation times are **100/1
code-time units**, respectively, and their durations are **1000/10**. The old
common-clock description does not apply to these files.

## 7. Small-scale dynamo

The four `small_scale_dynamo_re200_rm*_256.athinput` files start at rest in a
unit cube with `rho0=p0=K0=1`, no guide field, and `gamma=4/3`. Define nominal
`cs0=sqrt(4/15)`, target `v0=0.1*cs0`, `L0=0.5`, and
`t0=L0/v0=9.682458365518542`. These reference values specify the inputs;
entropy cooling does not guarantee a spatially uniform sound speed or an
achieved Mach number of 0.1.

The isotropic mechanical source is fully solenoidal, with parabola peak 2,
width 0.5, `nlow=1`, `nhigh=3`, `tcorr=t0`, and `accel_rms=0.0115`.
Entropy cooling uses `K0=1` and `t_cool=0.01 t0`. The duration is `10 t0`;
history, primitive/magnetic, and restart cadences are `(0.025,0.25,1) t0`.

| Rm tag | Nominal Pm | Uniform eta |
| ---: | ---: | ---: |
| 200 | 1 | 1.2909944487358055e-4 |
| 400 | 2 | 6.454972243679028e-5 |
| 800 | 4 | 3.227486121839514e-5 |
| 1000 | 5 | 2.5819888974716114e-5 |

All four use `nu_sh=1.2909944487358055e-4`, giving nominal `Re=200`.
The random magnetic initializer uses `driving_parabolic`, peak 2, width 0.5,
`nlow=1`, `nhigh=3`, and seed 918273. With `sigma_rms=1e-6` and stationary
enthalpy density `w0=5`, the intended normalization is
`<B^2>=5e-6` and magnetic energy `2.5e-6` in the unit volume, up to numerical
roundoff. See
[rsrmhd_decaying_turbulence.cpp](../../src/pgen/tests/rsrmhd_decaying_turbulence.cpp)
for the initialization and history definitions.

Local archived runs include actual `252^3` inputs and histories explicitly
named `*.pre_imex3_fix.user.hst`. Do not pool those with corrected-solver data.
The project record flags continuing magnetic decline after the solver repair;
this audit does not resolve its cause or establish a saturated dynamo. Measure
realized velocities, inspect the corrected energy budget, and identify a
kinematic interval before fitting growth. Endpoint `q_ohm` alone is not a
closed coordinate-frame electromagnetic energy budget. The moving-equilibrium
regression checks a distinct endpoint issue; it does not establish production
saturation or spatial convergence.

## Reconciliation record and remaining evidence

The earlier documents were preserved in commit `6876a891` on `feature/teno6`.
Their statements were compared with all 42 input files and the referenced
implementation at `973840bc`. Automated checks passed 443 numerical
comparisons covering documented parameters, clocks, cadences, inventory end
times, and payload estimates; all document links resolved. This revision
replaces stale specifications;
it does not implement the unperformed experiments in those notes.

| Earlier statement | Reconciled treatment |
| --- | --- |
| KHI 2048 x 4096, 512 blocks | Committed 4096 x 8192, 2048 blocks; external low-shear runs distinguished separately. |
| Uncooled antenna Re=50 | Committed Re=1000; older calibration kept distinct. |
| All native driving cases last 10 Alfvén times | Family- and strength-dependent clocks tabulated in the comparison plan. |
| Native antenna decorrelation factor 0.5 | Weak/strong factors are 0.027566444771089604 / 0.27566444771089604. Only the separate uncooled benchmark retains 0.5. |
| Weak mechanical amplitudes 0.003/0.008/0.011 | Committed value is 0.00005 for every magnetization. |
| Pilot fits, spectral roll-off estimates, and published resolution thresholds imply production acceptance | No such qualification made: the underlying evidence was not revalidated in this reconciliation. |
| Dynamo input describes achieved steady Mach and clean growth | Nominal target distinguished from measured state; solver lineage and unresolved decline retained. |

Local provenance inspected outside Git included the low-shear KHI input copies
and metadata under `trillium_runs/local_data`, plus dynamo inputs and the
pre-fix archive filenames. Full output transfers, run completion, historical
pilot statistics, and literature claims were not re-audited. Record these
separately when preparing scientific results. The original documents remain
recoverable from Git; omitted numerical pilot claims are not silently adopted
as current validation.

For binary storage estimates above, the writer stores each output variable as
`float`; estimates assume four bytes per value, full-domain output, and the
specified number of snapshots. Headers, restarts, profiles, and any extra
segment outputs are additional. See [binary.cpp](../../src/outputs/binary.cpp).

## Complete committed file inventory

The end times below are code-time values read from the files. This list is a
configuration inventory, not a run-completion manifest.

| Input | Mesh | End time |
| --- | --- | ---: |
| [antenna_sigma0p1_strong_256.athinput](antenna_sigma0p1_strong_256.athinput) | 256 x 256 x 256 | 33.166247903553995 |
| [antenna_sigma0p1_weak_256.athinput](antenna_sigma0p1_weak_256.athinput) | 256 x 256 x 256 | 331.66247903553995 |
| [antenna_sigma10_strong_256.athinput](antenna_sigma10_strong_256.athinput) | 256 x 256 x 256 | 10.488088481701514 |
| [antenna_sigma10_weak_256.athinput](antenna_sigma10_weak_256.athinput) | 256 x 256 x 256 | 104.88088481701516 |
| [antenna_sigma1_strong_256.athinput](antenna_sigma1_strong_256.athinput) | 256 x 256 x 256 | 14.14213562373095 |
| [antenna_sigma1_weak_256.athinput](antenna_sigma1_weak_256.athinput) | 256 x 256 x 256 | 141.4213562373095 |
| [antenna_zhdankin_f0p65_256.athinput](antenna_zhdankin_f0p65_256.athinput) | 256 x 256 x 256 | 10.40096149401583 |
| [decaying_turbulence_re50_pm10000_8192.athinput](decaying_turbulence_re50_pm10000_8192.athinput) | 8192 x 8192 x 1 | 2.0 |
| [decaying_turbulence_re50_pm1000_8192.athinput](decaying_turbulence_re50_pm1000_8192.athinput) | 8192 x 8192 x 1 | 2.0 |
| [decaying_turbulence_re50_pm100_8192.athinput](decaying_turbulence_re50_pm100_8192.athinput) | 8192 x 8192 x 1 | 2.0 |
| [decaying_turbulence_re50_pm10_8192.athinput](decaying_turbulence_re50_pm10_8192.athinput) | 8192 x 8192 x 1 | 2.0 |
| [khi_paper_ideal_4096x8192_t12.athinput](khi_paper_ideal_4096x8192_t12.athinput) | 4096 x 8192 x 1 | 12.0 |
| [khi_paper_viscous_4096x8192_t12.athinput](khi_paper_viscous_4096x8192_t12.athinput) | 4096 x 8192 x 1 | 12.0 |
| [mechanical_sigma0p1_cooling_none_256.athinput](mechanical_sigma0p1_cooling_none_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma0p1_cooling_tcool0p1_256.athinput](mechanical_sigma0p1_cooling_tcool0p1_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma0p1_cooling_tcool10_256.athinput](mechanical_sigma0p1_cooling_tcool10_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma0p1_cooling_tcool1_256.athinput](mechanical_sigma0p1_cooling_tcool1_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma0p1_mach0p1_256.athinput](mechanical_sigma0p1_mach0p1_256.athinput) | 256 x 256 x 256 | 1000.0 |
| [mechanical_sigma0p1_mach10_256.athinput](mechanical_sigma0p1_mach10_256.athinput) | 256 x 256 x 256 | 10.0 |
| [mechanical_sigma10_cooling_none_256.athinput](mechanical_sigma10_cooling_none_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma10_cooling_tcool0p1_256.athinput](mechanical_sigma10_cooling_tcool0p1_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma10_cooling_tcool10_256.athinput](mechanical_sigma10_cooling_tcool10_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma10_cooling_tcool1_256.athinput](mechanical_sigma10_cooling_tcool1_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma10_mach0p1_256.athinput](mechanical_sigma10_mach0p1_256.athinput) | 256 x 256 x 256 | 1000.0 |
| [mechanical_sigma10_mach10_256.athinput](mechanical_sigma10_mach10_256.athinput) | 256 x 256 x 256 | 10.0 |
| [mechanical_sigma1_cooling_none_256.athinput](mechanical_sigma1_cooling_none_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma1_cooling_tcool0p1_256.athinput](mechanical_sigma1_cooling_tcool0p1_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma1_cooling_tcool10_256.athinput](mechanical_sigma1_cooling_tcool10_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma1_cooling_tcool1_256.athinput](mechanical_sigma1_cooling_tcool1_256.athinput) | 256 x 256 x 256 | 38.72983346207417 |
| [mechanical_sigma1_mach0p1_256.athinput](mechanical_sigma1_mach0p1_256.athinput) | 256 x 256 x 256 | 1000.0 |
| [mechanical_sigma1_mach10_256.athinput](mechanical_sigma1_mach10_256.athinput) | 256 x 256 x 256 | 10.0 |
| [reconnection_nonuniform_delta1e3_16384x8192.athinput](reconnection_nonuniform_delta1e3_16384x8192.athinput) | 16384 x 8192 x 1 | 2.0 |
| [reconnection_nonuniform_delta1e3_32768x16384.athinput](reconnection_nonuniform_delta1e3_32768x16384.athinput) | 32768 x 16384 x 1 | 2.0 |
| [reconnection_nonuniform_delta1e3_8192x4096.athinput](reconnection_nonuniform_delta1e3_8192x4096.athinput) | 8192 x 4096 x 1 | 2.0 |
| [reconnection_uniform_s1e5_16384x8192.athinput](reconnection_uniform_s1e5_16384x8192.athinput) | 16384 x 8192 x 1 | 2.0 |
| [reconnection_uniform_s1e5_32768x16384.athinput](reconnection_uniform_s1e5_32768x16384.athinput) | 32768 x 16384 x 1 | 2.0 |
| [reconnection_uniform_s1e5_8192x4096.athinput](reconnection_uniform_s1e5_8192x4096.athinput) | 8192 x 4096 x 1 | 2.0 |
| [reconnection_uniform_s2e5_16384x8192.athinput](reconnection_uniform_s2e5_16384x8192.athinput) | 16384 x 8192 x 1 | 2.0 |
| [small_scale_dynamo_re200_rm1000_256.athinput](small_scale_dynamo_re200_rm1000_256.athinput) | 256 x 256 x 256 | 96.82458365518542 |
| [small_scale_dynamo_re200_rm200_256.athinput](small_scale_dynamo_re200_rm200_256.athinput) | 256 x 256 x 256 | 96.82458365518542 |
| [small_scale_dynamo_re200_rm400_256.athinput](small_scale_dynamo_re200_rm400_256.athinput) | 256 x 256 x 256 | 96.82458365518542 |
| [small_scale_dynamo_re200_rm800_256.athinput](small_scale_dynamo_re200_rm800_256.athinput) | 256 x 256 x 256 | 96.82458365518542 |
