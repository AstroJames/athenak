# Antenna and mechanical forcing: campaign definitions and controlled comparison

Reconciled on 2026-09-17 against `relativistic_viscosity` at `973840bc`.
Sections 1–3 describe the twelve committed native-regime inputs. Section 4
preserves a proposed controlled experiment; it is not an implemented or
completed campaign. The separate uncooled antenna benchmark and mechanical
cooling scan are documented in [README.md](README.md).

## 1. Shared settings and different physical states

All twelve inputs start at rest in a periodic unit cube on `256^3` cells,
with `32^3` MeshBlocks (512 blocks), uniform `B^z`, and `gamma=4/3`.
They use IMEX3, WENO-Z, LLF, FOFC, cell-centred electric fields, uniform
resistivity, relativistic viscosity, and one-sided entropy cooling.
`shear_relaxation_time=0.02`, `shear_chi_max=2`, and `cooling_cfl=0.1`.

For these stationary initial fluids, define `w0=rho0+4p0`,
`sigma0=B0^2/w0`, `vA0=sqrt(sigma0/(1+sigma0))`, and `tA0=1/vA0`.
The coefficients are `nu_sh=eta=0.5*max(vA0,0.5)/1000` in both families.
This gives nominal `Re=Rm=1000` only for the reference length 0.5 and
reference speed `max(vA0,0.5)`; it does not imply Re=1000 based on the
realized velocity in each weak or strong run.

| Sigma tag | vA0 | tA0 | nu_sh = eta |
| --- | ---: | ---: | ---: |
| sigma0p1 | 0.3015113446 | 3.3166247904 | 0.00025 |
| sigma1 | 0.7071067812 | 1.4142135624 | 0.0003535533905932738 |
| sigma10 | 0.9534625892 | 1.0488088482 | 0.0004767312946227962 |

The antenna fluid has `rho0=1`, `p0=K0=100`, with
`beta0=(4.987531172069825,0.49875311720698257,0.04987531172069825)` and
`B0^2=(40.1,401,4010)`. Mechanical inputs instead have `rho0=1`,
`p0=K0=0.0018891687657430734`, initial `cs0=0.05`, and
`beta0=(0.0375,0.00375,0.000375)`. Their guide fields follow
`B0^2=2p0/beta0`.

Differences between these families cannot be attributed solely to current
versus acceleration coupling. Sound speed, spatial modes, temporal process,
amplitude control, and the duration expressed in Alfvén times differ.

## 2. Native antenna inputs

The six `antenna_sigma*_{weak,strong}_256.athinput` files use `zhdankin8`,
`apar_double_curl`, guide axis z, `va_reference=initial_mean`,
`frequency_model=zhdankin2018`, `frequency_factor=0.6`, stationary
initialization, seed 210989, and equal plus/minus amplitude fractions.

In [antenna_driver.cpp](../../src/srcterms/antenna_driver.cpp),
`UpdateModeFrequencies` sets `omega_ref=2*pi/(sqrt(3)*tA0)` for this frequency
model; `AdvanceModeState` uses decay rate
`lambda=decorrelation_factor*omega_ref`. Consequently the **envelope
correlation time** is `1/lambda`; it is not the oscillation period.

| Setting | Weak | Strong |
| --- | ---: | ---: |
| Each amplitude fraction | 0.065 | 0.65 |
| decorrelation_factor | 0.027566444771089604 | 0.27566444771089604 |
| Envelope correlation time / tA0 | 10 | 1 |
| tlim / tA0 | 100 | 10 |
| cooling_time / tA0 | 0.1 | 0.01 |
| History dt / tA0 | 0.25 | 0.025 |
| Primitive/current dt / tA0 | 2.5 | 0.25 |
| Restart dt / tA0 | 10 | 1 |

Both strengths therefore cover ten envelope correlation times, with cooling
at 0.01 of that correlation time. The old statement that all six used
`decorrelation_factor=0.5` and ran for ten Alfvén times was incorrect for
these committed inputs. The separate uncooled benchmark retains 0.5.

The intended regimes are `delta B_rms/B0<1` and order-unity fluctuations.
These are outcomes to measure, not values enforced by an input amplitude.
The old pilot values were not revalidated during this audit. Changes to
correlation times also prevent importing their calibration without checking
which input and solver produced them.

## 3. Native mechanical inputs

The six `mechanical_sigma*_mach{0p1,10}_256.athinput` files use
`relativistic_forcing=mechanical`, `driving_geometry=isotropic`,
`driving_profile=parabola`, `sol_weight=1`, `nlow=1`, `nhigh=3`,
`parabola_peak=2`, and `parabola_width=0.5`. This gives compact amplitude
support between mode magnitudes 1.5 and 2.5; the endpoints have zero amplitude.

| Setting (code-time units where applicable) | Mach-0.1 target | Mach-10 target |
| --- | ---: | ---: |
| tcorr | 100 | 1 |
| tlim | 1000 | 10 |
| cooling_time | 1 | 0.01 |
| History dt | 2.5 | 0.025 |
| Primitive/force dt | 25 | 0.25 |
| Restart dt | 100 | 1 |

| Sigma tag | Weak accel_rms | Strong accel_rms |
| --- | ---: | ---: |
| sigma0p1 | 0.00005 | 1.25 |
| sigma1 | 0.00005 | 1.5 |
| sigma10 | 0.00005 | 1.25 |

These clocks are based on the nominal target eddy time `L0/v_target` with
`L0=0.5`, `cs0=0.05`, and `v_target=(0.005,0.5)`. Each duration is ten
nominal eddy times. The former weak amplitudes `(0.003,0.008,0.011)` and
`tcorr=sqrt(3)/pi*tA0` are not the committed settings.

The source in [turb_driver.cpp](../../src/srcterms/turb_driver.cpp) uses a
non-oscillating OU update with `exp(-dt/tcorr)` and initializes its RNG with
`rstate.idum=-1`. The spectral-initial-condition seed is not a mechanical
forcing seed. Matching a random seed would not by itself match the antenna's
oscillating coefficient process. The forcing draws also depend on timestep
history, so a CFL comparison needs a replayed physical-time realization or an
ensemble design if stochastic-path differences are to be controlled.

The Mach tags specify targets, not measured steady states. Measure the cooled
sound speed and velocity. Retain the existing high-sigma strong amplitude
1.25 until a separate calibration/solver study supports changing it. The
old note's report of failure at amplitude 2 is historical, not a new stability
boundary established here. A sub-target Mach measurement alone does not
prove physical magnetic suppression; numerical and forcing effects remain
possible explanations.

## 4. Proposed controlled comparison — not implemented by these inputs

The scientific objective is to isolate turbulence driven through an external
four-current from turbulence driven through a mechanical four-force, while
holding plasma state, dissipation, spatial modes, temporal statistics, and
injected power fixed. No existing native pair meets all those conditions.

The original proposal used the hot beta-one state as a candidate shared
reference: `rho0=1`, `p0=100`, `gamma=4/3`, `sigma0=200/401`, and
`tA0=sqrt(601/200)`. It proposed a `256^3` unit cube, `32^3` MeshBlocks,
IMEX3, cell-centred E, WENO-Z, LLF, FOFC, `tau_pi=0.02`, entropy target
`K0=100`, and `t_cool=0.1*tA0`. Its transport
`nu_sh=eta=0.001836233427266871` corresponds to the **older Re=Rm=50**
normalization with `Ldrive=1/(2*pi)`. It is retained as a proposal only;
it must not be confused with the committed uncooled Re=1000 benchmark.
Choosing Re=50 or 1000 for a future matched pair remains a scientific decision.

Before implementing that comparison:

1. Provide a shared eight-mode spatial catalogue with the same signed
   wavevectors and prescribed transverse polarization relative to `B0`.
   A generic isotropic solenoidal shell is not the same realization.
2. Provide the same balanced oscillating-OU coefficient process, explicit seed,
   and initialization in both drivers. In the original proposed frequency
   convention, `omega=+/-0.6*omega_ref`, decay rate `0.5*omega_ref`, and
   `omega_ref=2*pi/(sqrt(3)*tA0)`. The corresponding envelope time is
   `sqrt(3)/pi*tA0`, not the native mechanical times above.
3. Verify the coefficient histories before applying their different physical
   coupling. The inspected mechanical driver does not expose the antenna's
   shared `zhdankin8` oscillating-OU path or an input forcing seed.
4. Calibrate equal **measured injected power per volume** in the same state.
   Equal current and acceleration amplitudes are not an equal-power control.
   Keep calibration intervals separate from the final comparison interval.
5. Specify duration, equilibration, seeds, and acceptance criteria before
   production. The older proposal suggested ten Alfvén times with analysis
   over the latter five and at least three seeds if feasible; those are
   proposed choices, not completed measurements.

Compare cumulative injected energy as well as physical time. Measure momentum
and total-energy source closure, cooling, velocity and sound speed, magnetic
and electric energies, magnetic fluctuations, spectra and anisotropy, and
current/density statistics. Distinguish endpoint heating proxies from a
closed electromagnetic energy budget. Ratios, spectra, and any auxiliary
correlations absent from the standard histories require an explicitly checked
analysis rather than being assumed available.

## Reconciliation boundaries

The current settings were checked against all twelve native inputs, the
separate uncooled benchmark, and the two driver implementations at `973840bc`.
The clock relations were checked numerically against each input's duration,
cooling, and output cadence. No forcing implementation, input, plotting script,
or production output was changed. Historical pilot statistics and literature
claims were not independently revalidated; the old documents remain in
`feature/teno6` commit `6876a891` for provenance.
