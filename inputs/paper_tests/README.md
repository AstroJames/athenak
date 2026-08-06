# Relativistic transport paper tests

This directory contains the production input files used for the viscosity and
visco-resistive comparisons in the paper.  The smaller regression tests remain
under `inputs/tests/`; these inputs retain the resolutions, transport
coefficients, reconstruction, and random seeds used for the figures. The
reusable ideal-sector input now uses the preferred RK3 integrator described
below; the permanent campaign metadata retain the exact earlier run settings.

The complete 2026 rerun matrix, including command provenance and the higher
resolution figure cases, is documented in `PRODUCTION_CAMPAIGN.md`.

## Test matrix

| Test | Resolution | Physics | Parameters |
| --- | ---: | --- | --- |
| `rsrmhd_viscous_shear_nu005.athinput` | 256 | viscous SR hydro limit | `nu_sh=0.005`, `tau_pi=0.2` |
| `rsrmhd_viscous_shear_nu010.athinput` | 256 | viscous SR hydro limit | `nu_sh=0.010`, `tau_pi=0.2` |
| `rsrmhd_viscous_shear_nu020.athinput` | 256 | viscous SR hydro limit | `nu_sh=0.020`, `tau_pi=0.2` |
| `rsrmhd_viscous_shear_nu040.athinput` | 256 | viscous SR hydro limit | `nu_sh=0.040`, `tau_pi=0.2` |
| `rsrmhd_viscous_shear_nu050.athinput` | 256 | viscous SR hydro limit | `nu_sh=0.050`, `tau_pi=0.2` |
| `rsrmhd_ohmic_decay.athinput` | 2048 | uniform-resistivity SRMHD | `eta=(0.001,0.003,0.01,0.03)`, `a=0.02`, `B_guide=10` |
| `rsrmhd_decaying_turbulence_ideal.athinput` | 512 x 512 | ideal SRMHD | `eta=nu_sh=0`, RK3 |
| `rsrmhd_decaying_turbulence_pm1.athinput` | 512 x 512 | visco-resistive SRMHD | `nu_sh=0.0012`, `eta=0.0012`, `Re=50`, `Pm=1` |
| `rsrmhd_decaying_turbulence_pm10.athinput` | 512 x 512 | visco-resistive SRMHD | `nu_sh=0.0012`, `eta=0.00012`, `Re=50`, `Pm=10` |
| `rsrmhd_decaying_turbulence_pm50.athinput` | 512 x 512 | visco-resistive SRMHD | `nu_sh=0.0012`, `eta=0.000024`, `Re=50`, `Pm=50` |
| `../tests/rsrmhd_viscous_kh.athinput` | 384 x 768 | ideal and viscous SR hydrodynamics | AthenaK-paper density, shear, and perturbation profiles, without particles |
| `rsrmhd_driven_cooling_scan_3d64.athinput` | 64 x 64 x 64 | mechanically driven visco-resistive SRMHD | no cooling and `t_cool/t_0=(0.1,1,10)`, nominal `Re=Rm=100`, `Pm=1` |
| `rsrmhd_antenna_zhdankin32.athinput` | 32 x 32 x 32 | electromagnetic antenna-driven visco-resistive SRMHD | Zhdankin eight-mode baseline, `beta_0=1`, `sigma_0=0.5`, nominal `Re=Rm=50` |

Here `Pm = nu_sh/eta`.  All finite-`Pm` simulations therefore have the same
nominal initial Reynolds number
`Re=v_rms/(n_p nu_sh)=50`, with `v_rms=0.15` and `n_p=2.5`, and differ only in
magnetic diffusivity.  They use uniform resistivity, `tau_pi=0.02`, WENOZ
reconstruction, and FOFC.  The ideal case uses the same velocity and magnetic
initial conditions.

New ideal hydrodynamic and ideal-SRMHD calculations pair WENOZ with SSPRK(3,3),
selected by `integrator=rk3`.  Resistive, viscous, and visco-resistive inputs use
IMEX-SSP3(4,3,3), selected by `integrator=imex3`; its three explicit stages are
the same SSPRK(3,3) method.  The archived campaign predates this policy and must
not be relabelled as RK3/IMEX3 data.

The shear-wave inputs use the resistive SRMHD state container because that is
where the current viscous IMEX implementation lives, but initialize `B=E=0`.
The electromagnetic and Ohmic sectors therefore remain exactly inactive, and
the value of electrical resistivity in those five inputs has no dynamics.

## Relativistic AthenaK-paper KHI

The two-dimensional KHI uses the Lecoanet et al. profile implemented by the
AthenaK release-paper problem generator.  The domain is
`-0.5 < x < 0.5`, `-1 < y < 1`, the interfaces lie at `y=+-0.5`, the density
is 2 between the interfaces and 1 outside, the pressure is 10, the shear
transition width is 0.05, and the transverse perturbation has amplitude 0.01
and Gaussian width 0.2.  The matched production grid is `384 x 768`.  We omit both
the passive contaminant and the Lagrangian tracer particles because the
viscous comparison uses fluid vorticity, transverse velocity, and shear stress.

The published Newtonian setup has asymptotic shear speed 1, which is not a
valid relativistic three-velocity.  We retain its unit shear as a unit spatial
four-velocity, corresponding to asymptotic `|v^x|=1/sqrt(2)`, and use the
project-wide relativistic EOS choice `gamma=4/3`.  These are the only physical
adaptations.  Both cases use WENO-Z and FOFC and evolve to `t/t_c=3`.  The
ideal control disables the resistive and viscous sectors and uses SSPRK(3,3).
The `nu_sh=1e-4` case uses IMEX-SSP3(4,3,3).

## Zhdankin antenna calibration

The `rsrmhd_antenna_zhdankin32.athinput` pilot starts from an ultrarelativistic
fluid approximation to the published pair-plasma baseline and uses the same
eight signed wavevectors, balanced counter-propagating families, frequency,
decorrelation rate, and nominal current amplitude. The `zhdankin` amplitude
normalization converts the paper's Gaussian-unit current to AthenaK's
rationalized units, including the factor of `4 pi` in the Ampere source. It runs
without cooling so
that the magnetic-fluctuation saturation, injection efficiency, heating, and
declining magnetization can be compared directly.  Summarize the result with

```sh
/opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/analyze_rsrmhd_antenna.py \
  /path/to/antenna_zhdankin32.user.hst \
  --output /path/to/antenna_zhdankin32_summary.json
```

The main published targets are `delta B_rms/B0 approximately 1`,
`v_rms approximately 0.7 v_A`, and a late-time dimensionless injection rate
near 1.7.  These are calibration observables, not hard regression tolerances,
because the fluid dissipation model differs from the PIC calculation.

The `32^3` four-rank calibration on 2026-07-15 established three useful cases.
The unconverted Gaussian current underdrives the system, with developed
`delta B_rms/B0=0.154` and instantaneous injection efficiency `0.0173`. The
literal rationalized Zhdankin amplitude gives `delta B_rms/B0=1.355` and
`v_rms/v_A=0.702`, but its injection efficiency is high (`3.33`). Setting both
amplitude fractions to `0.65` gives `delta B_rms/B0=1.003+/-0.131`,
`v_rms/v_A=0.628+/-0.094`, instantaneous injection efficiency `1.628`, and a
heating-slope efficiency of `1.523` over `4 <= t/t_A0 <= 6`. Thus `0.65` is the
recommended fluid calibration when matching the published amplitude and energy
budget together; `1.0` remains the literal external-current normalization.

The full-amplitude face-centered-E run reaches order-unity fluctuations but its
multidimensional Picard iteration fails at `0.224 t_A0` for CFL `0.4`; reducing
the CFL to `0.1` only delays the failure to `0.281 t_A0`. The otherwise identical
cell-centered-E run completes all six crossing times with relative source-energy
closure `1.3e-12`. Until the strong-field face-E iteration is made more robust,
use cell-centered E for this benchmark.

Generate the comparison figure with

```sh
/opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/plot_rsrmhd_antenna_calibration.py \
  /path/to/unconverted.user.hst \
  /path/to/exact_zhdankin.user.hst \
  /path/to/fluid_calibrated.user.hst \
  --labels "unconverted control" "exact Zhdankin amplitude" \
           "fluid calibration (0.65)" \
  --output /path/to/antenna_calibration_comparison.pdf
```

## One-dimensional viscous shear wave

The smooth periodic shear mode is initialized as

```text
u^y(x,0) = 0.5 sin(2 pi x),    pi_xy(x,0) = 0.
```

For small velocity, the Israel--Stewart shear subsystem reduces to the
telegraph equation

```text
tau_pi d_t^2 v_y + d_t v_y - nu_sh d_x^2 v_y = 0.
```

Consequently a Fourier amplitude `V(t)` with wavenumber `k=2 pi` obeys

```text
tau_pi V'' + V' + nu_sh k^2 V = 0,
V(0)=0.5,  V'(0)=0,
pi_xy = (e+p) V'/k.
```

The final profile files are named `shear_nuXXX-profile.dat`.  Plot all five
profiles and the analytic telegraph solutions with

```sh
MPLCONFIGDIR=~/.matplotlib \
  /opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/plot_rsrmhd_viscous_shear_scan.py \
  --data-dir /path/to/shear/output \
  --output-prefix /path/to/figures/viscous_shear_scan
```

Build this family with the default `-DPROBLEM=built_in_pgens`.  Each input is
standalone; run it from a common output directory to assemble the scan.

## Strong-guide-field Harris-sheet decay

The Ohmic-decay test adapts the one-dimensional magnetic-diffusion experiment
of Grehan et al. (2025, arXiv:2503.20013) to AthenaK's Heaviside--Lorentz units.
It initializes

```text
By = B0 tanh(x/a),
Bz = sqrt(Bguide^2 + B0^2 - By^2),
```

with `B0=1`, `Bguide=10`, `a=0.02`, hot magnetization 10, temperature 1,
and zero velocity and electric field.  The uniform AthenaK resistivity is the
magnetic diffusivity that corresponds to `c^2 eta_cgs/(4 pi)` in the paper.
The plotting script measures the peak of `Jz=d_x By` and the RMS sheet width
`x_rms=<x^2>_(Jz^2)^1/2`; after the finite-width transient they approach
`Jz,max ~ t^-1/2` and `x_rms ~ (eta t)^1/2`.  The profile panel uses the
fiducial `eta=0.01` run.  The max and RMS panels compare
`eta=(0.001,0.003,0.01,0.03)`.  Generate the additional runs by overriding
`mhd/resistivity` and keeping each output in a separate directory.

Generate the profile and scaling comparison with

```sh
MPLCONFIGDIR=~/.matplotlib \
  /opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/plot_rsrmhd_ohmic_decay.py \
  --data-dir /path/to/eta0p01 \
  --resistivity 0.01 \
  --scan 0.001=/path/to/eta0p001 \
  --scan 0.003=/path/to/eta0p003 \
  --scan 0.03=/path/to/eta0p03 \
  --output-prefix /path/to/figures/rsrmhd_ohmic_harris_decay
```

## Two-dimensional decaying turbulence

All four cases use identical solenoidal velocity and magnetic fields with
`v_rms=B_rms=0.15`, modes `1 <= n <= 4`, and fixed random seeds.  This makes
changes among the non-ideal runs attributable to `Pm`, while the ideal run
measures numerical dissipation.  History output records kinetic, magnetic, and
electric energies; enstrophy; current; shear stress; thermodynamic means; and
the face-centered divergence error.  Final two-dimensional profiles are also
written for the paper figures.

Build this family with MPI and the default `-DPROBLEM=built_in_pgens`, then run
the four inputs in separate directories named `ideal`, `pm1`, `pm10`, and
`pm50`, for example with `mpirun -n 8`.  Generate the field/history comparison
and spectra with

```sh
MPLCONFIGDIR=~/.matplotlib \
  /opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/plot_rsrmhd_decaying_pm_ideal.py \
  --root /path/to/output/root \
  --output-dir /path/to/figures

MPLCONFIGDIR=~/.matplotlib \
  /opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/plot_rsrmhd_decaying_pm_spectra.py \
  --root /path/to/output/root \
  --output /path/to/figures/decaying_pm_spectra.pdf
```
