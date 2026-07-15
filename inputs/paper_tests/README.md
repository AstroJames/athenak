# Relativistic transport paper tests

This directory contains the production input files used for the viscosity and
visco-resistive comparisons in the paper.  The smaller regression tests remain
under `inputs/tests/`; these inputs retain the resolutions, transport
coefficients, reconstruction, and random seeds used for the figures.

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
| `rsrmhd_decaying_turbulence_ideal.athinput` | 512 x 512 | ideal SRMHD | `eta=nu_sh=0` |
| `rsrmhd_decaying_turbulence_pm1.athinput` | 512 x 512 | visco-resistive SRMHD | `nu_sh=0.0012`, `eta=0.0012`, `Re=50`, `Pm=1` |
| `rsrmhd_decaying_turbulence_pm10.athinput` | 512 x 512 | visco-resistive SRMHD | `nu_sh=0.0012`, `eta=0.00012`, `Re=50`, `Pm=10` |
| `rsrmhd_decaying_turbulence_pm50.athinput` | 512 x 512 | visco-resistive SRMHD | `nu_sh=0.0012`, `eta=0.000024`, `Re=50`, `Pm=50` |
| `rsrmhd_driven_turbulence_mach0p5_re200.athinput` | 128 x 128 | mechanically driven visco-resistive SRMHD | `M_turb=0.5`, nominal `Re=Rm=200`, `Pm=1` |
| `rsrmhd_driven_turbulence_3d_mach0p5_re50.athinput` | 32 x 32 x 32 | mechanically driven visco-resistive SRMHD | rest start, beta-one `B^z` guide field, target `M_turb=0.5`, nominal `Re=Rm=50`, `Pm=1` |
| `rsrmhd_antenna_zhdankin32.athinput` | 32 x 32 x 32 | electromagnetic antenna-driven visco-resistive SRMHD | Zhdankin eight-mode baseline, `beta_0=1`, `sigma_0=0.5`, nominal `Re=Rm=50` |

Here `Pm = nu_sh/eta`.  All finite-`Pm` simulations therefore have the same
nominal initial Reynolds number
`Re=v_rms/(n_p nu_sh)=50`, with `v_rms=0.15` and `n_p=2.5`, and differ only in
magnetic diffusivity.  They use uniform resistivity, `tau_pi=0.02`, WENOZ
reconstruction, and FOFC.  The ideal case uses the same velocity and magnetic
initial conditions.

The shear-wave inputs use the resistive SRMHD state container because that is
where the current viscous IMEX implementation lives, but initialize `B=E=0`.
The electromagnetic and Ohmic sectors therefore remain exactly inactive, and
the value of electrical resistivity in those five inputs has no dynamics.

## Zhdankin antenna calibration

The `rsrmhd_antenna_zhdankin32.athinput` pilot starts from an ultrarelativistic
fluid approximation to the published pair-plasma baseline and uses the same
eight signed wavevectors, balanced counter-propagating families, frequency,
decorrelation rate, and nominal current amplitude.  It runs without cooling so
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
