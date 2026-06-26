# CGM Turbulence Setup

This note documents the stratified CGM turbulence setup developed in
`src/pgen/turb_cgm.cpp` and the canonical 32 x 32 x 64 input files
`inputs/turbulence/cgm_turbulence_hydro.input` and
`inputs/turbulence/cgm_turbulence_mhd.input`.

The setup is Wibking-inspired, not an exact reproduction of Wibking, Voit &
O'Shea (2024), "Precipitation possible: turbulence-driven thermal instability
with constrained entropy profiles", [arXiv:2410.03886](https://arxiv.org/abs/2410.03886).
It uses their basic stratified CGM atmosphere and entropy-profile-maintenance
philosophy, but it is implemented inside AthenaK with the local Fourier
turbulence driver, spatially windowed forcing, and a simplified baseline
energy-control model. Some of the energy-balance experiments were also
motivated by Mohapatra, Federrath & Sharma (2022), "Multiphase turbulence in
galactic halos: effect of the driving", [arXiv:2206.03602](https://arxiv.org/abs/2206.03602),
but the current cleaned baseline is simpler than the intermediate
global-balance variants.

## Current Baseline

The current default model has three active pieces:

1. A hydrostatic, isothermal, stratified CGM atmosphere in a fixed external
   vertical gravitational field.
2. Disk-confined Fourier turbulence driving, currently with the central engine
   component turned off and an ISM/background component active.
3. Thermal source terms split by role:
   - the atmosphere has radiative cooling plus a plane-averaged thermostat that
     balances the first-moment vertical cooling profile;
   - the disk has no radiative cooling and no explicit thermostat heating;
   - the disk does have an explicit thermal sink that offsets the measured
     turbulent work injected by the Fourier driver.

This baseline was selected because it gave the best long-time profile behavior
among the tested models: the atmospheric entropy profile stayed close to the
initial Wibking-like profile, the disk did not catastrophically condense, and
the global first-moment energy accounting stayed transparent.

## Units

The default input uses:

```text
length_cgs = 1 kpc
time_cgs   = 1 Myr
density    = 1e-26 g cm^-3
mu         = 0.6
```

The problem generator converts physical temperatures and cooling coefficients
into code units using AthenaK's unit system.

## Mesh, Boundaries, and Solvers

The default smoke/diagnostic input is:

```text
nx1 x nx2 x nx3 = 32 x 32 x 64
x, y = [-50, 50] kpc
z    = [-100, 100] kpc
```

The boundary conditions are periodic in `x` and `y`, and reflecting in `z`.
The solver defaults are:

```text
integrator  = rk3
reconstruct = wenoz
gamma       = 5/3
fofc        = true
cfl_number  = 0.6
```

The hydro input uses HLLC. The MHD input uses HLLD and initializes a
divergence-free spectral random magnetic field through the shared
`SpectralICGenerator` utility in `src/utils/spectral_ic_gen.*`. The field is
generated from Fourier vector-potential modes, has zero mean by default, and is
normalized in `turb_cgm` to `beta_z0 = 100` by default. The mode band, spectrum,
and seed live in the input's `<spectral_ic>` block. The canonical MHD input uses
a large-scale parabolic spectrum with `nlow = 1`, `nhigh = 3`, and therefore a
peak at `k = 2`, with `iseed = 210989`.

WENO-Z plus FOFC became the preferred default after WENO-Z without FOFC showed
more fragile behavior near sharp thermal/source transitions.

### HEFFTE Build

For production MHD runs, build with the distributed heFFTe FFT backend:

```text
cmake -S . -B build_cgm_heffte \
  -DPROBLEM=turb_cgm \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_HEFFTE=ON \
  -DCMAKE_PREFIX_PATH=/Users/beattijr/.local/heffte-install
cmake --build build_cgm_heffte -j8
```

When AthenaK is compiled with `HEFFTE_ENABLED=1`, `SpectralICGenerator` uses
heFFTe automatically for the initial vector-potential transform and reports
`backend='heffte'` at startup.

## Hydrostatic Atmosphere

The atmosphere is isothermal at `temp0_cgs = 1e7 K`, with gas pressure

```text
P = rho * theta0
```

where `theta0` is the code-unit temperature. The density profile is

```text
rho(z) = rho_z0 * exp[
    1 - |z|/z0 + (h/z0) * (tanh(|z|/h) - tanh(z0/h))
]
```

with default scale parameters

```text
z0 = 50 kpc
h  = 5 kpc
```

The fixed vertical gravity is

```text
g_z(z) = -sgn(z) * theta0/z0 * tanh^2(|z|/h)
```

This is chosen so that the isothermal pressure gradient balances gravity
analytically:

```text
dP/dz = rho * g_z .
```

The galaxy/disk midplane is at `z = 0`. The parameter `h` sets the vertical
disk scale height used by the forcing and by the thermal masks.

If `<problem>/rho_z0` is not supplied, the code computes it from the requested
cooling-time to free-fall-time ratio at `z0`:

```text
tcool_tff_z0 = 4.0
```

The default cooling coefficient is based on constant `Lambda0` cooling with
`lambda0_cgs = 1e-22` and hydrogen mass fraction `X_H = 0.75`.

## Turbulent Driving

The AthenaK turbulence driver now supports multiple independent Fourier forcing
components. The current CGM input uses:

```text
<turb_driving>
num_components = 2
```

Each component has independent `dedt`, `tcorr`, mode band, mode geometry,
solenoidal/compressive projection, and spatial windowing. Components are
normalized independently and summed into the public `pturb->force` array.

### Component 0: Central Engine

The central engine is configured but currently off by default:

```text
name             = central_engine
dedt             = 0.0
tcorr            = 500 Myr
nlow, nhigh      = 1, 3
driving_geometry = vertical
driving_profile  = parabola
vertical_window  = smooth_tophat, width 5 kpc, transition 2.5 kpc
transverse_window = smooth_tophat, radius 5 kpc, transition 2.5 kpc
```

When enabled, it represents a compact vertical central engine localized near
the disk and near the origin in the `x-y` plane.

### Component 1: Background ISM Driving

The active baseline forcing is:

```text
name             = background_ism
dedt             = 8.0e-5
tcorr            = 25 Myr
nlow, nhigh      = 4, 8
driving_geometry = isotropic
driving_profile  = band
sol_weight       = 0.0
vertical_window  = smooth_tophat, width 5 kpc, transition 2.5 kpc
transverse_window = none
```

This means the ISM forcing:

- is confined to the disk layer in `z`;
- is not truncated in `x` or `y`;
- drives smaller Fourier modes than the central engine;
- is purely compressive because `sol_weight = 0.0`;
- has a short OU correlation time of 25 Myr.

For this setup, the forced region is the disk. Velocity and density
fluctuations away from the disk are generated by disk-launched motions and
their interaction with the stratified atmosphere.

## Thermal Model

The cleaned baseline deliberately avoids a menu of thermal modes. The active
model is always:

```text
cooling      = true
thermostat   = true
thermal_mask = outer_tanh
```

with

```text
thermal_transition = 2.5 kpc
cooling_cfl        = 0.1
```

### Cooling

The local cooling rate is

```text
C(x, y, z) = w_th(z) * cooling_coef * rho^2 .
```

For the default `outer_tanh` mask:

```text
w_th(z) = 0,                         |z| < h
w_th(z) = tanh((|z| - h)/delta_th),  |z| >= h
```

where `delta_th = thermal_transition`. Thus there is no radiative cooling in
the disk layer. Cooling turns on smoothly above `|z| = h` and approaches the
constant-Lambda atmosphere cooling law outside the transition.

### Atmospheric Plane-Balance Thermostat

Before applying thermal source terms, `TcgPreparePlaneAverages` computes
plane-wise reductions over each global `z` index:

```text
C_plane(k) = integral_plane C dV
M_plane(k) = integral_plane rho dV
```

The heating coefficient for that plane is

```text
b(k) = C_plane(k) / M_plane(k)
```

and each cell receives

```text
H(x, y, z_k) = rho(x, y, z_k) * b(k).
```

Therefore the plane-integrated thermostat exactly balances the plane-integrated
cooling source before per-cell source clipping:

```text
integral_plane H dV = integral_plane C dV .
```

Because `w_th = 0` inside `|z| < h`, the disk planes have zero cooling and
therefore zero thermostat heating. The thermostat is intended to hold the
first-moment atmosphere entropy profile steady, not to suppress turbulent
second moments.

### Disk Drive-Balance Sink

The Fourier driver injects kinetic energy. In the current disk-confined setup,
that work is treated as disk-local energy input. The problem generator reads
the measured driver power from

```text
pturb->last_power
```

and sets

```text
P_turb = pturb->last_power .
```

The disk sink is distributed with a smooth top-hat weight:

```text
w_disk(z) = 1,                                      |z| <= h - delta_disk
w_disk(z) = smoothstep((h - |z|)/delta_disk),       h - delta_disk < |z| < h
w_disk(z) = 0,                                      |z| >= h
```

with defaults

```text
disk_balance_width      = 5.0 kpc
disk_balance_transition = 2.5 kpc
```

The normalization is density-weighted:

```text
R_disk = -P_turb / integral w_disk rho dV
```

and the cell source is

```text
D(x, y, z) = w_disk(z) * rho(x, y, z) * R_disk .
```

This gives

```text
integral D dV = -P_turb
```

before per-cell source clipping. In the current default, this balances the
background ISM forcing. If the central engine is enabled, its power is also
included in `pturb->last_power`; that is still consistent if the active forcing
remains inside the disk balance window.

### Source Clipping and Floors

Thermal source terms are explicitly applied. The code limits the per-step
thermal energy change using

```text
|Delta e_int| <= cooling_cfl * e_int
```

and then enforces the temperature floor through `temp_floor_cgs`.

The HST diagnostics report both requested source terms and applied source
terms. Differences between requested and applied rates indicate clipping.

## Diagnostics

The default user history output is enabled with

```text
<output1>
file_type = hst
user_hist_only = true
dt = 5.0
```

Important columns include:

| Column | Meaning |
| --- | --- |
| `mean_T`, `mean_T_mid`, `mean_T_atm` | Volume-weighted temperature diagnostics. |
| `mean_mach`, `max_mach`, `max_abs_v` | Turbulence strength diagnostics. |
| `cool_src`, `heat_src` | Requested cooling and thermostat powers. |
| `turb_src` | Measured Fourier-driver power, `pturb->last_power`. |
| `bal_src` | Total atmosphere cooling target used by the plane thermostat. |
| `disk_turb` | Power assigned to the disk balance; currently equal to `turb_src`. |
| `disk_bal` | Requested disk sink power, expected to be `-disk_turb`. |
| `cool_disk`, `heat_disk` | Historical disk thermal columns; zero in the cleaned baseline. |
| `app_cool`, `app_heat`, `app_disk`, `app_net` | Applied source powers after source clipping. |
| `mass_disk`, `eth_disk`, `ekin_disk`, `etot_disk` | Disk-window integral quantities. |
| `mdot_adv`, `edot_adv`, `eint_adv` | Flux-like diagnostics through the smooth disk-window boundary. |
| `grav_work` | Disk-window gravitational work diagnostic. |
| `turb_wdsk` | Direct diagnostic of turbulence work inside the disk window. |

For a clean run without severe source clipping, the key budget checks are:

```text
heat_src ~= cool_src
disk_bal ~= -disk_turb
heat_src + turb_src + disk_bal - cool_src ~= 0
```

The last relation is the first-moment energy accounting we want for the
baseline model.

## Validated Behavior

The restored baseline was run for 1000 Myr at 32 x 32 x 64 on 8 MPI ranks. The
final history showed:

```text
mean_T       = 0.146053
mean_T_mid   = 0.128122
mean_T_atm   = 0.147494
mean_mach    = 0.143217
max_mach     = 2.4112
fcold_mid    = 0
cool_src     = 358.508
heat_src     = 358.508
turb_src     = 160.017
disk_turb    = 160.017
disk_bal     = -160.017
```

The energy budget closed as:

```text
heat_src + disk_bal + turb_src - cool_src = 0
```

Plane-averaged profile ratios from 0 to 1000 Myr were:

| Region | rho ratio | T ratio | K ratio |
| --- | ---: | ---: | ---: |
| Disk, `|z| <= h` | 0.988 | 0.859 | 0.867 |
| Inner atmosphere, `10 <= |z| <= 50 kpc` | 1.007 | 0.996 | 1.000 |
| Outer atmosphere, `50 <= |z| <= 100 kpc` | 0.998 | 1.040 | 1.042 |

This is the main reason the current model was kept: the atmosphere entropy
profile is nearly stationary over 1 Gyr, and the disk avoids the severe
runaway cooling/heating behavior seen in several earlier variants.

After the cleanup that removed the old thermal-mode knobs, an 8-rank 20 Myr
smoke run gave:

```text
cool_src  = 3.48533e+02
heat_src  = 3.48533e+02
turb_src  = 1.64817e+02
disk_turb = 1.64817e+02
disk_bal  = -1.64817e+02
cool_disk = 0
heat_disk = 0
```

with

```text
heat_src + turb_src + disk_bal - cool_src = 0 .
```

## Running the Baseline

Configure and build:

```bash
cmake -S . -B build_cgm_mpi -DPROBLEM=turb_cgm -DAthena_ENABLE_MPI=ON
cmake --build build_cgm_mpi -j8
```

Run a short MPI test:

```bash
mkdir -p /private/tmp/athenak_cgm_test
cd /private/tmp/athenak_cgm_test
mpirun -n 8 /Users/beattijr/Documents/Research/2025/athenak/build_cgm_mpi/src/athena \
  -i /Users/beattijr/Documents/Research/2025/athenak/inputs/turbulence/cgm_turbulence_hydro.input \
  time/tlim=100.0 \
  job/basename=CGM_Turbulence_Hydro_Test
```

For longer runs, use restart output:

```text
<output4>
file_type = rst
dt = 500.0
```

## Profile Diagnostics

The preferred diagnostic for deciding whether the CGM setup is behaving well is
the Wibking-style plane-averaged vertical profile evolution:

- density `n(|z|)`;
- entropy proxy `K = T n_e^{-2/3}`;
- `t_cool / t_ff`;
- `t_cool / t_BV`;
- time colorbar across snapshots.

The key point is to diagnose the first moments of the atmosphere and disk
separately. The current thermal model is designed to keep the first-moment
profiles steady while allowing turbulence to develop in the second moments.

Slice plots remain useful for visual inspection, but profile evolution is the
primary acceptance diagnostic for the thermal balance.

## Known Limitations and Caveats

- The setup supports pure hydro or pure MHD, but not simultaneous `<hydro>` and
  `<mhd>` blocks.
- The current MHD field generator is a local random vector-potential
  initializer inside `turb_cgm.cpp`. It is divergence-free by construction on
  the AthenaK face-centered grid, but it is not yet a standalone shared utility.
- The atmosphere is initially isothermal and hydrostatic in a fixed external
  potential; self-gravity, conduction, cosmic rays, and explicit feedback
  physics are not included.
- Cooling is a constant-Lambda model, not a tabulated temperature-dependent CGM
  cooling curve.
- The thermal model is an idealized first-moment control model. It is not a
  physical feedback model.
- The disk balance uses `pturb->last_power`. This is appropriate for the current
  disk-confined forcing. If future forcing deposits substantial power outside
  the disk balance window, the balance should be revisited.
- `turb_wdsk` is a diagnostic estimate of disk-window turbulent work; the
  active disk balance uses `pturb->last_power`, not `turb_wdsk`.
- Restart files currently preserve the summed public force field, but not the
  full per-component OU forcing state.
- Reflecting `z` boundaries are inherited from the current baseline. They are
  useful for keeping the hydrostatic box closed, but outflow/open vertical
  boundaries should be considered separately if studying mass exchange.

## Historical Cleanup

Earlier experiments included selectable thermal modes and extra parameters:

```text
thermal_mode
disk_thermal_mode
disk_cooling_multiplier
thermostat_kp
thermostat_kp_disk
thermostat_kp_atm
```

Those have been removed from the cleaned baseline. The current source code now
has one intended CGM thermal model: atmosphere plane balance plus disk
drive-balance sink.
