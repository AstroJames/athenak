# Visco-resistive SRMHD paper campaign

The archived IMEX2 campaign is stored in
`Research/2026/visco_resistive_SRMHD`.  The replacement SSPRK3/IMEX3 campaign
is stored in its `rk3_campaign` subdirectory.  One MPI-enabled release build
serves all cases.  Every individual calculation has a 3590 s wall-clock timeout and
stores its copied input, exact command, standard output, elapsed time, MPI rank
count, source commit, and completion status in its case directory.

## Test matrix

| Directory | Test | Production resolution or scan |
| --- | --- | --- |
| `01_current_sheet` | self-similar resistive sheet, both electric layouts | `1024`, four values of `eta` |
| `02_ohmic_harris` | strong-guide-field Harris-sheet spreading | `2048`, four values of `eta` |
| `03_charged_vortex` | charged vortex convergence, both electric layouts | `32^2`--`512^2` |
| `04_cyclic_and_decomposition` | dual-CT cyclic and block/rank invariance | `32^3` cyclic; `16^3` decomposition |
| `05_viscous_telegraph` | transverse Israel--Stewart scans | `128`, 60 nonzero-time runs |
| `06_viscous_phaseb` | diffusion limit, timestep scaling, longitudinal mode | `64`--`512` |
| `07_viscous_shear_layer` | finite-amplitude periodic shear profiles | `1024`, five viscosities |
| `08_boosted_and_rotated_shear` | finite boost and propagation along `x2`, `x3` | `256` along the wave direction |
| `09_viscous_khi` | ideal SSPRK3 and viscous IMEX3 relativistic KHI | `384 x 768` |
| `10_decaying_turbulence` | ideal and `Pm=1,10,50`, fixed `Re=50` | `512 x 512` |
| `11_driven_cooling_scan` | forced, cooled, visco-resistive turbulence | four matched `64^3` boxes |
| `12_antenna` | electromagnetic antenna-driven turbulence | `32^3` |

The parent campaign predates the RK3 policy and remains available only for
provenance.  New ideal calculations use SSPRK(3,3), while resistive, viscous,
and visco-resistive calculations use IMEX-SSP3(4,3,3), whose explicit tableau
is SSPRK(3,3).  The relativistic ideal-gas calculations use `gamma=4/3`.

The plotting pipeline in `scripts/rebuild_rsrmhd_paper_figures.py` reads only
these permanent outputs.  It applies `~/.matplotlib/matplotlibrc`, uses no
figure titles, and writes all manuscript PDFs and audit PNGs under `figures/`.
After plotting, `scripts/summarize_rsrmhd_paper_campaign.py` verifies that every
production run completed below the one-hour cap and writes the run table,
campaign summary, and figure checksums under `00_manifest/`.  The same directory
also stores the source commit, working-tree patch, executable hash, and initial
git status.  Superseded duplicate charged-vortex and non-dual 3D runs are
retained for provenance but are excluded from the production manifest.

## Reproduction commands

From the AthenaK repository, run the complete matrix with

```sh
/opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/run_rsrmhd_paper_campaign.py all \
  --root /Users/beattijr/Documents/Research/2026/visco_resistive_SRMHD/rk3_campaign \
  --repo /Users/beattijr/Documents/Research/2025/athenak \
  --athena /Users/beattijr/Documents/Research/2026/visco_resistive_SRMHD/rk3_campaign/build/src/athena
```

Completed cases are skipped unless `--force` is supplied.  Rebuild and audit
the manuscript products with

```sh
MPLCONFIGDIR=/Users/beattijr/.matplotlib \
  /opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/rebuild_rsrmhd_paper_figures.py \
  --root /Users/beattijr/Documents/Research/2026/visco_resistive_SRMHD/rk3_campaign \
  --repo /Users/beattijr/Documents/Research/2025/athenak

/opt/homebrew/Caskroom/miniconda/base/bin/python \
  vis/python/summarize_rsrmhd_paper_campaign.py \
  --root /Users/beattijr/Documents/Research/2026/visco_resistive_SRMHD/rk3_campaign
```
