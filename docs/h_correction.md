# H-Correction for the HLLD Riemann Solver

## Overview

The h-correction (Sanders, Morano & Druguet 1998) adds direction-dependent
numerical dissipation to the HLLD Riemann solver. It targets the **carbuncle
instability**, a numerical artifact that produces spurious grid-aligned
perturbations at strong shocks (e.g., blastwave fronts aligned with coordinate
axes).

The correction works by widening the outermost HLL wave speed estimates using
the maximum characteristic speed from the *transverse* directions at each
cell interface:

```
S_L = min(S_L, -eta)
S_R = max(S_R, +eta)
```

where `eta` is the maximum fast magnetosonic eigenvalue `|v| + c_f` evaluated
in the transverse directions at the two cells flanking the interface. This
selectively increases dissipation where transverse wave speeds are large
relative to the normal wave speeds, which is precisely the regime where the
carbuncle instability appears.

## Usage

Enable the h-correction by adding the following to the `<mhd>` input block:

```
<mhd>
rsolver    = hlld
h_correction = true   # default: false
```

The h-correction is only supported with `rsolver = hlld`. Attempting to enable
it with any other Riemann solver will produce a fatal error.

No additional parameters are required. The correction is applied automatically
in 2D and 3D problems. In 1D there are no transverse directions, so the
correction has no effect.

## Algorithm

### Step 1: Pre-compute cell-centered eigenvalues

Before the directional flux sweeps, a kernel computes the maximum absolute
eigenvalue at each cell center for all three coordinate directions:

```
eta_d(i,j,k) = |v_d| + c_{f,d}
```

where `c_{f,d}` is the fast magnetosonic speed with `B_d` as the normal
magnetic field component, and `d` ranges over `{x1, x2, x3}`. These are
stored in three `DvceArray4D<Real>` arrays (`eta1`, `eta2`, `eta3`).

### Step 2: Apply correction at each interface

When computing fluxes in direction `d`, the HLLD solver receives the two
*transverse* eta arrays. For each interface, `eta` is set to the maximum
value across:

- The two cells flanking the interface (left and right of the face)
- Both transverse directions (when available)

For example, for an x1-face at position `i`, with transverse directions x2
and x3:

```
eta = max(eta2(i-1,j,k), eta2(i,j,k), eta3(i-1,j,k), eta3(i,j,k))
```

In 2D, only one transverse direction contributes.

### Step 3: Clamp wave speeds

The outermost HLL wave speeds are widened:

```
S_L = min(S_L, -eta)
S_R = max(S_R, +eta)
```

This ensures the Riemann fan is at least as wide as the transverse
eigenvalues, adding dissipation proportional to the transverse wave activity.

## Implementation Details

### Files modified

| File | Changes |
|------|---------|
| `src/mhd/mhd.hpp` | Added `eta1`, `eta2`, `eta3` device arrays and `use_hcorr` flag |
| `src/mhd/mhd.cpp` | Input parsing, compatibility check, array allocation |
| `src/mhd/mhd_fluxes.cpp` | Eta pre-computation kernel; updated HLLD call sites |
| `src/mhd/rsolvers/hlld_mhd.hpp` | Extended signature; wave speed correction in both adiabatic and isothermal branches |

### Backward compatibility

The HLLD function signature uses default arguments (`use_hcorr = false`,
empty Kokkos views for `eta_t1`/`eta_t2`), so all existing call sites
(including FOFC) remain valid without modification.

When `h_correction = false` (the default), no eta arrays are allocated and no
additional computation is performed. The solver behavior is identical to the
uncorrected version.

### Equation of state support

Both ideal gas (adiabatic) and isothermal EOS branches are corrected. The eta
kernel dispatches to the appropriate `IdealMHDFastSpeed` overload based on
`eos.is_ideal`.

### Dimensionality handling

- **1D**: The eta kernel is skipped entirely (no transverse directions exist).
- **2D**: Only `eta1` and `eta2` are meaningful. The third array (`eta3`) is
  allocated but the second transverse lookup in HLLD is guarded by
  `eta_t2.extent(0) > 0`, which evaluates to false for an empty view.
- **3D**: Both transverse directions contribute to `eta` at each interface.

## References

- Sanders, R., Morano, E., & Druguet, M.-C. (1998). "Multidimensional
  dissipation for upwind schemes: stability and applications to gas dynamics."
  *Journal of Computational Physics*, 145(2), 511-537.
- Miyoshi, T. & Kusano, K. (2005). "A multi-state HLL approximate Riemann
  solver for ideal magnetohydrodynamics." *Journal of Computational Physics*,
  208(1), 315-344.
