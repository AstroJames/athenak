# TENO6 reconstruction

> [!WARNING]
> `teno6` and `teno6_opt` are currently supported only for hydrodynamics. AthenaK
> deliberately rejects these methods for MHD and RMHD calculations. Three-dimensional
> oblique slow-wave tests found a non-convergent constrained-transport mode, including
> when first-order flux correction was enabled. Use `teno5` or `teno5_opt` for MHD/RMHD
> until a compatible stabilization has been implemented and validated.

The hydrodynamics module provides two six-point targeted ENO reconstructions:

- `teno6` uses a sixth-order central background operator.
- `teno6_opt` uses a fifth-order, slightly upwind-biased background operator with
  controlled high-wavenumber dissipation.

Select either method with `hydro/reconstruct`. The default stencil-selection cutoff is
`hydro/teno_cutoff = 1.0e-7`. Valid cutoff values are in `(0, 0.25]`.

Hydrodynamic validation covers three-dimensional linear-wave convergence and the
first-order flux-correction path. MHD/RMHD support should not be enabled solely from
one- or two-dimensional results; it requires convergence of the three-dimensional
oblique MHD wave matrix and multidimensional shock tests.
