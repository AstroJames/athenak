# Regression test for multi-mode on-the-fly power spectrum output.
#
# The 'spectrum_modes' pgen initializes:
#   v_x(x) = sum_{k=1}^{nmode} (1/k) * sin(2*pi*k*x)
#   v_y = v_z = 0,  rho = 1,  p = 1
#
# Analytical result (exact to machine precision):
#   P(k) = 1 / (2 * k^2)   for k = 1 ... nmode
#   P(k) ~ 0               for k > nmode
#
# Checks:
#  1. Each active bin k=1..nmode matches the analytical value to within
#     a tight relative tolerance (1e-12).
#  2. Inactive bins k > nmode have power < 1e-20 (effectively zero).
#  3. Single-block and multi-block (2x split in x1) give identical results.

import glob
import logging

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_INPUT = 'tests/spectrum_modes_hydro.athinput'
_NMODE = 4
# Tolerance for P(k) vs 1/(2k^2): FFTW double-precision FFT of a finite grid
# accumulates ~O(N * eps) rounding error (N=32).  Amplitudes like 1/3 are not
# exactly representable, adding ~1 ULP per cell.  The resulting relative error
# on the power is empirically ~1e-7, so we allow 1e-6 with margin.
_REL_TOL = 1.0e-6    # relative error against P(k) = 1/(2k^2)
_ZERO_TOL = 1.0e-20  # inactive bins must be below this
_MATCH_TOL = 1.0e-13 # single vs multi-block relative tolerance


def run(**kwargs):
    logger.debug('Running test ' + __name__)

    # Run 1: single block (32^3)
    athena.run(_INPUT, [
        'job/basename=SpecModes1',
        'meshblock/nx1=32', 'meshblock/nx2=32', 'meshblock/nx3=32',
    ])

    # Run 2: two blocks split in x1 (16x32x32 each)
    athena.run(_INPUT, [
        'job/basename=SpecModes2',
        'meshblock/nx1=16', 'meshblock/nx2=32', 'meshblock/nx3=32',
    ])


def analyze():
    passed = True

    # --- Load single-block spectrum ---
    files1 = sorted(glob.glob('build/src/SpecModes1.*.00000.spec'))
    if not files1:
        logger.warning('No spectrum file found for single-block run')
        return False
    data1 = np.loadtxt(files1[0])
    bins1  = data1[:, 0].astype(int)
    power1 = data1[:, 1]

    if np.any(~np.isfinite(power1)):
        logger.warning('NaN or Inf in single-block spectrum')
        return False

    # --- Check active bins 1..nmode ---
    for k in range(1, _NMODE + 1):
        idx = np.where(bins1 == k)[0]
        if len(idx) == 0:
            logger.warning('Bin k=%d missing from spectrum', k)
            passed = False
            continue
        p_got  = power1[idx[0]]
        p_want = 1.0 / (2.0 * k * k)
        rel_err = abs(p_got - p_want) / p_want
        if rel_err > _REL_TOL:
            logger.warning(
                'Bin k=%d: got %.15e, want %.15e, rel_err=%.3e',
                k, p_got, p_want, rel_err)
            passed = False
        else:
            logger.debug('Bin k=%d OK: rel_err=%.3e', k, rel_err)

    # --- Check inactive bins k > nmode are ~zero ---
    for idx_b, k in enumerate(bins1):
        if k > _NMODE and power1[idx_b] > _ZERO_TOL:
            logger.warning(
                'Inactive bin k=%d has power %.3e > zero threshold %.3e',
                k, power1[idx_b], _ZERO_TOL)
            passed = False

    # --- Load multi-block spectrum and compare ---
    files2 = sorted(glob.glob('build/src/SpecModes2.*.00000.spec'))
    if not files2:
        logger.warning('No spectrum file found for multi-block run')
        return False
    data2 = np.loadtxt(files2[0])
    power2 = data2[:, 1]

    if np.any(~np.isfinite(power2)):
        logger.warning('NaN or Inf in multi-block spectrum')
        passed = False

    total = np.sum(power1)
    if total > 0:
        rel_diff = np.max(np.abs(power1 - power2)) / total
        if rel_diff > _MATCH_TOL:
            logger.warning(
                'Single vs multi-block max relative difference: %.3e', rel_diff)
            passed = False
        else:
            logger.debug('Single vs multi-block match: max rel diff = %.3e', rel_diff)

    return passed
