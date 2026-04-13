# Regression test for on-the-fly power spectrum output.
#
# Initializes a single-mode sound wave (wave_flag=0) along x1 on a
# 32^3 cubic periodic domain and checks that:
#
#  1. The velocity power spectrum peaks at wavenumber bin k=1.
#  2. The peak bin contains >99% of the total spectral power.
#  3. Results are consistent between single-block and multi-block runs.
#
# Requires: Athena_ENABLE_FFT=ON (default).

import glob
import logging
import sys

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_INPUT = 'tests/power_spectrum_hydro.athinput'
_CONCENTRATION_THRESHOLD = 0.99  # bin 1 must hold >= 99% of total power
_MATCH_TOL = 1.0e-10             # single vs multi-block relative tolerance


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(**kwargs):
    logger.debug('Running test ' + __name__)

    # --- Run 1: single MeshBlock (32^3) ---
    athena.run(_INPUT, [
        'job/basename=PwrSpec1',
        'meshblock/nx1=32', 'meshblock/nx2=32', 'meshblock/nx3=32',
    ])

    # --- Run 2: two MeshBlocks in x1 (16x32x32 each) ---
    athena.run(_INPUT, [
        'job/basename=PwrSpec2',
        'meshblock/nx1=16', 'meshblock/nx2=32', 'meshblock/nx3=32',
    ])


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def analyze():
    analyze_passed = True

    # --- Check Run 1 (single block) ---
    spec_files = sorted(glob.glob('build/src/PwrSpec1.*.00000.spec'))
    if len(spec_files) == 0:
        logger.warning('No spectrum file found for single-block run')
        return False

    data1 = np.loadtxt(spec_files[0])
    bins1 = data1[:, 0].astype(int)
    power1 = data1[:, 1]

    # Check for NaN / Inf
    if np.any(~np.isfinite(power1)):
        logger.warning('NaN or Inf in single-block spectrum')
        analyze_passed = False

    # Check peak location
    peak_bin = bins1[np.argmax(power1)]
    if peak_bin != 1:
        logger.warning('Single-block: peak at bin %d, expected bin 1', peak_bin)
        analyze_passed = False

    # Check spectral concentration
    total_power = np.sum(power1)
    if total_power <= 0:
        logger.warning('Single-block: total power is non-positive')
        analyze_passed = False
    else:
        concentration = power1[0] / total_power  # bin 1 is index 0
        if concentration < _CONCENTRATION_THRESHOLD:
            logger.warning(
                'Single-block: bin-1 concentration %.6f < threshold %.2f',
                concentration, _CONCENTRATION_THRESHOLD)
            analyze_passed = False
        else:
            logger.debug('Single-block: bin-1 concentration = %.6f',
                         concentration)

    # --- Check Run 2 (multi-block) ---
    spec_files2 = sorted(glob.glob('build/src/PwrSpec2.*.00000.spec'))
    if len(spec_files2) == 0:
        logger.warning('No spectrum file found for multi-block run')
        return False

    data2 = np.loadtxt(spec_files2[0])
    power2 = data2[:, 1]

    # Check for NaN / Inf
    if np.any(~np.isfinite(power2)):
        logger.warning('NaN or Inf in multi-block spectrum')
        analyze_passed = False

    # Check peak location
    peak_bin2 = data2[:, 0].astype(int)[np.argmax(power2)]
    if peak_bin2 != 1:
        logger.warning('Multi-block: peak at bin %d, expected bin 1', peak_bin2)
        analyze_passed = False

    # Check single-block vs multi-block consistency
    if total_power > 0:
        rel_diff = np.max(np.abs(power1 - power2)) / total_power
        if rel_diff > _MATCH_TOL:
            logger.warning(
                'Single vs multi-block max relative difference: %.6e',
                rel_diff)
            analyze_passed = False
        else:
            logger.debug(
                'Single vs multi-block match: max rel diff = %.6e', rel_diff)

    return analyze_passed
