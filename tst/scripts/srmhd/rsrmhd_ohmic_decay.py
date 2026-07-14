"""Strong-guide-field Harris-sheet Ohmic-decay regression."""

import glob
import logging
import os
import sys

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import athena_read  # noqa

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    error_file = 'build/src/rsrmhd_ohmic_decay-errs.dat'
    if os.path.exists(error_file):
        os.remove(error_file)
    athena.run('tests/rsrmhd_ohmic_decay.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_ohmic_decay-errs.dat')
    if data.shape != (8,) or not np.all(np.isfinite(data)):
        logger.warning('Ohmic-decay diagnostics are invalid: %s', data)
        return False

    nx1, _, peak, peak_model, moment, moment_model, failures, time = data
    if nx1 != 512 or failures != 0 or abs(time - 0.2) > 1.0e-12:
        logger.warning('Ohmic-decay run metadata are wrong: %s', data)
        return False
    if abs(peak/peak_model - 1.0) > 0.05:
        logger.warning('Peak current misses the diffusive model: %s %s',
                       peak, peak_model)
        return False
    if abs(moment/moment_model - 1.0) > 0.08:
        logger.warning('Current-squared width misses the diffusive model: %s %s',
                       moment, moment_model)
        return False

    output_files = sorted(glob.glob(
        'build/src/tab/rsrmhd_ohmic_decay.primitive.*.tab'))
    if len(output_files) != 2:
        logger.warning('Expected initial and final Ohmic-decay tables: %s',
                       output_files)
        return False
    initial = athena_read.tab(output_files[0])
    required = ('dens', 'velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3')
    if not all(name in initial for name in required):
        logger.warning('Ohmic-decay output labels are missing: %s', initial.keys())
        return False
    if np.max(np.abs(initial['dens'] - 2.02)) > 1.0e-12:
        logger.warning('Initial density does not match sigma_hot=10')
        return False
    velocity_max = max(np.max(np.abs(initial[name]))
                       for name in ('velx', 'vely', 'velz'))
    if velocity_max > 1.0e-14:
        logger.warning('Initial Harris sheet is not stationary: %s', velocity_max)
        return False
    magnetic_norm = (initial['bcc1']**2 + initial['bcc2']**2
                     + initial['bcc3']**2)
    if np.max(np.abs(magnetic_norm - 101.0)) > 1.0e-10:
        logger.warning('Initial guide-field balance is wrong')
        return False
    return True
