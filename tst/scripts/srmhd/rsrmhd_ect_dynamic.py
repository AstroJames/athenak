"""One-step dynamic charge-conserving dual-CT regression."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    error_file = 'build/src/rsrmhd_ect_dynamic-errs.dat'
    if os.path.exists(error_file):
        os.remove(error_file)
    athena.run('tests/rsrmhd_ect_dynamic.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.atleast_1d(
        np.loadtxt('build/src/rsrmhd_ect_dynamic-errs.dat'))
    if data.shape != (6,) or not np.all(np.isfinite(data)):
        logger.warning('Dynamic dual-CT diagnostics are invalid: %s', data)
        return False
    if data[0] > 5.0e-13:
        logger.warning('Dynamic charge continuity is not exact: %s', data[0])
        return False
    if np.any(data[1:4] > 5.0e-12):
        logger.warning('Dynamic face/source/mirror errors are too large: %s', data[1:4])
        return False
    if data[4] != 0 or data[5] != 1:
        logger.warning('Dynamic recovery or cycle count is invalid: %s', data[4:])
        return False
    return True
