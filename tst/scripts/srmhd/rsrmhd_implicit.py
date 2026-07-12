"""Regression test for the local resistive-SRMHD implicit stage recovery."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_roundtrip.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_implicit-errs.dat')
    failures, primitive_error, electric_error, residual_error, max_iterations = data

    threshold = 2.0e-10
    if failures != 0:
        logger.warning('Implicit SRRMHD recovery failed in %d states', int(failures))
        return False
    if max(primitive_error, electric_error, residual_error) > threshold:
        logger.warning('Implicit SRRMHD errors exceed threshold: %s', data)
        return False
    if max_iterations >= 30:
        logger.warning('Implicit SRRMHD recovery exhausted its iteration budget')
        return False
    return True
