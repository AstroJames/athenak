"""Regression test for conservative one-dimensional relativistic-shear transport."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_transport.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    l1_error, max_error, flux_error = np.loadtxt(
        'build/src/rsrmhd_viscous_transport-errs.dat')
    if l1_error > 2.0e-3 or max_error > 4.0e-3 or flux_error > 2.0e-10:
        logger.warning('Viscous transport errors are too large: '
                       'L1=%g max=%g flux=%g', l1_error, max_error, flux_error)
        return False
    return True
