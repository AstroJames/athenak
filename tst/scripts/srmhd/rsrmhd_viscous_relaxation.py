"""Regression test for homogeneous Israel--Stewart shear relaxation."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_relaxation.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    stress_error, fluid_error, exact = np.loadtxt(
        'build/src/rsrmhd_viscous_relaxation-errs.dat')
    if stress_error > 5.0e-4 or fluid_error > 2.0e-10:
        logger.warning('Viscous relaxation errors are too large: '
                       'stress=%g fluid=%g exact=%g',
                       stress_error, fluid_error, exact)
        return False
    return True
