"""Regression test for a linear transverse Israel--Stewart telegraph mode."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_telegraph.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    values = np.loadtxt('build/src/rsrmhd_viscous_telegraph-errs.dat')
    velocity_error, stress_error = values[:2]
    if velocity_error > 2.0e-3 or stress_error > 2.0e-3:
        logger.warning('Viscous telegraph errors are too large: velocity=%g stress=%g',
                       velocity_error, stress_error)
        return False
    return True
