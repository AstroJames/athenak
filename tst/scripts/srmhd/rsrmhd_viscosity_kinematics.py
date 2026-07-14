"""Regression test for covariant viscous kinematics and characteristic speeds."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_roundtrip.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_viscosity_kinematics-errs.dat')
    errors = data[:26]
    transverse_speed, longitudinal_speed = data[26:28]

    threshold = 2.0e-11
    if np.any(errors > threshold):
        logger.warning('Relativistic-viscosity kinematics errors exceed threshold: '
                       'errors=%s threshold=%g', errors, threshold)
        return False
    if abs(transverse_speed - 0.1) > threshold:
        logger.warning('Unexpected transverse shear speed squared: %g',
                       transverse_speed)
        return False
    expected_longitudinal = 0.2 + 4.0*0.01/(3.0*0.1)
    if abs(longitudinal_speed - expected_longitudinal) > threshold:
        logger.warning('Unexpected longitudinal viscous speed squared: %g',
                       longitudinal_speed)
        return False
    return True
