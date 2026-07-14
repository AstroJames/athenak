"""Regression test for block-eliminated electric-plus-shear recovery."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_roundtrip.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_viscous_implicit-errs.dat')
    failures = data[0]
    errors = data[1:9]
    max_iterations = data[9]
    persistent_state_error = data[10]

    threshold = 2.0e-10
    if failures != 0:
        logger.warning('Viscous implicit recovery failed in %d states', int(failures))
        return False
    if np.any(errors > threshold):
        logger.warning('Viscous implicit recovery errors exceed threshold: '
                       'errors=%s threshold=%g', errors, threshold)
        return False
    if max_iterations >= 30:
        logger.warning('Viscous implicit recovery exhausted its iteration budget')
        return False
    if persistent_state_error > threshold:
        logger.warning('Persistent conservative stress error exceeds threshold: %g',
                       persistent_state_error)
        return False
    return True
