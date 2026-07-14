"""Regression test for relativistic mechanical-forcing algebra."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_roundtrip.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    errors = np.loadtxt('build/src/rsrmhd_forcing_algebra-errs.dat')
    threshold = 2.0e-11
    if np.any(errors > threshold):
        logger.warning('Relativistic-forcing algebra errors exceed threshold: '
                       'errors=%s threshold=%g', errors, threshold)
        return False
    return True
