"""Regression test for relativistic shear-stress algebra and relaxation."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_roundtrip.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_viscosity_algebra-errs.dat')
    errors = data[:6]
    enabled = int(data[6])
    nu, tau, chi_max, causal_margin = data[7:11]

    threshold = 2.0e-11
    if np.any(errors > threshold):
        logger.warning('Relativistic-viscosity algebra errors exceed threshold: '
                       'errors=%s threshold=%g', errors, threshold)
        return False
    if enabled != 1:
        logger.warning('Relativistic viscosity was not enabled by the test input')
        return False
    if abs(nu - 0.01) > threshold or abs(tau - 0.1) > threshold:
        logger.warning('Unexpected viscosity configuration: nu=%g tau=%g', nu, tau)
        return False
    if abs(chi_max - 2.0) > threshold:
        logger.warning('Unexpected shear inverse-Reynolds limit: chi_max=%g', chi_max)
        return False
    if causal_margin < 0.0:
        logger.warning('Test viscosity parameters violate the causal gate: margin=%g',
                       causal_margin)
        return False
    return True
