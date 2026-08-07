"""Regression for magnetization-normalized random magnetic initialization."""

import logging

import numpy as np

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_sigma_seed.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    history = np.atleast_2d(np.loadtxt(
        'build/src/rsrmhd_sigma_seed.user.hst'))
    if history.ndim != 2 or history.shape[1] != 12 or not np.all(np.isfinite(history)):
        logger.warning('Sigma-seed history is invalid: shape=%s', history.shape)
        return False

    # rho=1, P=1, gamma=4/3 gives w0=5.  Thus sigma_rms=0.01 sets
    # <B^2>=0.05 and the unit-volume magnetic energy to 0.025 exactly.
    if not np.allclose(history[:, 4], 0.025, rtol=1.0e-12, atol=1.0e-14):
        logger.warning('sigma_rms produced incorrect magnetic energy: %g',
                       history[-1, 4])
        return False
    if np.any(history[:, 3] != 0.0) or np.any(history[:, 5] != 0.0):
        logger.warning('Sigma seed generated kinetic or electric energy: %s',
                       history[-1, [3, 5]])
        return False
    if np.any(history[:, 11] > 1.0e-20):
        logger.warning('Sigma seed violates the divergence constraint: %g',
                       history[-1, 11])
        return False
    return True
