"""One-cycle smoke test for the resistive-SRMHD IMEX2 task graph."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_imex_smoke.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_roundtrip-errs.dat')
    if not np.all(np.isfinite(data)):
        logger.warning('IMEX smoke test produced non-finite diagnostics: %s', data)
        return False
    counters = data[2:6]
    if np.any(counters != 0):
        logger.warning('IMEX smoke test used EOS corrections or failed recovery: %s',
                       counters)
        return False
    if int(data[6]) != 8:
        logger.warning('IMEX smoke test did not retain the eight-component state')
        return False
    dx1 = 1.0 / 16.0
    if data[7] > dx1 * (1.0 + 2.0e-13):
        logger.warning('SRRMHD timestep does not include the light-wave bound: %g > %g',
                       data[7], dx1)
        return False
    return True
