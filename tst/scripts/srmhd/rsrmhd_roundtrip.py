"""Regression test for known-E resistive-SRMHD primitive recovery."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_roundtrip.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_roundtrip-errs.dat')
    max_rel, max_abs = data[0:2]
    counters = data[2:6]
    nmhd = int(data[6])

    threshold = 2.0e-11
    if max_rel > threshold or max_abs > threshold:
        logger.warning('Known-E round-trip error exceeds threshold: '
                       'relative=%g absolute=%g threshold=%g',
                       max_rel, max_abs, threshold)
        return False
    if np.any(counters != 0):
        logger.warning('Known-E round trip unexpectedly used EOS corrections: %s',
                       counters)
        return False
    if nmhd != 8:
        logger.warning('Resistive SRMHD allocated %d variables instead of 8', nmhd)
        return False
    return True
