"""Compatible dual-CT charge-conservation operator regression."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for name, block_size in (('single', 8), ('multi', 4)):
        basename = 'rsrmhd_ect_' + name
        error_file = 'build/src/' + basename + '-errs.dat'
        if os.path.exists(error_file):
            os.remove(error_file)
        athena.run('tests/rsrmhd_ect.athinput', [
            'job/basename=' + basename,
            'meshblock/nx1=' + repr(block_size),
            'meshblock/nx2=' + repr(block_size),
            'meshblock/nx3=' + repr(block_size),
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.asarray([
        np.loadtxt('build/src/rsrmhd_ect_single-errs.dat'),
        np.loadtxt('build/src/rsrmhd_ect_multi-errs.dat'),
    ])
    if data.shape != (2, 5) or not np.all(np.isfinite(data)):
        logger.warning('Dual-CT diagnostics are invalid: %s', data)
        return False
    if np.any(data[:, 0] > 2.0e-13):
        logger.warning('Dual-CT charge update is not exact: %s', data[:, 0])
        return False
    if np.any(data[:, 1] > 2.0e-13):
        logger.warning('Dual-CT edge curl is incorrect: %s', data[:, 1])
        return False
    if np.any(data[:, 2] > 2.0e-11):
        logger.warning('Face-to-cell electric average is inconsistent: %s', data[:, 2])
        return False
    if np.any(data[:, 3] != 1) or np.any(data[:, 4] != 0):
        logger.warning('Dual-CT mode or recovery state is invalid: %s', data[:, 3:])
        return False
    if not np.array_equal(data[0], data[1]):
        logger.warning('Single- and eight-block operator results differ: %s', data)
        return False
    return True
