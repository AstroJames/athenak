"""Serial multiblock equivalence for the face-staggered charged vortex."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    error_file = 'build/src/rsrmhd_ect_multiblock-errs.dat'
    if os.path.exists(error_file):
        os.remove(error_file)
    for block_size in (64, 32):
        athena.run('tests/rsrmhd_charged_vortex.athinput', [
            'job/basename=rsrmhd_ect_multiblock',
            'mhd/electric_ct=true',
            'mesh/nx1=64',
            'mesh/nx2=64',
            'meshblock/nx1=' + repr(block_size),
            'meshblock/nx2=' + repr(block_size),
            'time/nlim=-1',
            'time/tlim=5.0',
            'output1/dt=-1.0',
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_ect_multiblock-errs.dat', ndmin=2)
    if data.shape != (2, 11) or not np.all(np.isfinite(data)):
        logger.warning('Dual-CT multiblock diagnostics are invalid: %s', data)
        return False
    if not np.array_equal(data[:, :2], ((64, 32), (64, 32))):
        logger.warning('Dual-CT multiblock resolution/cycles are wrong: %s', data[:, :2])
        return False
    if not np.allclose(data[0, 2:6], data[1, 2:6], rtol=0.0, atol=5.0e-14):
        logger.warning('Single/multiblock solution norms differ: %s', data[:, 2:6])
        return False
    if np.any(data[:, 6] > 1.0e-13) or np.any(data[:, 7] != 0.0) or \
            np.any(data[:, 8] > 5.0e-14) or np.any(data[:, 9] != 0.0):
        logger.warning('Multiblock constraints/recovery failed: %s', data[:, 6:10])
        return False
    if np.any(data[:, 10] != 5.0):
        logger.warning('Multiblock final times are wrong: %s', data[:, 10])
        return False
    return True
