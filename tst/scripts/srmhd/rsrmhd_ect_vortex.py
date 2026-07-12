"""Single-block 2D charged vortex with face-staggered electric fields."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    error_file = 'build/src/rsrmhd_ect_vortex-errs.dat'
    if os.path.exists(error_file):
        os.remove(error_file)
    for resolution in (32, 64, 128):
        athena.run('tests/rsrmhd_charged_vortex.athinput', [
            'job/basename=rsrmhd_ect_vortex',
            'mhd/electric_ct=true',
            'mesh/nx1=' + repr(resolution),
            'mesh/nx2=' + repr(resolution),
            'meshblock/nx1=' + repr(resolution),
            'meshblock/nx2=' + repr(resolution),
            'time/nlim=-1',
            'time/tlim=5.0',
            'output1/dt=-1.0',
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_ect_vortex-errs.dat', ndmin=2)
    if data.shape != (3, 11) or not np.all(np.isfinite(data)):
        logger.warning('Staggered vortex diagnostics are invalid: %s', data)
        return False
    if not np.array_equal(data[:, :2], ((32, 16), (64, 32), (128, 64))):
        logger.warning('Staggered vortex resolution/cycles are wrong: %s', data[:, :2])
        return False
    q_rates = np.log2(data[:-1, 2]/data[1:, 2])
    p_rates = np.log2(data[:-1, 4]/data[1:, 4])
    if np.any(q_rates < (1.2, 1.6)) or np.any(p_rates < (1.3, 1.8)):
        logger.warning('Staggered vortex convergence regressed: q=%s p=%s',
                       q_rates, p_rates)
        return False
    if np.any(data[:, 6] > 1.0e-13) or np.any(data[:, 7] != 0.0) or \
            np.any(data[:, 8] > 5.0e-14) or np.any(data[:, 9] != 0.0):
        logger.warning('Staggered vortex constraints/recovery failed: %s', data[:, 6:10])
        return False
    return True
