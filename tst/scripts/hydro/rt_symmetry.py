"""Exact x-reflection symmetry regression for the Xu--Shu hydro RT problem."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_INPUT = 'hydro/rt2d_xu_shu.athinput'
_RECONSTRUCTIONS = ('plm', 'wenoz', 'teno5', 'teno5_opt')
_CYCLES = (0, 1, 10)


def _basename(reconstruction, cycles):
    return 'rt_symmetry_' + reconstruction + '_' + repr(cycles)


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    common = [
        'mesh/nghost=3',
        'mesh/nx1=64',
        'mesh/nx2=256',
        'meshblock/nx1=32',
        'meshblock/nx2=64',
        'time/integrator=rk3',
        'hydro/rsolver=hllc',
        'problem/check_symmetry=true',
        'output1/dt=-1.0',
        'output2/dt=-1.0',
    ]
    for reconstruction in _RECONSTRUCTIONS:
        for cycles in _CYCLES:
            basename = _basename(reconstruction, cycles)
            filename = 'build/src/' + basename + '-symmetry.dat'
            if os.path.exists(filename):
                os.remove(filename)
            athena.run(_INPUT, common + [
                'job/basename=' + basename,
                'time/nlim=' + repr(cycles),
                'hydro/reconstruct=' + reconstruction,
            ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    for reconstruction in _RECONSTRUCTIONS:
        for cycles in _CYCLES:
            basename = _basename(reconstruction, cycles)
            filename = 'build/src/' + basename + '-symmetry.dat'
            data = np.loadtxt(filename, ndmin=2)
            if data.shape != (1, 45) or not np.all(np.isfinite(data)):
                logger.warning('Invalid symmetry diagnostics in %s: %s', filename, data)
                return False
            if not np.array_equal(data[0, :4], (64, 256, 1, cycles)):
                logger.warning('Wrong mesh or cycle count in %s: %s',
                               filename, data[0, :4])
                return False
            # Flux arrays have not been computed when nlim=0, so only inspect
            # the state diagnostics for the initial-condition check.
            residuals = data[0, 5:25] if cycles == 0 else data[0, 5:]
            if np.any(residuals != 0.0):
                logger.warning('X-reflection symmetry broke in %s: %s',
                               filename, residuals)
                return False
    return True
