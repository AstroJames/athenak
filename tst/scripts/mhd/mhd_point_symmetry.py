"""Exact point-reflection symmetry regression for Newtonian MHD."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_INPUT = 'mhd/orszag_tang.athinput'
_RECONSTRUCTIONS = ('plm', 'wenoz', 'teno5', 'teno5_opt')
_LAYOUTS = (('single', 64), ('multiblock', 32))


def _basename(reconstruction, layout):
    return 'mhd_point_symmetry_' + reconstruction + '_' + layout


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    common = [
        'mesh/nghost=3',
        'mesh/nx1=64',
        'mesh/nx2=64',
        'time/integrator=rk3',
        'time/nlim=10',
        'mhd/rsolver=hlld',
        'problem/check_symmetry=true',
        'output1/dt=-1.0',
        'output2/dt=-1.0',
        'output3/dt=-1.0',
    ]
    for reconstruction in _RECONSTRUCTIONS:
        for layout, block_size in _LAYOUTS:
            basename = _basename(reconstruction, layout)
            filename = 'build/src/' + basename + '-symmetry.dat'
            if os.path.exists(filename):
                os.remove(filename)
            athena.run(_INPUT, common + [
                'job/basename=' + basename,
                'meshblock/nx1=' + repr(block_size),
                'meshblock/nx2=' + repr(block_size),
                'mhd/reconstruct=' + reconstruction,
            ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    for reconstruction in _RECONSTRUCTIONS:
        for layout, _ in _LAYOUTS:
            basename = _basename(reconstruction, layout)
            filename = 'build/src/' + basename + '-symmetry.dat'
            data = np.loadtxt(filename, ndmin=2)
            if data.shape != (1, 31) or not np.all(np.isfinite(data)):
                logger.warning('Invalid symmetry diagnostics in %s: %s', filename, data)
                return False
            if not np.array_equal(data[0, :4], (64, 64, 1, 10)):
                logger.warning('Wrong mesh or cycle count in %s: %s',
                               filename, data[0, :4])
                return False
            if np.any(data[0, 5:] != 0.0):
                logger.warning('Point-reflection symmetry broke in %s: %s',
                               filename, data[0, 5:])
                return False
    return True
