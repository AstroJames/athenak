"""Exact point-inversion symmetry regression for three-dimensional Newtonian MHD."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_INPUT = 'mhd/orszag_tang.athinput'
_LAYOUTS = (('single', 16), ('multiblock', 8))


def _basename(layout):
    return 'mhd_point_symmetry_3d_wenoz_' + layout


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    common = [
        'mesh/nghost=3',
        'mesh/nx1=16',
        'mesh/nx2=16',
        'mesh/nx3=16',
        'time/integrator=rk3',
        'time/nlim=10',
        'mhd/rsolver=hlld',
        'mhd/reconstruct=wenoz',
        'problem/check_symmetry=true',
        'problem/three_dimensional=true',
        'output1/dt=-1.0',
        'output2/dt=-1.0',
        'output3/dt=-1.0',
    ]
    for layout, block_size in _LAYOUTS:
        basename = _basename(layout)
        filename = 'build/src/' + basename + '-symmetry.dat'
        if os.path.exists(filename):
            os.remove(filename)
        athena.run(_INPUT, common + [
            'job/basename=' + basename,
            'meshblock/nx1=' + repr(block_size),
            'meshblock/nx2=' + repr(block_size),
            'meshblock/nx3=' + repr(block_size),
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    residual_columns = list(range(5, 39)) + [40, 41, 43, 44]
    magnitude_columns = [39, 42, 45]
    for layout, _ in _LAYOUTS:
        basename = _basename(layout)
        filename = 'build/src/' + basename + '-symmetry.dat'
        data = np.loadtxt(filename, ndmin=2)
        if data.shape != (1, 46) or not np.all(np.isfinite(data)):
            logger.warning('Invalid symmetry diagnostics in %s: %s', filename, data)
            return False
        if not np.array_equal(data[0, :4], (16, 16, 16, 10)):
            logger.warning('Wrong mesh or cycle count in %s: %s',
                           filename, data[0, :4])
            return False
        if np.any(data[0, residual_columns] != 0.0):
            logger.warning('Point-inversion symmetry broke in %s: %s',
                           filename, data[0, residual_columns])
            return False
        if np.any(data[0, magnitude_columns] == 0.0):
            logger.warning('A corner-electric component was not exercised in %s: %s',
                           filename, data[0, magnitude_columns])
            return False
    return True
