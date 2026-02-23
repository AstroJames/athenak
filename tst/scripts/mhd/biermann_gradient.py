# Regression test for Biermann battery generation with misaligned gradients.
#
# Uses a manufactured 2D state with:
#   rho(x1) = rho0 + drho_dx1*x1
#   p(x2)   = p0 + dp_dx2*x2
# and checks that the generated Bz after one tiny RK1 step matches the expected discrete
# update from the implemented Biermann E-field stencil. Also checks linear scaling with
# coefficient.

import glob
import logging
import re
import sys

import numpy as np
import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import athena_read  # noqa

athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])

_INPUT = 'tests/biermann_gradient.athinput'
_ID = 'bz'

_RHO0 = 1.0
_DRHO_DX1 = 0.5
_DP_DX2 = 0.25

_COEFF_1 = 1.0e-3
_COEFF_2 = 2.0e-3

_INTERIOR_PAD = 2
_ERR_TOL = 1.0e-11
_SCALE_TOL = 1.0e-10


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    args_common = [
        'time/nlim=1',
        'time/integrator=rk1',
        'time/cfl_number=1.0e-4',
        'output1/dt=1.0e-8',
        'output1/id=' + _ID
    ]
    athena.run(_INPUT, args_common + [
        'job/basename=biermann_grad_c1',
        'mhd/biermann_coeff={:.16e}'.format(_COEFF_1)
    ])
    athena.run(_INPUT, args_common + [
        'job/basename=biermann_grad_c2',
        'mhd/biermann_coeff={:.16e}'.format(_COEFF_2)
    ])


def _latest_tab(basename):
    pattern = 'build/src/tab/{}.{}.*.tab'.format(basename, _ID)
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise RuntimeError('No tab files found for {}'.format(basename))

    def _idx(path):
        match = re.search(r'\.(\d{5})\.tab$', path)
        if match is None:
            raise RuntimeError('Could not parse index from {}'.format(path))
        return int(match.group(1))

    return max(paths, key=_idx)


def _lineout_x1(data):
    x1 = np.asarray(data['x1v']).ravel()
    b3 = np.asarray(data['bcc3']).ravel()

    if 'j' in data:
        j = np.asarray(data['j']).astype(int).ravel()
        j_target = np.sort(np.unique(j))[len(np.unique(j)) // 2]
        mask = (j == j_target)
    elif 'x2v' in data:
        x2 = np.asarray(data['x2v']).ravel()
        x2_unique = np.sort(np.unique(x2))
        x2_target = x2_unique[len(x2_unique) // 2]
        mask = np.isclose(x2, x2_target, rtol=0.0, atol=1.0e-14)
    else:
        # 1D tab output: data are already on an x1 line.
        mask = np.ones_like(x1, dtype=bool)

    x1_row = x1[mask]
    b3_row = b3[mask]
    order = np.argsort(x1_row)
    return x1_row[order], b3_row[order]


def _expected_bz_discrete(x1, coeff, time):
    nx = len(x1)
    dx1 = x1[1] - x1[0]
    rho = _RHO0 + _DRHO_DX1 * x1
    inv_rho = 1.0 / rho

    # Edge-centered E2 in 2D implementation:
    # E2(i) = -C * dp_dx2 * avg[1/rho(i), 1/rho(i-1)]
    e2 = np.zeros(nx + 1)
    e2[0] = -coeff * _DP_DX2 * inv_rho[0]
    e2[nx] = -coeff * _DP_DX2 * inv_rho[nx - 1]
    e2[1:nx] = -coeff * _DP_DX2 * 0.5 * (inv_rho[1:nx] + inv_rho[0:nx - 1])

    # CT update for B3 in 2D, with E1 = 0 for this manufactured state.
    b3 = np.zeros(nx)
    b3[:] = -time * (e2[1:nx + 1] - e2[0:nx]) / dx1
    return b3


def _max_abs_err(a, b):
    return np.max(np.abs(a - b))


def analyze():
    logger.debug('Analyzing test ' + __name__)
    status = True

    path1 = _latest_tab('biermann_grad_c1')
    path2 = _latest_tab('biermann_grad_c2')

    data1 = athena_read.tab(path1)
    data2 = athena_read.tab(path2)

    x1_1, b3_1 = _lineout_x1(data1)
    x1_2, b3_2 = _lineout_x1(data2)

    if len(x1_1) != len(x1_2) or np.max(np.abs(x1_1 - x1_2)) > 0.0:
        logger.warning('x1 grids do not match between coefficient runs')
        return False

    time1 = float(data1['time'])
    time2 = float(data2['time'])
    if abs(time1 - time2) > 0.0:
        logger.warning('times do not match between runs: %e vs %e', time1, time2)
        return False

    b3_exp_1 = _expected_bz_discrete(x1_1, _COEFF_1, time1)
    b3_exp_2 = _expected_bz_discrete(x1_2, _COEFF_2, time2)

    sl = slice(_INTERIOR_PAD, len(x1_1) - _INTERIOR_PAD)
    err1 = _max_abs_err(b3_1[sl], b3_exp_1[sl])
    err2 = _max_abs_err(b3_2[sl], b3_exp_2[sl])
    if err1 > _ERR_TOL:
        logger.warning('c1 max abs error too large: %e > %e', err1, _ERR_TOL)
        status = False
    if err2 > _ERR_TOL:
        logger.warning('c2 max abs error too large: %e > %e', err2, _ERR_TOL)
        status = False

    # Scaling check: Bz(C2) should equal 2*Bz(C1) for this setup.
    scale_err = _max_abs_err(b3_2[sl], 2.0 * b3_1[sl])
    if scale_err > _SCALE_TOL:
        logger.warning('scaling error too large: %e > %e', scale_err, _SCALE_TOL)
        status = False

    # Sign check for positive drho/dx1, dp/dx2, and coeff: Bz should be negative.
    if np.any(b3_1[sl] >= 0.0):
        logger.warning('expected negative Bz in interior for c1 run')
        status = False

    return status
