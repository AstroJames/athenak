# Regression test for Biermann battery generation with misaligned gradients.
#
# Uses a manufactured 2D state with:
#   rho(x1) = rho0 + drho_dx1*x1
#   p(x2)   = p0 + dp_dx2*x2
# and performs three checks:
#
#  1. Stencil accuracy / linear scaling: B3 after one RK1 step matches the
#     discrete stencil prediction to round-off, and scales linearly with coeff.
#
#  2. Null test: with dp_dx2=0 (no pressure gradient), B3 stays exactly zero.
#
#  3. Energy flux: the Biermann Poynting flux E_batt x B is verified by
#     comparing the change in total energy between a bz0=0 baseline run and a
#     bz0>0 run. The difference (dE_bz - dE_0)/bz0 must match the ΔB3 stencil
#     prediction, confirming that BiermannEnergyFlux adds the correct E2*B3
#     contribution to the energy flux.

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

_RHO0     = 1.0
_DRHO_DX1 = 0.5
_DP_DX2   = 0.25

_COEFF_1  = 1.0e-3
_COEFF_2  = 2.0e-3
_COEFF_EN = 1.0       # larger coeff for energy test to boost signal
_BZ0_TEST = 1.0e-3   # uniform initial Bz for energy flux test

_INTERIOR_PAD = 2
_ERR_TOL      = 1.0e-11   # stencil absolute error tolerance
_SCALE_TOL    = 1.0e-10   # linear-scaling absolute error tolerance
_NULL_TOL     = 0.0       # null test: B3 must be exactly zero
_ENERGY_TOL   = 1.0e-5    # energy flux relative error tolerance


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(**kwargs):
    logger.debug('Running test ' + __name__)
    args_common = [
        'time/nlim=1',
        'time/integrator=rk1',
        'time/cfl_number=1.0e-4',
        'output1/dt=1.0e-8',
        'output2/dt=1.0e-8',
    ]

    # --- stencil / scaling runs (bz output only) ---
    athena.run(_INPUT, args_common + [
        'job/basename=biermann_grad_c1',
        'mhd/biermann_coeff={:.16e}'.format(_COEFF_1),
        'output1/id=bz',
        'output2/id=en',
    ])
    athena.run(_INPUT, args_common + [
        'job/basename=biermann_grad_c2',
        'mhd/biermann_coeff={:.16e}'.format(_COEFF_2),
        'output1/id=bz',
        'output2/id=en',
    ])

    # --- null test: no pressure gradient -> no Biermann source ---
    athena.run(_INPUT, args_common + [
        'job/basename=biermann_null',
        'mhd/biermann_coeff={:.16e}'.format(_COEFF_1),
        'problem/dp_dx2=0.0',
        'problem/dp_dx1=0.0',
        'output1/id=bz',
        'output2/id=en',
    ])

    # --- energy flux runs: baseline (bz0=0) and bz0>0 ---
    athena.run(_INPUT, args_common + [
        'job/basename=biermann_en0',
        'mhd/biermann_coeff={:.16e}'.format(_COEFF_EN),
        'problem/bz0=0.0',
        'output1/id=bz',
        'output2/id=en',
    ])
    athena.run(_INPUT, args_common + [
        'job/basename=biermann_enbz',
        'mhd/biermann_coeff={:.16e}'.format(_COEFF_EN),
        'problem/bz0={:.16e}'.format(_BZ0_TEST),
        'output1/id=bz',
        'output2/id=en',
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tab_files(basename, fid):
    """Return all tab files for basename.fid, sorted by output index."""
    pattern = 'build/src/tab/{}.{}.*.tab'.format(basename, fid)
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise RuntimeError('No tab files found matching {}'.format(pattern))

    def _idx(p):
        m = re.search(r'\.(\d{5})\.tab$', p)
        if m is None:
            raise RuntimeError('Could not parse index from {}'.format(p))
        return int(m.group(1))

    return sorted(paths, key=_idx)


def _read_x1_col(path, col):
    """Read tab file and return (x1, values) sorted by x1."""
    data = np.loadtxt(path, comments='#')
    x1  = data[:, 2]
    val = data[:, col]
    order = np.argsort(x1)
    return x1[order], val[order]


def _time_from(path):
    with open(path) as f:
        return float(re.search(r'time=(\S+)', f.readline()).group(1))


def _bz_lineout(basename):
    """Latest B3 (col 5 in mhd_bcc tab) and its timestamp."""
    files = _tab_files(basename, 'bz')
    path  = files[-1]
    x1, b3 = _read_x1_col(path, 5)
    return x1, b3, _time_from(path)


def _energy_delta(basename):
    """Return (x1, dE) where dE = E(final) - E(initial) for mhd_u output."""
    files = _tab_files(basename, 'en')
    x1, E_init  = _read_x1_col(files[0],  7)
    _,  E_final = _read_x1_col(files[-1], 7)
    return x1, E_final - E_init


def _bz_stencil_pred(x1, coeff, time):
    """Discrete stencil prediction for B3 after one RK1 step."""
    nx  = len(x1)
    dx1 = x1[1] - x1[0]
    rho    = _RHO0 + _DRHO_DX1 * x1
    inv_rho = 1.0 / rho

    # Edge-centred E2: average of adjacent cell-centred values
    e2 = np.zeros(nx + 1)
    e2[0]    = -coeff * _DP_DX2 * inv_rho[0]
    e2[nx]   = -coeff * _DP_DX2 * inv_rho[nx - 1]
    e2[1:nx] = -coeff * _DP_DX2 * 0.5 * (inv_rho[1:nx] + inv_rho[0:nx - 1])

    # CT update: dB3/dt = -dE2/dx1  (E1=0 for this state)
    return -time * (e2[1:nx + 1] - e2[0:nx]) / dx1


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def analyze():
    logger.debug('Analyzing test ' + __name__)
    status = True
    sl = slice(_INTERIOR_PAD, None)  # trim leading boundary cells only;
    # trailing boundary trimmed per-variable below

    # -----------------------------------------------------------------------
    # 1. Stencil accuracy and linear scaling
    # -----------------------------------------------------------------------
    x1_1, b3_1, t1 = _bz_lineout('biermann_grad_c1')
    x1_2, b3_2, t2 = _bz_lineout('biermann_grad_c2')

    if len(x1_1) != len(x1_2) or np.max(np.abs(x1_1 - x1_2)) > 0.0:
        logger.warning('x1 grids do not match between coefficient runs')
        return False
    if abs(t1 - t2) > 0.0:
        logger.warning('times do not match: %e vs %e', t1, t2)
        return False

    nx = len(x1_1)
    interior = slice(_INTERIOR_PAD, nx - _INTERIOR_PAD)

    pred1 = _bz_stencil_pred(x1_1, _COEFF_1, t1)
    pred2 = _bz_stencil_pred(x1_2, _COEFF_2, t2)

    err1  = np.max(np.abs(b3_1[interior] - pred1[interior]))
    err2  = np.max(np.abs(b3_2[interior] - pred2[interior]))
    if err1 > _ERR_TOL:
        logger.warning('stencil error c1 too large: %e > %e', err1, _ERR_TOL)
        status = False
    if err2 > _ERR_TOL:
        logger.warning('stencil error c2 too large: %e > %e', err2, _ERR_TOL)
        status = False

    scale_err = np.max(np.abs(b3_2[interior] - 2.0 * b3_1[interior]))
    if scale_err > _SCALE_TOL:
        logger.warning('scaling error too large: %e > %e', scale_err, _SCALE_TOL)
        status = False

    if np.any(b3_1[interior] >= 0.0):
        logger.warning('expected negative B3 in interior for c1 run')
        status = False

    # -----------------------------------------------------------------------
    # 2. Null test: zero pressure gradient -> zero B3 generation
    # -----------------------------------------------------------------------
    _, b3_null, _ = _bz_lineout('biermann_null')
    null_max = np.max(np.abs(b3_null[interior]))
    if null_max > _NULL_TOL:
        logger.warning('null test: B3 nonzero with zero pressure gradient: %e',
                       null_max)
        status = False

    # -----------------------------------------------------------------------
    # 3. Energy flux: (dE_bz - dE_0) / bz0 should equal ΔB3 stencil pred
    # -----------------------------------------------------------------------
    x1_en, dE_0  = _energy_delta('biermann_en0')
    _,     dE_bz = _energy_delta('biermann_enbz')
    _, b3_enbz, t_en = _bz_lineout('biermann_enbz')

    flux_ratio = (dE_bz[interior] - dE_0[interior]) / _BZ0_TEST
    pred_en    = _bz_stencil_pred(x1_en, _COEFF_EN, t_en)
    scale_pred = np.max(np.abs(pred_en[interior]))

    if scale_pred > 0.0:
        en_rel_err = np.max(np.abs(flux_ratio - pred_en[interior])) / scale_pred
        if en_rel_err > _ENERGY_TOL:
            logger.warning('energy flux relative error too large: %e > %e',
                           en_rel_err, _ENERGY_TOL)
            status = False
    else:
        logger.warning('energy flux test: zero stencil prediction, skipping')

    return status
