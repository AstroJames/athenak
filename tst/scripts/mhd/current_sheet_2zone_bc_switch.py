# Regression test for current_sheet_2zone BC switching.
#
# Verifies that:
# 1) Before t_switch: x1 uses diode behavior and x2 remains periodic.
# 2) After t_switch: ghost zones are overwritten by the IC-matching profile.

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

_INPUT = 'tests/current_sheet_2zone_bc_switch.athinput'
_BASENAME = 'cs2_bc_switch'

_NG = 3
_NX1 = 32
_NX2 = 24
_T_SWITCH = 0.03
_X1_SLICE = 0.03125

_GAMMA = 5.0 / 3.0
_B0 = 1.0
_A0 = 0.5
_BG = 0.0
_P0 = 0.6

_TOL_STRICT = 1.0e-12
_TOL_BCC2_X2 = 5.0e-4


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run(_INPUT, [])


def _file_index(path):
    match = re.search(r'\.(\d{5})\.tab$', path)
    if match is None:
        raise RuntimeError('Could not parse file index from {}'.format(path))
    return int(match.group(1))


def _load_series(tag):
    paths = sorted(glob.glob('build/src/tab/{}.{}.*.tab'.format(_BASENAME, tag)))
    if len(paths) == 0:
        raise RuntimeError('No output files found for tag {}'.format(tag))
    series = []
    for path in paths:
        data = athena_read.tab(path)
        series.append((data['time'], data, path))
    return series


def _load_tag_at_index(tag, index):
    path = 'build/src/tab/{}.{}.{:05d}.tab'.format(_BASENAME, tag, index)
    return athena_read.tab(path)


def _sorted_view(data, index_key):
    idx = data[index_key].astype(int)
    order = np.argsort(idx)
    sorted_data = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) == len(idx):
            sorted_data[key] = val[order]
        else:
            sorted_data[key] = val
    sorted_data[index_key] = idx[order]
    return sorted_data


def _analytic_profile(x1):
    p_eq = 0.5 * _B0 * _B0 / np.cosh(x1 / _A0) ** 2 + _P0
    by_eq = _B0 * np.tanh(x1 / _A0)
    dens_eq = p_eq ** (1.0 / _GAMMA)
    ener_eq = p_eq / (_GAMMA - 1.0) + 0.5 * by_eq * by_eq
    return dens_eq, by_eq, ener_eq


def _allclose_or_warn(name, arr, ref, tol):
    err = np.max(np.abs(arr - ref))
    if err > tol:
        logger.warning('%s max abs error %e exceeds tolerance %e', name, err, tol)
        return False
    return True


def _check_pre_x1_diode(pre_x1u):
    ok = True
    data = _sorted_view(pre_x1u, 'i')
    i = data['i']
    expected_i = np.arange(_NX1 + 2 * _NG)
    if not np.array_equal(i, expected_i):
        logger.warning('Unexpected i-index layout in pre x1 line output')
        return False

    inner = slice(0, _NG)
    outer = slice(_NG + _NX1, _NG + _NX1 + _NG)
    left = _NG
    right = _NG + _NX1 - 1

    for key in ('dens', 'mom2', 'mom3', 'ener'):
        ok &= _allclose_or_warn('pre x1 diode copy inner {}'.format(key),
                                data[key][inner], data[key][left], _TOL_STRICT)
        ok &= _allclose_or_warn('pre x1 diode copy outer {}'.format(key),
                                data[key][outer], data[key][right], _TOL_STRICT)

    left_m1 = data['mom1'][left]
    right_m1 = data['mom1'][right]
    inner_expected = np.minimum(0.0, left_m1)
    outer_expected = np.maximum(0.0, right_m1)
    ok &= _allclose_or_warn('pre x1 diode mom1 inner',
                            data['mom1'][inner], inner_expected, _TOL_STRICT)
    ok &= _allclose_or_warn('pre x1 diode mom1 outer',
                            data['mom1'][outer], outer_expected, _TOL_STRICT)
    return ok


def _check_pre_x2_periodic(pre_x2u, pre_x2b):
    ok = True
    data_u = _sorted_view(pre_x2u, 'j')
    data_b = _sorted_view(pre_x2b, 'j')
    j = data_u['j']
    expected_j = np.arange(_NX2 + 2 * _NG)
    if not np.array_equal(j, expected_j):
        logger.warning('Unexpected j-index layout in pre x2 line output')
        return False

    for key in ('dens', 'mom1', 'mom2', 'mom3', 'ener'):
        errs = []
        arr = data_u[key]
        for g in range(_NG):
            errs.append(abs(arr[g] - arr[g + _NX2]))
            errs.append(abs(arr[_NG + _NX2 + g] - arr[_NG + g]))
        if max(errs) > _TOL_STRICT:
            logger.warning('pre x2 periodic wrap failed for %s (max err=%e)',
                           key, max(errs))
            ok = False

    for key in ('bcc1', 'bcc2', 'bcc3'):
        errs = []
        arr = data_b[key]
        for g in range(_NG):
            errs.append(abs(arr[g] - arr[g + _NX2]))
            errs.append(abs(arr[_NG + _NX2 + g] - arr[_NG + g]))
        if max(errs) > _TOL_STRICT:
            logger.warning('pre x2 periodic wrap failed for %s (max err=%e)',
                           key, max(errs))
            ok = False

    return ok


def _check_post_x1_ic(post_x1u, post_x1b):
    ok = True
    data_u = _sorted_view(post_x1u, 'i')
    data_b = _sorted_view(post_x1b, 'i')
    i = data_u['i']

    ghost_mask = np.zeros_like(i, dtype=bool)
    ghost_mask[:_NG] = True
    ghost_mask[_NG + _NX1:_NG + _NX1 + _NG] = True
    x1 = data_u['x1v'][ghost_mask]
    dens_eq, by_eq, ener_eq = _analytic_profile(x1)

    ok &= _allclose_or_warn('post x1 ic dens', data_u['dens'][ghost_mask], dens_eq,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x1 ic mom1', data_u['mom1'][ghost_mask], 0.0,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x1 ic mom2', data_u['mom2'][ghost_mask], 0.0,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x1 ic mom3', data_u['mom3'][ghost_mask], 0.0,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x1 ic ener', data_u['ener'][ghost_mask], ener_eq,
                            _TOL_STRICT)

    ok &= _allclose_or_warn('post x1 ic bcc1', data_b['bcc1'][ghost_mask], 0.0,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x1 ic bcc2', data_b['bcc2'][ghost_mask], by_eq,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x1 ic bcc3', data_b['bcc3'][ghost_mask], _BG,
                            _TOL_STRICT)
    return ok


def _check_post_x2_ic(post_x2u, post_x2b):
    ok = True
    data_u = _sorted_view(post_x2u, 'j')
    data_b = _sorted_view(post_x2b, 'j')
    j = data_u['j']

    ghost_mask = np.zeros_like(j, dtype=bool)
    ghost_mask[:_NG] = True
    ghost_mask[_NG + _NX2:_NG + _NX2 + _NG] = True

    dens_eq, by_eq, ener_eq = _analytic_profile(_X1_SLICE)
    ok &= _allclose_or_warn('post x2 ic dens', data_u['dens'][ghost_mask], dens_eq,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x2 ic mom1', data_u['mom1'][ghost_mask], 0.0,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x2 ic mom2', data_u['mom2'][ghost_mask], 0.0,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x2 ic mom3', data_u['mom3'][ghost_mask], 0.0,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x2 ic ener', data_u['ener'][ghost_mask], ener_eq,
                            _TOL_STRICT)

    ok &= _allclose_or_warn('post x2 ic bcc1', data_b['bcc1'][ghost_mask], 0.0,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x2 ic bcc3', data_b['bcc3'][ghost_mask], _BG,
                            _TOL_STRICT)
    ok &= _allclose_or_warn('post x2 ic bcc2', data_b['bcc2'][ghost_mask], by_eq,
                            _TOL_BCC2_X2)
    return ok


def analyze():
    logger.debug('Analyzing test ' + __name__)
    status = True

    x1u_series = _load_series('x1u')
    pre_candidates = [entry for entry in x1u_series if entry[0] < _T_SWITCH]
    if len(pre_candidates) == 0:
        logger.warning('No pre-switch snapshot found (t < t_switch)')
        return False
    pre_time, pre_x1u, pre_path = max(pre_candidates, key=lambda x: x[0])
    post_time, post_x1u, post_path = max(x1u_series, key=lambda x: x[0])

    if post_time <= _T_SWITCH:
        logger.warning('No post-switch snapshot found (latest time=%e, t_switch=%e)',
                       post_time, _T_SWITCH)
        return False

    pre_idx = _file_index(pre_path)
    post_idx = _file_index(post_path)
    pre_x2u = _load_tag_at_index('x2u', pre_idx)
    pre_x2b = _load_tag_at_index('x2b', pre_idx)
    post_x1b = _load_tag_at_index('x1b', post_idx)
    post_x2u = _load_tag_at_index('x2u', post_idx)
    post_x2b = _load_tag_at_index('x2b', post_idx)

    logger.info('Using pre-switch snapshot at t=%e (index=%05d)', pre_time, pre_idx)
    logger.info('Using post-switch snapshot at t=%e (index=%05d)', post_time, post_idx)

    status &= _check_pre_x1_diode(pre_x1u)
    status &= _check_pre_x2_periodic(pre_x2u, pre_x2b)
    status &= _check_post_x1_ic(post_x1u, post_x1b)
    status &= _check_post_x2_ic(post_x2u, post_x2b)

    return status
