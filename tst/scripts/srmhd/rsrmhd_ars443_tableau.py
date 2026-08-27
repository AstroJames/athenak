"""Independent regressions for the globally stiffly accurate ARS(4,4,3) pair."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_EXPLICIT_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0/2.0, 0.0, 0.0, 0.0, 0.0],
    [11.0/18.0, 1.0/18.0, 0.0, 0.0, 0.0],
    [5.0/6.0, -5.0/6.0, 1.0/2.0, 0.0, 0.0],
    [1.0/4.0, 7.0/4.0, 3.0/4.0, -7.0/4.0, 0.0],
])
_IMPLICIT_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0/2.0, 0.0, 0.0, 0.0],
    [0.0, 1.0/6.0, 1.0/2.0, 0.0, 0.0],
    [0.0, -1.0/2.0, 1.0/2.0, 1.0/2.0, 0.0],
    [0.0, 3.0/2.0, -3.0/2.0, 1.0/2.0, 1.0/2.0],
])
_EXPLICIT_B = _EXPLICIT_A[-1].copy()
_IMPLICIT_B = _IMPLICIT_A[-1].copy()
_GAM0 = np.array([0.0, 11.0/9.0, 15.0/17.0, 3.0/2.0])
_GAM1 = np.array([1.0, -2.0/9.0, -4.0, 27.0/2.0])
_GAM2 = np.array([0.0, 0.0, 5.0/2.0, -17.0/2.0])
_BETA = np.array([1.0/2.0, 1.0/18.0, 1.0/2.0, -7.0/4.0])
_DELTA = np.array([0.0, 8.0, -108.0/17.0, 0.0])
_A_TWID = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [-4.0/9.0, 0.0, 0.0, 0.0],
    [-8.0, 8.0, 0.0, 0.0],
    [109.0/4.0, -117.0/4.0, -1.0/4.0, 0.0],
])
_H_VALUES = (1.0, 10.0, 100.0, 1000.0)
_INITIAL_STRESS = np.array((0.004, -0.001, -0.003, 0.002, -0.0015, 0.01))
_INITIAL_ELECTRIC = np.array((0.02, -0.015, 0.01))
_CONVERGENCE_RESOLUTIONS = (8, 16, 32)


def _stability_function(h):
    stages = np.linalg.solve(
        np.eye(_IMPLICIT_A.shape[0]) + h*_IMPLICIT_A,
        np.ones(_IMPLICIT_A.shape[0]))
    return 1.0 - h*np.dot(_IMPLICIT_B, stages)


def _forced_equilibrium_response(h):
    stages = np.linalg.solve(
        np.eye(_IMPLICIT_A.shape[0]) + h*_IMPLICIT_A,
        np.ones(_IMPLICIT_A.shape[0]) + h*_EXPLICIT_A.dot(
            np.ones(_EXPLICIT_A.shape[0])))
    return 1.0 + h*np.sum(_EXPLICIT_B) - h*np.dot(_IMPLICIT_B, stages)


def _audit_low_storage_coefficients():
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    accum = np.zeros_like(state)
    initial = state.copy()
    for stage in range(4):
        accum += _DELTA[stage]*state
        rhs = np.zeros_like(state)
        rhs[stage + 1] = 1.0
        state = (_GAM0[stage]*state + _GAM1[stage]*initial
                 + _GAM2[stage]*accum + _BETA[stage]*rhs)
        target = np.concatenate(([1.0], _EXPLICIT_A[stage + 1, :4]))
        if not np.allclose(state, target, rtol=0.0, atol=5.0e-15):
            return False

    state = np.zeros(4)
    accum = np.zeros(4)
    for stage in range(4):
        accum += _DELTA[stage]*state
        inherited = _GAM0[stage]*state + _GAM2[stage]*accum
        correction = np.zeros(4)
        correction[:stage] = (_IMPLICIT_A[stage + 1, 1:stage + 1]
                              - inherited[:stage])
        if not np.allclose(correction, _A_TWID[stage], rtol=0.0,
                           atol=5.0e-15):
            return False
        state = _IMPLICIT_A[stage + 1, 1:].copy()
    return True


def _run_relaxation(basename, resolution, eta, tlim):
    athena.run('tests/rsrmhd_viscous_relaxation.athinput', [
        'job/basename=' + basename,
        'time/integrator=imex3_ars443',
        'time/cfl_number=0.5',
        'time/tlim=' + repr(tlim),
        'mhd/electric_ct=false',
        'mhd/resistivity=' + repr(eta),
        'mhd/shear_relaxation_time=' + repr(eta),
        'mesh/nx1=' + repr(resolution),
        'meshblock/nx1=' + repr(resolution),
        'problem/background_e1=' + repr(float(_INITIAL_ELECTRIC[0])),
        'problem/background_e2=' + repr(float(_INITIAL_ELECTRIC[1])),
        'problem/background_e3=' + repr(float(_INITIAL_ELECTRIC[2])),
        'problem/viscous_diagnostic_name=' + basename,
    ])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for h in _H_VALUES:
        basename = 'rsrmhd_ars443_h{}'.format(int(h))
        filename = 'build/src/' + basename + '-errs.dat'
        if os.path.exists(filename):
            os.remove(filename)
        _run_relaxation(basename, 4, 0.125/h, 0.125)

    for resolution in _CONVERGENCE_RESOLUTIONS:
        basename = 'rsrmhd_ars443_order_{}'.format(resolution)
        filename = 'build/src/' + basename + '-errs.dat'
        if os.path.exists(filename):
            os.remove(filename)
        _run_relaxation(basename, resolution, 0.25, 0.125)


def analyze():
    logger.debug('Analyzing test ' + __name__)
    if not _audit_low_storage_coefficients():
        logger.warning('ARS(4,4,3) low-storage factorization is inconsistent')
        return False
    if not np.allclose(_EXPLICIT_A.sum(axis=1), _IMPLICIT_A.sum(axis=1),
                       rtol=0.0, atol=5.0e-15):
        logger.warning('ARS explicit and implicit abscissae differ')
        return False
    if not np.array_equal(_EXPLICIT_B, _EXPLICIT_A[-1]) or not np.array_equal(
            _IMPLICIT_B, _IMPLICIT_A[-1]):
        logger.warning('ARS pair is not globally stiffly accurate')
        return False
    forced = np.array([_forced_equilibrium_response(h) for h in _H_VALUES])
    if not np.allclose(forced, 1.0, rtol=0.0, atol=2.0e-12):
        logger.warning('ARS forced equilibria drift: %s', forced)
        return False
    if abs(_stability_function(1.0e8)) >= 1.0e-7:
        logger.warning('ARS implicit response is not L-stable')
        return False

    for h in _H_VALUES:
        basename = 'rsrmhd_ars443_h{}'.format(int(h))
        row = np.loadtxt('build/src/' + basename + '-errs.dat')
        expected = _stability_function(h)
        stress_ratio = row[4:10]/_INITIAL_STRESS
        electric_ratio = row[16:19]/_INITIAL_ELECTRIC
        if (row.shape != (19,) or not np.all(np.isfinite(row))
                or not np.allclose(stress_ratio, expected, rtol=2.0e-6,
                                   atol=5.0e-10)
                or not np.allclose(electric_ratio, expected, rtol=2.0e-6,
                                   atol=5.0e-9)):
            logger.warning('%s does not match the ARS stability function: %s',
                           basename, row)
            return False

    exact = np.exp(-0.5)
    stress_errors = []
    electric_errors = []
    for resolution in _CONVERGENCE_RESOLUTIONS:
        basename = 'rsrmhd_ars443_order_{}'.format(resolution)
        row = np.loadtxt('build/src/' + basename + '-errs.dat')
        stress_errors.append(np.max(np.abs(row[4:10]/_INITIAL_STRESS - exact)))
        electric_errors.append(np.max(np.abs(row[16:19]/_INITIAL_ELECTRIC - exact)))
    stress_rates = np.log2(np.asarray(stress_errors[:-1])/stress_errors[1:])
    electric_rates = np.log2(np.asarray(electric_errors[:-1])/electric_errors[1:])
    if np.any(stress_rates < 2.8) or np.any(electric_rates < 2.8):
        logger.warning('ARS third-order relaxation regressed: shear=%s E=%s',
                       stress_rates, electric_rates)
        return False
    return True
