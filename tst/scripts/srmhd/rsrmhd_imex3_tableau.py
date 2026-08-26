"""Independent regressions for the Pareschi--Russo IMEX-SSP3(4,3,3) tableau."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_ALPHA = 0.24169426078821
_BETA = 0.06042356519705
_ETA = 0.12915286960590
_IMPLICIT_A = np.array([
    [_ALPHA, 0.0, 0.0, 0.0],
    [-_ALPHA, _ALPHA, 0.0, 0.0],
    [0.0, 1.0 - _ALPHA, _ALPHA, 0.0],
    [_BETA, _ETA, 0.5 - _BETA - _ETA - _ALPHA, _ALPHA],
])
_WEIGHTS = np.array([0.0, 1.0/6.0, 1.0/6.0, 2.0/3.0])
_EXPLICIT_C = np.array([0.0, 0.0, 1.0, 0.5])
_H_VALUES = (1.0, 10.0, 100.0, 1000.0)
_LAYOUTS = (('cell', False), ('face', True))
_INITIAL_STRESS = np.array((0.004, -0.001, -0.003, 0.002, -0.0015, 0.01))
_INITIAL_ELECTRIC = np.array((0.02, -0.015, 0.01))
_CONVERGENCE_RESOLUTIONS = (8, 16, 32)
_MOVING_TLIM = 0.125
_MOVING_VELOCITY = 2.0e-3
_MOVING_B2 = 5.0e-5
_MOVING_MOMENTUM_RATE = (1.0 + 4.0/3.0)*_MOVING_VELOCITY/_MOVING_TLIM


def _stability_function(h):
    """Return R(h) directly from the published implicit Butcher tableau."""
    stages = np.linalg.solve(
        np.eye(_IMPLICIT_A.shape[0]) + h*_IMPLICIT_A,
        np.ones(_IMPLICIT_A.shape[0]))
    return 1.0 - h*np.dot(_WEIGHTS, stages)


def _moving_equilibrium_response(h):
    """Return E(n+1)/Eeq(n+1) for Eeq proportional to the explicit stage time."""
    stages = np.linalg.solve(
        np.eye(_IMPLICIT_A.shape[0]) + h*_IMPLICIT_A, _EXPLICIT_C)
    return h*np.dot(_WEIGHTS, stages)


def _run_relaxation(basename, electric_ct, resolution, eta, tlim):
    athena.run('tests/rsrmhd_viscous_relaxation.athinput', [
        'job/basename=' + basename,
        'time/integrator=imex3',
        'time/cfl_number=0.5',
        'time/tlim=' + repr(tlim),
        'mhd/electric_ct=' + str(electric_ct).lower(),
        'mhd/resistivity=' + repr(float(eta)),
        'mhd/shear_relaxation_time=' + repr(eta),
        'mesh/nx1=' + repr(resolution),
        'meshblock/nx1=' + repr(resolution),
        'problem/background_e1=' + repr(float(_INITIAL_ELECTRIC[0])),
        'problem/background_e2=' + repr(float(_INITIAL_ELECTRIC[1])),
        'problem/background_e3=' + repr(float(_INITIAL_ELECTRIC[2])),
        'problem/viscous_diagnostic_name=' + basename,
    ])


def _run_moving_equilibrium(basename, electric_ct, h):
    eta = _MOVING_TLIM/h
    athena.run('tests/rsrmhd_moving_equilibrium.athinput', [
        'job/basename=' + basename,
        'mhd/electric_ct=' + str(electric_ct).lower(),
        'mhd/resistivity=' + repr(float(eta)),
        'problem/moving_momentum_rate=' + repr(_MOVING_MOMENTUM_RATE),
        'problem/background_b2=' + repr(_MOVING_B2),
        'problem/moving_diagnostic_name=' + basename,
    ])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for layout, electric_ct in _LAYOUTS:
        for h in _H_VALUES:
            basename = 'rsrmhd_imex3_{}_h{}'.format(layout, int(h))
            filename = 'build/src/' + basename + '-errs.dat'
            if os.path.exists(filename):
                os.remove(filename)
            _run_relaxation(basename, electric_ct, 4, 0.125/h, 0.125)

        for resolution in _CONVERGENCE_RESOLUTIONS:
            basename = 'rsrmhd_imex3_order_{}_{}'.format(layout, resolution)
            filename = 'build/src/' + basename + '-errs.dat'
            if os.path.exists(filename):
                os.remove(filename)
            _run_relaxation(basename, electric_ct, resolution, 0.25, 0.125)

        for h in _H_VALUES:
            basename = 'rsrmhd_imex3_moving_{}_h{}'.format(layout, int(h))
            filename = 'build/src/' + basename + '-errs.dat'
            if os.path.exists(filename):
                os.remove(filename)
            _run_moving_equilibrium(basename, electric_ct, h)

    explicit_args = [
        'time/tlim=0.1',
        'time/nlim=1000',
        'mesh/nghost=2',
        'mesh/nx1=16',
        'mesh/nx2=1',
        'mesh/nx3=1',
        'meshblock/nx1=16',
        'meshblock/nx2=1',
        'meshblock/nx3=1',
        'hydro/reconstruct=plm',
        'hydro/rsolver=llf',
        'problem/along_x1=true',
        'problem/wave_flag=3',
        'problem/vflow=1.0',
        'problem/amp=1.0e-6',
        'output1/dt=-1.0',
        'output2/dt=-1.0',
        'output3/dt=-1.0',
    ]
    for integrator in ('rk3', 'imex3'):
        basename = 'rsrmhd_imex3_explicit_' + integrator
        filename = 'build/src/' + basename + '-errs.dat'
        if os.path.exists(filename):
            os.remove(filename)
        athena.run('tests/linear_wave_hydro.athinput', explicit_args + [
            'job/basename=' + basename,
            'time/integrator=' + integrator,
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    reference_values = np.array([
        0.3673276720995303,
        0.08612452203047904,
        0.053661101328145056,
        0.006509787846270765,
    ])
    tableau_values = np.array([_stability_function(h) for h in _H_VALUES])
    if not np.allclose(tableau_values, reference_values, rtol=0.0, atol=5.0e-15):
        logger.warning('Published-tableau stability values changed: %s',
                       tableau_values)
        return False

    moving_reference_values = np.array([
        0.3729574942374578,
        1.070854597606288,
        1.2632611360264192,
        1.2822911301410311,
    ])
    moving_tableau_values = np.array(
        [_moving_equilibrium_response(h) for h in _H_VALUES])
    moving_exact_values = np.array(
        [1.0 - (1.0-np.exp(-h))/h for h in _H_VALUES])
    if not np.allclose(moving_tableau_values, moving_reference_values,
                       rtol=0.0, atol=5.0e-15):
        logger.warning('Published moving-equilibrium values changed: %s',
                       moving_tableau_values)
        return False
    moving_stiff_limit = np.dot(
        _WEIGHTS, np.linalg.solve(_IMPLICIT_A, _EXPLICIT_C))
    if abs(moving_stiff_limit - 1.2843646522044607) > 5.0e-15:
        logger.warning('Published moving-equilibrium stiff limit changed: %s',
                       moving_stiff_limit)
        return False
    if moving_tableau_values[-1] - moving_exact_values[-1] < 0.25:
        logger.warning('Stiff moving endpoint unexpectedly follows equilibrium: %s %s',
                       moving_tableau_values[-1], moving_exact_values[-1])
        return False

    one_step_results = {}
    for layout, _ in _LAYOUTS:
        one_step_results[layout] = {}
        for h, expected in zip(_H_VALUES, tableau_values):
            basename = 'rsrmhd_imex3_{}_h{}'.format(layout, int(h))
            row = np.loadtxt('build/src/' + basename + '-errs.dat')
            if row.shape != (19,) or not np.all(np.isfinite(row)):
                logger.warning('%s diagnostics are invalid: %s', basename, row)
                return False
            if row[0] != 4 or row[1] != 1 or row[2] != 0.125:
                logger.warning('%s metadata are wrong: %s', basename, row[:3])
                return False
            stress_ratio = row[4:10]/_INITIAL_STRESS
            electric_ratio = row[16:19]/_INITIAL_ELECTRIC
            if not np.allclose(stress_ratio, expected, rtol=1.0e-6,
                               atol=3.0e-10):
                logger.warning('%s shear relaxation is not tableau-exact: %s != %s',
                               basename, stress_ratio, expected)
                return False
            # The face-centred path uses a coupled Picard/C2P solve; allow its
            # accumulated nonlinear-solve error while still separating the old
            # O(1) stability-function defect by more than four orders of magnitude.
            if not np.allclose(electric_ratio, expected, rtol=1.0e-6,
                               atol=5.0e-6):
                logger.warning('%s electric relaxation is not tableau-exact: %s != %s',
                               basename, electric_ratio, expected)
                return False
            one_step_results[layout][h] = np.concatenate(
                (stress_ratio, electric_ratio))

    if abs(tableau_values[-1]) >= 0.01:
        logger.warning('The h=1000 response is not approaching zero: %s',
                       tableau_values[-1])
        return False
    asymptotic_response = _stability_function(1.0e8)
    if abs(asymptotic_response) >= 1.0e-6:
        logger.warning('The published-tableau response is not L-stable: %s',
                       asymptotic_response)
        return False
    for h in _H_VALUES:
        if not np.allclose(one_step_results['cell'][h],
                           one_step_results['face'][h],
                           rtol=0.0, atol=5.0e-6):
            logger.warning('Cell/face stiff responses differ at h=%s', h)
            return False

    moving_results = {}
    lorentz_factor = np.sqrt(1.0 + _MOVING_VELOCITY**2)
    final_u1 = _MOVING_VELOCITY
    final_b2 = _MOVING_B2
    final_equilibrium = -_MOVING_VELOCITY*final_b2
    for layout, _ in _LAYOUTS:
        moving_results[layout] = {}
        for h, expected in zip(_H_VALUES, moving_tableau_values):
            basename = 'rsrmhd_imex3_moving_{}_h{}'.format(layout, int(h))
            row = np.loadtxt('build/src/' + basename + '-errs.dat')
            if row.shape != (9,) or not np.all(np.isfinite(row)):
                logger.warning('%s diagnostics are invalid: %s', basename, row)
                return False
            if row[0] != 4 or row[1] != 1 or row[2] != _MOVING_TLIM:
                logger.warning('%s metadata are wrong: %s', basename, row[:3])
                return False
            if abs(row[3] - final_u1) > 2.0e-7 or row[8] > 2.0e-7:
                logger.warning('%s fluid did not remain in the linear limit: %s',
                               basename, row[[3, 8]])
                return False
            if abs(row[4] - final_b2) > 2.0e-15:
                logger.warning('%s explicit moving target is wrong: %s != %s',
                               basename, row[4], final_b2)
                return False
            response = row[5]/final_equilibrium
            if not np.isclose(response, expected, rtol=1.0e-5, atol=1.0e-7):
                logger.warning('%s moving response is not tableau-exact: %s != %s',
                               basename, response, expected)
                return False
            residual_response = row[6]/(-final_equilibrium)
            if not np.isclose(residual_response, 1.0-expected,
                              rtol=1.0e-5, atol=1.0e-7):
                logger.warning('%s accepted-state residual is wrong: %s != %s',
                               basename, residual_response, 1.0-expected)
                return False
            eta = _MOVING_TLIM/h
            normalized_qohm = row[7]*eta/(lorentz_factor*final_equilibrium)**2
            # qohm squares a small difference of face-averaged quantities.  Near
            # the h~10 zero crossing, FC-E Picard/C2P noise is amplified relative
            # to the residual while the signed endpoint response remains accurate.
            if not np.isclose(normalized_qohm, (1.0-expected)**2,
                              rtol=3.0e-3, atol=1.0e-4):
                logger.warning('%s endpoint qohm is inconsistent: %s != %s',
                               basename, normalized_qohm, (1.0-expected)**2)
                return False
            moving_results[layout][h] = response
    if moving_tableau_values[-1] < 1.25:
        logger.warning('The stiff moving-equilibrium endpoint layer disappeared: %s',
                       moving_tableau_values[-1])
        return False
    for h in _H_VALUES:
        if not np.isclose(moving_results['cell'][h], moving_results['face'][h],
                          rtol=0.0, atol=3.0e-8):
            logger.warning('Cell/face moving responses differ at h=%s', h)
            return False

    exact = np.exp(-0.5)
    for layout, _ in _LAYOUTS:
        stress_errors = []
        electric_errors = []
        for resolution in _CONVERGENCE_RESOLUTIONS:
            basename = 'rsrmhd_imex3_order_{}_{}'.format(layout, resolution)
            row = np.loadtxt('build/src/' + basename + '-errs.dat')
            if row.shape != (19,) or not np.all(np.isfinite(row)):
                logger.warning('%s diagnostics are invalid: %s', basename, row)
                return False
            stress_errors.append(np.max(np.abs(row[4:10]/_INITIAL_STRESS - exact)))
            electric_errors.append(
                np.max(np.abs(row[16:19]/_INITIAL_ELECTRIC - exact)))
        stress_rates = np.log2(np.asarray(stress_errors[:-1])
                               / np.asarray(stress_errors[1:]))
        electric_rates = np.log2(np.asarray(electric_errors[:-1])
                                 / np.asarray(electric_errors[1:]))
        if np.any(stress_rates < 2.8) or np.any(electric_rates < 2.8):
            logger.warning('%s third-order relaxation regressed: shear=%s E=%s',
                           layout, stress_rates, electric_rates)
            return False

    rk3 = np.loadtxt('build/src/rsrmhd_imex3_explicit_rk3-errs.dat')
    imex3 = np.loadtxt('build/src/rsrmhd_imex3_explicit_imex3-errs.dat')
    if not np.array_equal(rk3, imex3):
        logger.warning('The zero-stiff-source IMEX3 path differs from SSPRK3: %s %s',
                       rk3, imex3)
        return False
    return True
