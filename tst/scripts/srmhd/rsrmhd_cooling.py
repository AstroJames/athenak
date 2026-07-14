"""Regression test for covariant entropy-relaxation cooling."""

import logging

import numpy as np

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_entropy_cooling.athinput', [])
    athena.run('tests/rsrmhd_entropy_cooling.athinput', [
        'job/basename=rsrmhd_entropy_cooling_cold',
        'problem/pressure=0.5',
        'problem/profile_name=rsrmhd_entropy_cooling_cold',
    ])
    athena.run('tests/rsrmhd_entropy_cooling.athinput', [
        'job/basename=rsrmhd_entropy_cooling_boosted',
        'problem/uniform_velocity1=0.4',
        'problem/profile_name=rsrmhd_entropy_cooling_boosted',
    ])
    athena.run('tests/rsrmhd_driven_turbulence.athinput', [
        'job/basename=rsrmhd_drive_cooling',
        'mesh/nx1=16',
        'mesh/nx2=16',
        'meshblock/nx1=16',
        'meshblock/nx2=16',
        'time/tlim=0.2',
        'time/ndiag=10',
        'mhd/relativistic_cooling=entropy',
        'problem/profile_name=rsrmhd_drive_cooling',
        'output2/dt=1.0',
        'output3/dt=1.0',
    ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    warm = np.atleast_2d(np.loadtxt(
        'build/src/rsrmhd_entropy_cooling.user.hst'))
    cold = np.atleast_2d(np.loadtxt(
        'build/src/rsrmhd_entropy_cooling_cold.user.hst'))
    boosted = np.atleast_2d(np.loadtxt(
        'build/src/rsrmhd_entropy_cooling_boosted.user.hst'))
    for name, history in (('warm', warm), ('cold', cold),
                          ('boosted', boosted)):
        if history.shape[0] < 2 or history.shape[1] != 29 \
                or not np.all(np.isfinite(history)):
            logger.warning('%s cooling history is invalid: shape=%s',
                           name, history.shape)
            return False

    gamma = 5.0/3.0
    target_adiabat = 1.0
    cooling_time = 0.1
    initial_pressure = 2.0
    final_time = warm[-1, 0]
    exact_pressure = target_adiabat*np.exp(
        np.log(initial_pressure/target_adiabat)
        * np.exp(-final_time/cooling_time))
    pressure_error = abs(warm[-1, 10] - exact_pressure)
    if pressure_error > 5.0e-5:
        logger.warning('Entropy cooling missed the analytic solution: %g',
                       pressure_error)
        return False

    if np.max(np.abs(warm[:, 12:15])) > 2.0e-13 \
            or np.max(np.abs(warm[:, 20:23])) > 2.0e-13 \
            or np.max(np.abs(warm[:, 24:27])) > 2.0e-13:
        logger.warning('A comoving cooling state generated momentum')
        return False
    if np.max(np.abs(warm[:, 27:29])) > 2.0e-13:
        logger.warning('The smooth cooling test unexpectedly used its limiter')
        return False
    if np.max(np.abs(warm[:, 9] - 1.0)) > 2.0e-13:
        logger.warning('Cooling changed the rest density')
        return False

    if np.max(np.abs(cold[:, 10] - 0.5)) > 2.0e-13 \
            or np.max(np.abs(cold[:, 19:29])) > 2.0e-13:
        logger.warning('One-sided cooling acted below the target adiabat')
        return False

    for name, history in (('warm', warm), ('boosted', boosted)):
        energy_residual = (
            history[:, 15] - history[0, 15] + history[:, 23]
        )
        momentum_residual = (
            history[:, 12:15] - history[0, 12:15] + history[:, 24:27]
        )
        if np.max(np.abs(energy_residual)) > 2.0e-10:
            logger.warning('%s cooling energy audit failed: %s',
                           name, energy_residual)
            return False
        if np.max(np.abs(momentum_residual)) > 2.0e-10:
            logger.warning('%s cooling momentum audit failed: %s',
                           name, momentum_residual)
            return False
        if np.max(np.abs(history[:, 16]/history[0, 16] - 1.0)) > 2.0e-12:
            logger.warning('%s cooling changed the conserved mass', name)
            return False

    driven = np.atleast_2d(np.loadtxt(
        'build/src/rsrmhd_drive_cooling.user.hst'))
    if driven.shape != (5, 48) or not np.all(np.isfinite(driven)):
        logger.warning('Driven cooling history is invalid: shape=%s', driven.shape)
        return False
    driven_energy_residual = (
        driven[:, 20] - driven[0, 20] - driven[:, 21] + driven[:, 42]
    )
    driven_momentum_residual = (
        driven[:, 17:20] - driven[0, 17:20] - driven[:, 22:25]
        + driven[:, 43:46]
    )
    if np.max(np.abs(driven_energy_residual)) > 2.0e-10 \
            or np.max(np.abs(driven_momentum_residual)) > 2.0e-10:
        logger.warning('Driven cooling source audit failed: %s %s',
                       driven_energy_residual, driven_momentum_residual)
        return False
    if driven[-1, 42] <= 0.0 or np.max(np.abs(driven[:, 46:48])) > 2.0e-13:
        logger.warning('Driven cooling was inactive or unexpectedly limited')
        return False
    return True
