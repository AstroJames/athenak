"""Regression test for the full nonlinear 1D Israel--Stewart source operator."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_telegraph.athinput', [
        'mhd/linearized_shear_target_1d=false',
    ])
    athena.run('tests/rsrmhd_viscous_longitudinal.athinput', [
        'mhd/linearized_shear_target_1d=false',
    ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    values = np.loadtxt('build/src/rsrmhd_viscous_telegraph-errs.dat')
    velocity_error, stress_error = values[:2]
    threshold = 5.0e-4
    if velocity_error > threshold or stress_error > threshold:
        logger.warning('Nonlinear viscous-wave errors exceed threshold: '
                       'velocity=%g stress=%g threshold=%g',
                       velocity_error, stress_error, threshold)
        return False

    numerical = np.loadtxt('build/src/rsrmhd_viscous_longitudinal-amps.dat')
    gamma = 5.0 / 3.0
    pressure = gamma - 1.0
    enthalpy = 1.0 + gamma
    wave_number = 2.0 * np.pi
    amplitude = 1.0e-4
    matrix = np.array([
        [0.0, wave_number / enthalpy, wave_number / enthalpy],
        [-gamma * pressure * wave_number, 0.0, 0.0],
        [-(4.0 / 3.0) * enthalpy * 0.03 * wave_number / 0.2, 0.0, -1.0 / 0.2],
    ])
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    coefficients = np.linalg.solve(eigenvectors, np.array([amplitude, 0.0, 0.0]))
    exact = eigenvectors @ (np.exp(eigenvalues * numerical[3]) * coefficients)
    exact = np.real_if_close(exact).real
    errors = np.abs(numerical[:3] - exact)
    normalized = np.array([
        errors[0] / amplitude,
        errors[1] / (enthalpy * amplitude),
        errors[2] / (enthalpy * amplitude),
    ])
    if np.max(normalized) > 1.0e-3:
        logger.warning('Nonlinear longitudinal viscous-wave errors exceed threshold: %s',
                       normalized)
        return False
    return True
