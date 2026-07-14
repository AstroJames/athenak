"""Regression test for a linear longitudinal Israel--Stewart sound mode."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_longitudinal.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    numerical = np.loadtxt('build/src/rsrmhd_viscous_longitudinal-amps.dat')
    gamma = 5.0 / 3.0
    pressure = gamma - 1.0
    enthalpy = 1.0 + gamma
    nu = 0.03
    tau = 0.2
    wave_number = 2.0 * np.pi
    amplitude = 1.0e-4
    matrix = np.array([
        [0.0, wave_number / enthalpy, wave_number / enthalpy],
        [-gamma * pressure * wave_number, 0.0, 0.0],
        [-(4.0 / 3.0) * enthalpy * nu * wave_number / tau, 0.0, -1.0 / tau],
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
    if np.max(normalized) > 2.0e-3:
        logger.warning('Longitudinal viscous-sound errors are too large: %s',
                       normalized)
        return False
    return True
