"""Regression test for a transverse viscous mode on a boosted background."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_boosted.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    numerical = np.loadtxt('build/src/rsrmhd_viscous_boosted-amps.dat')
    velocity = numerical[0] - 1j * numerical[1]
    stress = numerical[2] - 1j * numerical[3]
    mean_u1, boost, time = numerical[4:7]

    gamma_gas = 5.0 / 3.0
    enthalpy = 1.0 + gamma_gas
    nu = 0.05
    tau = 0.2
    wave_number = 2.0 * np.pi
    amplitude = 1.0e-4
    boost_lorentz = 1.0 / np.sqrt(1.0 - boost**2)
    dynamic_viscosity = enthalpy * nu
    derivative_matrix = np.array([
        [enthalpy * boost_lorentz, boost],
        [dynamic_viscosity * boost_lorentz * boost, tau],
    ], dtype=complex)
    source_matrix = np.array([
        [-1j * wave_number * enthalpy * boost_lorentz * boost,
         -1j * wave_number],
        [-1j * wave_number * dynamic_viscosity * boost_lorentz,
         -1.0 / boost_lorentz - 1j * wave_number * tau * boost],
    ], dtype=complex)
    matrix = np.linalg.solve(derivative_matrix, source_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    coefficients = np.linalg.solve(eigenvectors,
                                   np.array([-1j * amplitude, 0.0]))
    exact = eigenvectors @ (np.exp(eigenvalues * time) * coefficients)

    velocity_error = abs(velocity - exact[0]) / amplitude
    stress_error = abs(stress - exact[1]) / (enthalpy * amplitude)
    background_error = abs(mean_u1 - boost_lorentz * boost)
    threshold = 5.0e-4
    if velocity_error > threshold or stress_error > threshold:
        logger.warning('Boosted viscous-wave errors exceed threshold: '
                       'velocity=%g stress=%g threshold=%g',
                       velocity_error, stress_error, threshold)
        return False
    if background_error > 2.0e-8:
        logger.warning('Boosted background drifted: error=%g', background_error)
        return False
    return True
