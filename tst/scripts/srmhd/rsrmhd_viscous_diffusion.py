"""Regression test for the Navier--Stokes limit of causal shear relaxation."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_telegraph.athinput', [
        'time/tlim=2.0',
        'mhd/shear_viscosity=0.0025',
        'mhd/shear_relaxation_time=0.01',
        'mesh/nx1=128',
        'meshblock/nx1=64',
    ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    values = np.loadtxt('build/src/rsrmhd_viscous_telegraph-errs.dat')
    amplitude = 1.0e-4
    nu = 0.0025
    time = 2.0
    exact_diffusion = amplitude * np.exp(-nu * (2.0 * np.pi)**2 * time)
    diffusion_error = abs(values[2] - exact_diffusion) / amplitude
    crossing_ratio = values[6] / (1.0 / 128.0)
    if diffusion_error > 1.0e-3 or not 0.99 < crossing_ratio < 1.01:
        logger.warning('Viscous diffusion-limit errors are too large: '
                       'diffusion=%g dtnew/dx=%g',
                       diffusion_error, crossing_ratio)
        return False
    return True
