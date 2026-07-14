"""Regression test for nonlinear causal Israel--Stewart shear smoothing."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    cases = (
        ('inviscid_128', 0.0, 128),
        ('viscous_128', 0.02, 128),
        ('inviscid_256', 0.0, 256),
        ('viscous_256', 0.02, 256),
    )
    for name, viscosity, resolution in cases:
        athena.run('tests/rsrmhd_viscous_shear_layer.athinput', [
            f'job/basename=rsrmhd_shear_{name}',
            f'problem/viscous_diagnostic_name=rsrmhd_shear_{name}',
            f'mhd/shear_viscosity={viscosity}',
            f'mesh/nx1={resolution}',
            f'meshblock/nx1={resolution // 2}',
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    names = ('inviscid_128', 'viscous_128', 'inviscid_256', 'viscous_256')
    data = {
        name: np.loadtxt(f'build/src/rsrmhd_shear_{name}-errs.dat')
        for name in names
    }

    inviscid = data['inviscid_256']
    viscous = data['viscous_256']
    if not viscous[0] < 0.9 * inviscid[0]:
        logger.warning('Finite viscosity does not sufficiently damp the nonlinear shear: '
                       'inviscid=%g viscous=%g', inviscid[0], viscous[0])
        return False

    for name, values in data.items():
        resolution = int(name.rsplit('_', 1)[1])
        x = (np.arange(resolution) + 0.5) / resolution
        lorentz = np.sqrt(1.0 + (0.5 * np.sin(2.0 * np.pi * x))**2)
        gamma = 5.0 / 3.0
        enthalpy = 1.0 + gamma
        pressure = gamma - 1.0
        initial_energy = np.mean(enthalpy * lorentz**2 - pressure - lorentz)
        if abs(values[6] - initial_energy) > 2.0e-12:
            logger.warning('%s does not conserve total energy: initial=%g final=%g',
                           name, initial_energy, values[6])
            return False
        if values[10] > 1.0e-6 or values[11] > 2.0 + 1.0e-12:
            logger.warning('%s violates a shear constraint: constraint=%g chi=%g',
                           name, values[10], values[11])
            return False
        if np.max(np.abs(values[7:10])) > 1.0e-10:
            logger.warning('%s does not preserve zero total momentum: %s',
                           name, values[7:10])
            return False

    return True
