"""Regression test for separated and coupled decaying-turbulence transport."""

import logging

import numpy as np

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    common = [
        'mesh/nx1=32',
        'mesh/nx2=32',
        'meshblock/nx1=16',
        'meshblock/nx2=16',
        'time/tlim=0.1',
        'time/ndiag=20',
        'output1/dt=0.05',
    ]
    variants = (
        ('viscous_hydro', [
            'problem/magnetic_rms=0.0',
            'mhd/relativistic_viscosity=true',
        ]),
        ('resistive', [
            'problem/magnetic_rms=0.15',
            'mhd/relativistic_viscosity=false',
        ]),
        ('viscoresistive', [
            'problem/magnetic_rms=0.15',
            'mhd/relativistic_viscosity=true',
        ]),
    )
    for name, arguments in variants:
        athena.run('tests/rsrmhd_decaying_turbulence.athinput', common + arguments + [
            f'job/basename=rsrmhd_turb_{name}',
            f'problem/profile_name=rsrmhd_turb_{name}',
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    names = ('viscous_hydro', 'resistive', 'viscoresistive')
    histories = {
        name: np.atleast_2d(np.loadtxt(
            f'build/src/rsrmhd_turb_{name}.user.hst'))
        for name in names
    }
    for name, history in histories.items():
        if history.shape != (3, 12) or not np.all(np.isfinite(history)):
            logger.warning('%s turbulence history is invalid: shape=%s',
                           name, history.shape)
            return False

    hydro = histories['viscous_hydro']
    resistive = histories['resistive']
    combined = histories['viscoresistive']
    if not np.array_equal(hydro[0, [3, 6]], resistive[0, [3, 6]]) or \
            not np.array_equal(hydro[0, [3, 6]], combined[0, [3, 6]]):
        logger.warning('The three cases do not share the same initial velocity field')
        return False

    hydro_em = hydro[:, [4, 5, 7, 11]]
    if np.max(np.abs(hydro_em)) != 0.0:
        logger.warning('The viscous-hydro run generated electromagnetic fields: %s',
                       hydro_em)
        return False
    if np.max(np.abs(resistive[:, 8])) != 0.0:
        logger.warning('The resistive-only run generated viscous shear stress')
        return False
    if np.max(hydro[:, 8]) <= 0.0 or np.max(combined[:, 8]) <= 0.0:
        logger.warning('A viscous run did not develop causal shear stress')
        return False
    if hydro[-1, 6] >= 0.95*resistive[-1, 6] or \
            combined[-1, 6] >= 0.95*resistive[-1, 6]:
        logger.warning('Viscosity did not sufficiently damp enstrophy: %g %g %g',
                       hydro[-1, 6], resistive[-1, 6], combined[-1, 6])
        return False
    for name, history in (('resistive', resistive), ('combined', combined)):
        if history[-1, 4] >= history[0, 4]:
            logger.warning('%s magnetic energy did not decay', name)
            return False
        if np.max(history[:, 11]) > 1.0e-20:
            logger.warning('%s violates the face-centered divergence constraint: %g',
                           name, np.max(history[:, 11]))
            return False
    return True
