"""Regression test for homogeneous Israel--Stewart shear relaxation."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_relaxation.athinput', [])
    athena.run('tests/rsrmhd_viscous_relaxation.athinput', [
        'job/basename=rsrmhd_viscous_relaxation_face',
        'mhd/electric_ct=true',
        'mesh/nx1=64',
        'meshblock/nx1=64',
        'problem/viscous_diagnostic_name=rsrmhd_viscous_relaxation_face',
    ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    for name in ('rsrmhd_viscous_relaxation',
                 'rsrmhd_viscous_relaxation_face'):
        stress_error, fluid_error, exact = np.loadtxt(
            'build/src/' + name + '-errs.dat')
        if stress_error > 5.0e-4 or fluid_error > 2.0e-10:
            logger.warning('%s errors are too large: '
                           'stress=%g fluid=%g exact=%g',
                           name, stress_error, fluid_error, exact)
            return False
    return True
