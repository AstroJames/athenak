"""Regression test for multidimensional Israel--Stewart transport and source terms."""

import logging
import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_viscous_multid.athinput', [])
    athena.run('tests/rsrmhd_viscous_multid.athinput', [
        'job/basename=rsrmhd_viscous_multid_x3',
        'mesh/nx1=4',
        'mesh/nx2=4',
        'mesh/nx3=256',
        'meshblock/nx1=4',
        'meshblock/nx2=4',
        'meshblock/nx3=128',
        'problem/wave_direction=3',
    ])

    common = [
        'mesh/nghost=4',
        'mesh/nx1=4',
        'mesh/nx2=4',
        'mesh/nx3=16',
        'meshblock/nx1=4',
        'meshblock/nx2=4',
        'meshblock/nx3=16',
        'time/tlim=0.02',
        'problem/wave_direction=3',
        'problem/amplitude=0.01',
        'problem/background_b1=0.13',
        'problem/background_b2=-0.07',
        'problem/background_b3=0.11',
        'problem/background_e1=-0.03',
        'problem/background_e2=0.05',
        'problem/background_e3=0.02',
    ]
    athena.run('tests/rsrmhd_viscous_multid.athinput', common + [
        'job/basename=rsrmhd_viscous_fofc_x3_dc',
        'mhd/reconstruct=dc',
        'mhd/fofc=false',
        'problem/viscous_diagnostic_name=rsrmhd_viscous_fofc_x3_dc',
    ])
    athena.run('tests/rsrmhd_viscous_multid.athinput', common + [
        'job/basename=rsrmhd_viscous_fofc_x3_forced',
        'mhd/reconstruct=wenoz',
        'mhd/fofc=true',
        'mhd/fofc_force=true',
        'problem/viscous_diagnostic_name=rsrmhd_viscous_fofc_x3_forced',
    ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    threshold = 6.0e-4
    fofc_dc = np.loadtxt('build/src/rsrmhd_viscous_fofc_x3_dc-errs.dat')
    fofc_forced = np.loadtxt(
        'build/src/rsrmhd_viscous_fofc_x3_forced-errs.dat')
    if not np.allclose(fofc_forced, fofc_dc, rtol=0.0, atol=2.0e-14):
        logger.warning('Forced x3 visco-resistive FOFC does not reproduce donor cell: '
                       'dc=%s forced=%s', fofc_dc, fofc_forced)
        return False
    for direction in ('x2', 'x3'):
        filename = f'build/src/rsrmhd_viscous_multid_{direction}-errs.dat'
        velocity_error, stress_error = np.loadtxt(filename)[:2]
        if velocity_error > threshold or stress_error > threshold:
            logger.warning('Multidimensional %s viscous-wave errors exceed threshold: '
                           'velocity=%g stress=%g threshold=%g', direction,
                           velocity_error, stress_error, threshold)
            return False
    return True
