"""Regression test for the two-dimensional relativistic viscous KH problem."""

import logging

import numpy as np

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for name, viscosity in (('inviscid', 0.0), ('viscous', 0.02)):
        athena.run('tests/rsrmhd_viscous_kh.athinput', [
            f'job/basename=rsrmhd_kh_{name}',
            f'mhd/shear_viscosity={viscosity}',
            'mesh/nx1=32',
            'mesh/nx2=32',
            'meshblock/nx1=16',
            'meshblock/nx2=16',
            'time/tlim=0.5',
            'output1/dt=0.05',
            f'problem/viscous_diagnostic_name=rsrmhd_kh_{name}',
        ])

    common = [
        'mhd/shear_viscosity=0.02',
        'mesh/nx1=16',
        'mesh/nx2=16',
        'meshblock/nx1=16',
        'meshblock/nx2=16',
        'time/tlim=0.02',
        'problem/background_b1=0.13',
        'problem/background_b2=-0.07',
        'problem/background_b3=0.11',
        'problem/background_e1=-0.03',
        'problem/background_e2=0.05',
        'problem/background_e3=0.02',
    ]
    athena.run('tests/rsrmhd_viscous_kh.athinput', common + [
        'job/basename=rsrmhd_kh_fofc_dc',
        'mhd/reconstruct=dc',
        'mhd/fofc=false',
        'problem/viscous_diagnostic_name=rsrmhd_kh_fofc_dc',
    ])
    athena.run('tests/rsrmhd_viscous_kh.athinput', common + [
        'job/basename=rsrmhd_kh_fofc_forced',
        'mhd/reconstruct=wenoz',
        'mhd/fofc=true',
        'mhd/fofc_force=true',
        'problem/viscous_diagnostic_name=rsrmhd_kh_fofc_forced',
    ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    inviscid = np.loadtxt('build/src/rsrmhd_kh_inviscid-errs.dat')
    viscous = np.loadtxt('build/src/rsrmhd_kh_viscous-errs.dat')
    fofc_dc = np.loadtxt('build/src/rsrmhd_kh_fofc_dc-errs.dat')
    fofc_forced = np.loadtxt('build/src/rsrmhd_kh_fofc_forced-errs.dat')

    if not np.allclose(fofc_forced, fofc_dc, rtol=0.0, atol=2.0e-14):
        logger.warning('Forced visco-resistive FOFC does not reproduce donor cell: '
                       'dc=%s forced=%s', fofc_dc, fofc_forced)
        return False

    if not viscous[1] < 0.75 * inviscid[1]:
        logger.warning('Viscosity does not sufficiently reduce KH enstrophy: '
                       'inviscid=%g viscous=%g', inviscid[1], viscous[1])
        return False
    if abs(viscous[2] - inviscid[2]) > 2.0e-11:
        logger.warning('The KH cases do not conserve the same total energy: %g %g',
                       inviscid[2], viscous[2])
        return False
    for name, values in (('inviscid', inviscid), ('viscous', viscous)):
        if np.max(np.abs(values[3:6])) > 2.0e-6:
            logger.warning('%s KH run does not preserve global momentum: %s',
                           name, values[3:6])
            return False
        if values[6] > 1.0e-4 or values[7] > 2.0 + 1.0e-12:
            logger.warning('%s KH run violates a stress constraint: %g %g',
                           name, values[6], values[7])
            return False
    return True
