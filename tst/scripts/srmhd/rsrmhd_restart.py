"""Restart equivalence for both resistive-SRMHD electric-field layouts."""

import glob
import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def _remove(pattern):
    for filename in glob.glob(pattern):
        os.remove(filename)


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    variants = (
        ('cell', 'false', []),
        ('face', 'true', []),
        ('nonuniform_cell', 'false', [
            'mhd/resistivity_model=charge_starvation',
            'mhd/eta_floor=1.0e-8',
            'mhd/eta_scale=1.0e-2',
            'mhd/number_per_mass=1.0',
            'problem/nonideal_e_scale=1.05',
        ]),
        ('nonuniform_face', 'true', [
            'mhd/resistivity_model=charge_starvation',
            'mhd/eta_floor=1.0e-8',
            'mhd/eta_scale=1.0e-2',
            'mhd/number_per_mass=1.0',
            'problem/nonideal_e_scale=1.05',
        ]),
    )
    for name, electric_ct, model_arguments in variants:
        reference = 'rsrmhd_restart_' + name + '_reference'
        split = 'rsrmhd_restart_' + name + '_split'
        restarted = 'rsrmhd_restart_' + name + '_restarted'
        for basename in (reference, split, restarted):
            _remove('build/src/' + basename + '-errs.dat')
            _remove('build/src/rst/' + basename + '.*.rst')

        common = ['mhd/electric_ct=' + electric_ct] + model_arguments
        athena.run('tests/rsrmhd_restart.athinput',
                   ['job/basename=' + reference] + common)
        athena.run('tests/rsrmhd_restart.athinput',
                   ['job/basename=' + split, 'time/nlim=1'] + common)

        checkpoints = sorted(glob.glob('build/src/rst/' + split + '.*.rst'))
        if not checkpoints:
            raise RuntimeError('No restart checkpoint was written for ' + name)
        checkpoint = os.path.relpath(checkpoints[-1], 'build/src')
        athena.restart(checkpoint, [
            'job/basename=' + restarted,
            'time/nlim=-1',
            'time/tlim=1.0',
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    success = True
    for name in ('cell', 'face', 'nonuniform_cell', 'nonuniform_face'):
        reference = np.atleast_1d(np.loadtxt(
            'build/src/rsrmhd_restart_' + name + '_reference-errs.dat'))
        restarted = np.atleast_1d(np.loadtxt(
            'build/src/rsrmhd_restart_' + name + '_restarted-errs.dat'))
        if reference.shape != (11,) or restarted.shape != (11,) or \
                not np.all(np.isfinite(reference)) or \
                not np.all(np.isfinite(restarted)):
            logger.warning('%s restart diagnostics are invalid: %s %s',
                           name, reference, restarted)
            success = False
            continue
        if not np.allclose(reference, restarted, rtol=0.0, atol=2.0e-13):
            logger.warning('%s continuous/restarted solutions differ: %s',
                           name, restarted - reference)
            success = False
        if reference[1] != 2 or reference[9] != 0 or reference[10] != 1.0:
            logger.warning('%s reference run has invalid cycle/recovery/time: %s',
                           name, reference[[1, 9, 10]])
            success = False
    return success
