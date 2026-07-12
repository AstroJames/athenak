"""Two-dimensional resistive-SRMHD charged-vortex regression."""

import logging
import glob
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    initial_file = 'build/src/rsrmhd_charged_vortex_initial-errs.dat'
    evolved_file = 'build/src/rsrmhd_charged_vortex_evolved-errs.dat'
    for filename in (initial_file, evolved_file):
        if os.path.exists(filename):
            os.remove(filename)

    for resolution in (32, 64, 128):
        athena.run('tests/rsrmhd_charged_vortex.athinput', [
            'job/basename=rsrmhd_charged_vortex_initial',
            'mesh/nx1=' + repr(resolution),
            'mesh/nx2=' + repr(resolution),
            'meshblock/nx1=' + repr(resolution),
            'meshblock/nx2=' + repr(resolution),
            'time/nlim=0',
            'output1/dt=-1.0',
        ])

    for resolution in (32, 64, 128):
        output_dt = '5.0' if resolution == 128 else '-1.0'
        athena.run('tests/rsrmhd_charged_vortex.athinput', [
            'job/basename=rsrmhd_charged_vortex_evolved',
            'mesh/nx1=' + repr(resolution),
            'mesh/nx2=' + repr(resolution),
            'meshblock/nx1=' + repr(resolution),
            'meshblock/nx2=' + repr(resolution),
            'time/nlim=-1',
            'time/tlim=5.0',
            'output1/dt=' + output_dt,
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    initial = np.loadtxt(
        'build/src/rsrmhd_charged_vortex_initial-errs.dat', ndmin=2)
    evolved = np.loadtxt(
        'build/src/rsrmhd_charged_vortex_evolved-errs.dat', ndmin=2)
    if initial.shape != (3, 11) or evolved.shape != (3, 11):
        logger.warning('Charged-vortex diagnostics have unexpected shapes')
        return False
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(evolved)):
        logger.warning('Charged-vortex diagnostics contain non-finite values')
        return False
    if not np.array_equal(initial[:, 0], (32, 64, 128)):
        logger.warning('Charged-vortex resolutions are wrong: %s', initial[:, 0])
        return False

    q_rates = np.log2(initial[:-1, 2] / initial[1:, 2])
    if np.any(q_rates < 1.8):
        logger.warning('Discrete charge is not second-order: %s', q_rates)
        return False
    if np.any(initial[:, 6] > 2.0e-13):
        logger.warning('Compatible discrete Gauss balance failed: %s', initial[:, 6])
        return False
    if np.any(initial[:, 7:10] != 0.0):
        logger.warning('Initial divB, symmetry, or recovery diagnostic failed: %s',
                       initial[:, 7:10])
        return False

    if not np.array_equal(evolved[:, 0], (32, 64, 128)):
        logger.warning('Evolved charged-vortex resolutions are wrong: %s', evolved[:, 0])
        return False
    if not np.array_equal(evolved[:, 1], (16, 32, 64)) or \
            np.any(np.abs(evolved[:, 10] - 5.0) > 1.0e-13):
        logger.warning('The charged vortex has wrong cycle counts or final times')
        return False

    q_evolved_rates = np.log2(evolved[:-1, 2] / evolved[1:, 2])
    p_rates = np.log2(evolved[:-1, 4] / evolved[1:, 4])
    if np.any(q_evolved_rates < (1.1, 1.5)):
        logger.warning('Evolved charge convergence regressed: %s', q_evolved_rates)
        return False
    if np.any(p_rates < (1.3, 1.8)):
        logger.warning('Pressure-equilibrium convergence regressed: %s', p_rates)
        return False
    if np.any(evolved[:, 6] > 2.0e-13) or np.any(evolved[:, 7:10] != 0.0):
        logger.warning('The evolved vortex broke Gauss balance, divB, parity, or '
                       'recovery: %s', evolved[:, 6:10])
        return False

    charge_outputs = sorted(glob.glob(
        'build/src/tab/rsrmhd_charged_vortex_evolved.charge.*.tab'))
    if not charge_outputs:
        logger.warning('Named mhd_q output was not written')
        return False
    with open(charge_outputs[-1], encoding='utf-8') as output_file:
        output_file.readline()
        labels = output_file.readline().split()
    if 'q' not in labels:
        logger.warning('Named mhd_q output is missing its q label: %s', labels)
        return False
    return True
