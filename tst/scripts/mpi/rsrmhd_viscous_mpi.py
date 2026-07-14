"""MPI decomposition and restart redistribution for relativistic viscosity."""

import glob
import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def _remove(pattern):
    for filename in glob.glob(pattern):
        os.remove(filename)


def _arguments(name, direction):
    arguments = [
        'job/basename=' + name,
        'problem/viscous_diagnostic_name=' + name,
        'mesh/nx1=4',
        'meshblock/nx1=4',
        'time/tlim=0.1',
    ]
    if direction == 'x2':
        arguments += [
            'mesh/nx2=64',
            'meshblock/nx2=32',
        ]
    else:
        arguments += [
            'mesh/nx2=4',
            'mesh/nx3=64',
            'meshblock/nx2=4',
            'meshblock/nx3=32',
            'problem/wave_direction=3',
        ]
    return arguments


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    input_file = 'tests/rsrmhd_viscous_multid.athinput'
    names = (
        'rsrmhd_viscous_mpi_x2_one',
        'rsrmhd_viscous_mpi_x2_two',
        'rsrmhd_viscous_mpi_x2_split',
        'rsrmhd_viscous_mpi_x2_restarted',
        'rsrmhd_viscous_mpi_x3_one',
        'rsrmhd_viscous_mpi_x3_two',
    )
    for name in names:
        _remove('build/src/' + name + '-errs.dat')
        _remove('build/src/rst/' + name + '.*.rst')

    for direction in ('x2', 'x3'):
        one = 'rsrmhd_viscous_mpi_' + direction + '_one'
        two = 'rsrmhd_viscous_mpi_' + direction + '_two'
        athena.mpirun(1, input_file, _arguments(one, direction))
        athena.mpirun(2, input_file, _arguments(two, direction))

    split = 'rsrmhd_viscous_mpi_x2_split'
    split_arguments = _arguments(split, 'x2') + [
        'time/nlim=8',
        'output1/dcycle=8',
    ]
    athena.mpirun(1, input_file, split_arguments)
    checkpoints = sorted(glob.glob('build/src/rst/' + split + '.*.rst'))
    if not checkpoints:
        raise RuntimeError('No viscous MPI restart checkpoint was written')
    checkpoint = os.path.relpath(checkpoints[-1], 'build/src')
    restarted = 'rsrmhd_viscous_mpi_x2_restarted'
    athena.mpirestart(2, checkpoint, [
        'job/basename=' + restarted,
        'problem/viscous_diagnostic_name=' + restarted,
        'time/nlim=-1',
        'time/tlim=0.1',
        'output1/dcycle=0',
    ])


def _load(name):
    filename = 'build/src/' + name + '-errs.dat'
    return np.atleast_1d(np.loadtxt(filename))


def analyze():
    logger.debug('Analyzing test ' + __name__)
    success = True
    for direction in ('x2', 'x3'):
        one = _load('rsrmhd_viscous_mpi_' + direction + '_one')
        two = _load('rsrmhd_viscous_mpi_' + direction + '_two')
        if one.shape != (14,) or two.shape != (14,) or \
                not np.all(np.isfinite(one)) or not np.all(np.isfinite(two)):
            logger.warning('%s MPI diagnostics are invalid: %s %s',
                           direction, one, two)
            success = False
            continue
        if not np.allclose(one, two, rtol=0.0, atol=5.0e-13):
            logger.warning('%s one-rank/two-rank solutions differ: %s',
                           direction, two - one)
            success = False
        if one[0] > 1.0e-2 or one[1] > 1.0e-2 or one[11] > 2.0e-11:
            logger.warning('%s MPI wave or constraints failed: %s',
                           direction, one[[0, 1, 11]])
            success = False
        if one[12] != 0.1 or one[13] != 16:
            logger.warning('%s MPI time/cycle diagnostics are wrong: %s',
                           direction, one[12:14])
            success = False

    reference = _load('rsrmhd_viscous_mpi_x2_two')
    restarted = _load('rsrmhd_viscous_mpi_x2_restarted')
    if restarted.shape != (14,) or not np.all(np.isfinite(restarted)):
        logger.warning('Viscous MPI restart diagnostics are invalid: %s', restarted)
        success = False
    elif not np.allclose(reference, restarted, rtol=0.0, atol=2.0e-11):
        logger.warning('Continuous/repartitioned viscous solutions differ: %s',
                       restarted - reference)
        success = False
    return success
