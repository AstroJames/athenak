"""MPI restart coverage for the full cooled/driven/visco-resistive system."""

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
    reference = 'rsrmhd_coupled_restart_mpi_reference'
    split = 'rsrmhd_coupled_restart_mpi_split'
    restarted = 'rsrmhd_coupled_restart_mpi_restarted'
    rank_source = 'rsrmhd_coupled_restart_rank_source'
    rank_loaded = 'rsrmhd_coupled_restart_rank_loaded'
    for basename in (reference, split, restarted, rank_source, rank_loaded):
        _remove('build/src/' + basename + '*')
        _remove('build/src/rst/' + basename + '.*.rst')

    common = [
        'mhd/electric_ct=true',
        'turb_driving/num_components=2',
    ]
    athena.mpirun(2, 'tests/rsrmhd_coupled_restart.athinput', [
        'job/basename=' + reference,
        'problem/profile_name=' + reference,
        'problem/restart_state_name=' + reference + '_state',
    ] + common)
    athena.mpirun(2, 'tests/rsrmhd_coupled_restart.athinput', [
        'job/basename=' + split,
        'problem/profile_name=' + split,
        'problem/restart_state_name=' + split + '_state',
        'time/nlim=2',
    ] + common)

    checkpoints = sorted(glob.glob('build/src/rst/' + split + '.*.rst'))
    if not checkpoints:
        raise RuntimeError('No rank-changing coupled checkpoint was written')
    checkpoint = os.path.relpath(checkpoints[-1], 'build/src')
    athena.mpirestart(2, checkpoint, [
        'job/basename=' + restarted,
        'problem/profile_name=' + restarted,
        'problem/restart_state_name=' + restarted + '_state',
        'time/nlim=4',
        'time/tlim=10.0',
    ])

    # Separately verify that a checkpoint written on one rank is redistributed
    # without changing any active cell state when loaded on two ranks.  Evolving
    # the two decompositions is intentionally not compared: global floating-point
    # reductions need not be bitwise decomposition invariant.
    athena.mpirun(1, 'tests/rsrmhd_coupled_restart.athinput', [
        'job/basename=' + rank_source,
        'problem/profile_name=' + rank_source,
        'problem/restart_state_name=' + rank_source + '_state',
        'time/nlim=2',
    ] + common)
    rank_checkpoints = sorted(glob.glob(
        'build/src/rst/' + rank_source + '.*.rst'))
    if not rank_checkpoints:
        raise RuntimeError('No rank-changing coupled checkpoint was written')
    rank_checkpoint = os.path.relpath(rank_checkpoints[-1], 'build/src')
    athena.mpirestart(2, rank_checkpoint, [
        'job/basename=' + rank_loaded,
        'problem/profile_name=' + rank_loaded,
        'problem/restart_state_name=' + rank_loaded + '_state',
        'time/nlim=2',
        'time/tlim=10.0',
    ])


def _load_state(name):
    files = sorted(glob.glob('build/src/' + name + '_state*.dat'))
    state = np.vstack([np.atleast_2d(np.loadtxt(path)) for path in files])
    order = np.lexsort((state[:, 2], state[:, 1], state[:, 0]))
    return state[order]


def analyze():
    logger.debug('Analyzing test ' + __name__)
    reference = 'rsrmhd_coupled_restart_mpi_reference'
    restarted = 'rsrmhd_coupled_restart_mpi_restarted'
    state_ref = _load_state(reference)
    state_rst = _load_state(restarted)
    if state_ref.shape != (512, 32) or state_rst.shape != state_ref.shape:
        logger.warning('MPI restart states have invalid shapes: %s %s',
                       state_ref.shape, state_rst.shape)
        return False
    if not np.allclose(state_ref, state_rst, rtol=0.0, atol=8.0e-12):
        logger.warning('MPI restart state differs by %g',
                       np.max(np.abs(state_ref - state_rst)))
        return False

    rank_source = _load_state('rsrmhd_coupled_restart_rank_source')
    rank_loaded = _load_state('rsrmhd_coupled_restart_rank_loaded')
    if rank_source.shape != (512, 32) or rank_loaded.shape != rank_source.shape:
        logger.warning('Rank-load states have invalid shapes: %s %s',
                       rank_source.shape, rank_loaded.shape)
        return False
    if not np.allclose(rank_source, rank_loaded, rtol=0.0, atol=3.0e-13):
        logger.warning('Rank-load state differs by %g',
                       np.max(np.abs(rank_source - rank_loaded)))
        return False

    history_ref = np.atleast_2d(np.loadtxt(
        'build/src/' + reference + '.user.hst'))
    history_rst = np.atleast_2d(np.loadtxt(
        'build/src/' + restarted + '.user.hst'))
    if history_ref.shape[1] != 48 or history_rst.shape[1] != 48:
        logger.warning('MPI histories have invalid shapes: %s %s',
                       history_ref.shape, history_rst.shape)
        return False
    if not np.allclose(history_ref[-1], history_rst[-1],
                       rtol=0.0, atol=1.0e-11):
        logger.warning('MPI final histories differ by %g',
                       np.max(np.abs(history_ref[-1] - history_rst[-1])))
        return False
    return True
