"""Exact full-stack restart equivalence in serial."""

import glob
import logging
import os

import numpy as np

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def _remove(pattern):
    for filename in glob.glob(pattern):
        os.remove(filename)


def _run_variant(name, electric_ct, components):
    reference = name + '_reference'
    split = name + '_split'
    restarted = name + '_restarted'
    for basename in (reference, split, restarted):
        _remove('build/src/' + basename + '*')
        _remove('build/src/rst/' + basename + '.*.rst')

    common = [
        'mhd/electric_ct=' + electric_ct,
        'turb_driving/num_components=' + str(components),
    ]
    athena.run('tests/rsrmhd_coupled_restart.athinput', [
        'job/basename=' + reference,
        'problem/profile_name=' + reference,
        'problem/restart_state_name=' + reference + '_state',
    ] + common)
    athena.run('tests/rsrmhd_coupled_restart.athinput', [
        'job/basename=' + split,
        'problem/profile_name=' + split,
        'problem/restart_state_name=' + split + '_state',
        'time/nlim=2',
    ] + common)

    checkpoints = sorted(glob.glob('build/src/rst/' + split + '.*.rst'))
    if not checkpoints:
        raise RuntimeError('No coupled restart checkpoint was written for ' + name)
    checkpoint = os.path.relpath(checkpoints[-1], 'build/src')
    athena.restart(checkpoint, [
        'job/basename=' + restarted,
        'problem/profile_name=' + restarted,
        'problem/restart_state_name=' + restarted + '_state',
        'time/nlim=4',
        'time/tlim=10.0',
    ])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    _run_variant('rsrmhd_coupled_restart_single_cell', 'false', 1)
    _run_variant('rsrmhd_coupled_restart_multi_face', 'true', 2)


def _load_state(name):
    files = sorted(glob.glob('build/src/' + name + '_state*.dat'))
    if not files:
        raise RuntimeError('Missing restart state for ' + name)
    state = np.vstack([np.atleast_2d(np.loadtxt(path)) for path in files])
    order = np.lexsort((state[:, 2], state[:, 1], state[:, 0]))
    return state[order]


def analyze():
    logger.debug('Analyzing test ' + __name__)
    success = True
    variants = (
        ('rsrmhd_coupled_restart_single_cell', 29),
        ('rsrmhd_coupled_restart_multi_face', 32),
    )
    for name, columns in variants:
        reference = name + '_reference'
        restarted = name + '_restarted'
        state_ref = _load_state(reference)
        state_rst = _load_state(restarted)
        if state_ref.shape != (512, columns) or state_rst.shape != state_ref.shape:
            logger.warning('%s restart states have invalid shapes: %s %s',
                           name, state_ref.shape, state_rst.shape)
            success = False
            continue
        if not np.all(np.isfinite(state_ref)) or not np.all(np.isfinite(state_rst)):
            logger.warning('%s restart state is non-finite', name)
            success = False
            continue
        if not np.allclose(state_ref, state_rst, rtol=0.0, atol=3.0e-13):
            logger.warning('%s continuous/restarted state differs by %g', name,
                           np.max(np.abs(state_ref - state_rst)))
            success = False

        history_ref = np.atleast_2d(np.loadtxt(
            'build/src/' + reference + '.user.hst'))
        history_rst = np.atleast_2d(np.loadtxt(
            'build/src/' + restarted + '.user.hst'))
        if history_ref.shape[1] != 48 or history_rst.shape[1] != 48:
            logger.warning('%s histories have invalid shapes: %s %s',
                           name, history_ref.shape, history_rst.shape)
            success = False
            continue
        if not np.allclose(history_ref[-1], history_rst[-1],
                           rtol=0.0, atol=3.0e-12):
            logger.warning('%s final histories differ by %g', name,
                           np.max(np.abs(history_ref[-1] - history_rst[-1])))
            success = False
        if history_ref[-1, 21] <= 0.0 or history_ref[-1, 42] <= 0.0:
            logger.warning('%s forcing or cooling cumulative audit is inactive', name)
            success = False
    return success
