"""Exact stochastic antenna restart equivalence in CC-E and FC-E layouts."""

import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'vis/python'))
import bin_convert  # noqa: E402

logger = logging.getLogger('athena' + __name__[7:])


def _remove(pattern):
    for filename in glob.glob(pattern):
        os.remove(filename)


def _run_variant(layout, electric_ct):
    reference = f'rsrmhd_antenna_restart_{layout}_reference'
    split = f'rsrmhd_antenna_restart_{layout}_split'
    restarted = f'rsrmhd_antenna_restart_{layout}_restarted'
    for basename in (reference, split, restarted):
        _remove('build/src/' + basename + '*')
        _remove('build/src/bin/' + basename + '*')
        _remove('build/src/rst/' + basename + '.*.rst')

    common = ['mhd/electric_ct=' + electric_ct]
    athena.run('tests/rsrmhd_antenna_source.athinput', [
        'job/basename=' + reference,
        'problem/profile_name=' + reference,
    ] + common)
    athena.run('tests/rsrmhd_antenna_source.athinput', [
        'job/basename=' + split,
        'problem/profile_name=' + split,
        'time/nlim=4',
    ] + common)

    checkpoints = sorted(glob.glob('build/src/rst/' + split + '.*.rst'))
    if not checkpoints:
        raise RuntimeError('No antenna restart checkpoint for ' + layout)
    checkpoint = os.path.relpath(checkpoints[-1], 'build/src')
    athena.restart(checkpoint, [
        'job/basename=' + restarted,
        'problem/profile_name=' + restarted,
        'time/nlim=8',
        'time/tlim=100.0',
    ])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    _run_variant('cell', 'false')
    _run_variant('face', 'true')


def _final_binary(basename, output_id):
    paths = sorted(glob.glob(
        f'build/src/bin/{basename}.{output_id}.*.bin'))
    if not paths:
        raise RuntimeError(f'Missing {output_id} output for {basename}')
    data = bin_convert.read_binary(paths[-1])
    order = sorted(range(data['n_mbs']), key=lambda block: tuple(
        data['mb_logical'][block]))
    values = []
    for name in data['var_names']:
        values.extend(np.asarray(data['mb_data'][name][block]).ravel()
                      for block in order)
    return np.concatenate(values)


def analyze():
    logger.debug('Analyzing test ' + __name__)
    success = True
    for layout in ('cell', 'face'):
        reference = f'rsrmhd_antenna_restart_{layout}_reference'
        restarted = f'rsrmhd_antenna_restart_{layout}_restarted'
        for output_id in ('state', 'electric', 'antenna'):
            state_ref = _final_binary(reference, output_id)
            state_rst = _final_binary(restarted, output_id)
            if not np.allclose(state_ref, state_rst, rtol=0.0, atol=3.0e-13):
                logger.warning('%s %s restart state differs by %.3e', layout,
                               output_id,
                               np.max(np.abs(state_ref - state_rst)))
                success = False

        history_ref = np.atleast_2d(np.loadtxt(
            'build/src/' + reference + '.user.hst'))
        history_rst = np.atleast_2d(np.loadtxt(
            'build/src/' + restarted + '.user.hst'))
        history_ref = history_ref[np.concatenate(
            ([True], np.diff(history_ref[:, 0]) != 0.0))]
        history_rst = history_rst[np.concatenate(
            ([True], np.diff(history_rst[:, 0]) != 0.0))]
        if history_ref.shape[1] != 41 or history_rst.shape[1] != 41:
            logger.warning('%s antenna restart histories have invalid shapes', layout)
            success = False
        elif not np.allclose(history_ref[-1], history_rst[-1],
                             rtol=0.0, atol=3.0e-13):
            logger.warning('%s final restart history differs by %.3e', layout,
                           np.max(np.abs(history_ref[-1] - history_rst[-1])))
            success = False
    return success
