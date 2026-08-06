"""Exact stochastic antenna restart equivalence in CC-E and FC-E layouts."""

import glob
import logging
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'vis/python'))
import bin_convert  # noqa: E402

logger = logging.getLogger('athena' + __name__[7:])

ANTENNA_MAGIC = 0x414E54454E4E4133


def _convert_zhdankin_v3_to_v2(source, target):
    """Convert a four-mode v3 checkpoint record to the historical v2 layout."""
    data = Path(source).read_bytes()
    data, replacements = re.subn(
        rb'(state_version\s*=\s*)3', rb'\g<1>2', data, count=1)
    if replacements != 1:
        raise RuntimeError('Checkpoint does not contain one version-3 marker')

    header_format = '=7Q'
    header_bytes = struct.calcsize(header_format)
    record_offset = data.find(struct.pack('=Q', ANTENNA_MAGIC))
    if record_offset < 0:
        raise RuntimeError('Checkpoint has no antenna v3 record')
    header = struct.unpack_from(header_format, data, record_offset)
    _, schema, payload_bytes, rng_bytes, diagnostics, modes, _ = header
    if schema != 2 or diagnostics != 18 or modes != 4:
        raise RuntimeError(f'Unexpected antenna v3 header: {header}')
    weighted_values = diagnostics + 4*modes
    if (payload_bytes - rng_bytes) % weighted_values != 0:
        raise RuntimeError('Antenna v3 payload has an invalid Real size')
    real_bytes = (payload_bytes - rng_bytes)//weighted_values

    payload_offset = record_offset + header_bytes
    mode_offset = payload_offset + rng_bytes + diagnostics*real_bytes
    record_end = payload_offset + payload_bytes
    legacy_record = (
        data[payload_offset:payload_offset + rng_bytes + 16*real_bytes]
        + data[mode_offset:mode_offset + 16*real_bytes]
    )
    Path(target).write_bytes(
        data[:record_offset] + legacy_record + data[record_end:])


def _remove(pattern):
    for filename in glob.glob(pattern):
        os.remove(filename)


def _run_variant(layout, electric_ct, overrides=None):
    reference = f'rsrmhd_antenna_restart_{layout}_reference'
    split = f'rsrmhd_antenna_restart_{layout}_split'
    restarted = f'rsrmhd_antenna_restart_{layout}_restarted'
    legacy = f'rsrmhd_antenna_restart_{layout}_v2'
    for basename in (reference, split, restarted, legacy):
        _remove('build/src/' + basename + '*')
        _remove('build/src/bin/' + basename + '*')
        _remove('build/src/rst/' + basename + '.*.rst')

    common = ['mhd/electric_ct=' + electric_ct]
    if overrides:
        common.extend(overrides)
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
    if layout == 'cell':
        legacy_checkpoint = 'build/src/rst/' + legacy + '.00000.rst'
        _convert_zhdankin_v3_to_v2(checkpoints[-1], legacy_checkpoint)
        athena.restart(os.path.relpath(legacy_checkpoint, 'build/src'), [
            'job/basename=' + legacy,
            'problem/profile_name=' + legacy,
            'time/nlim=8',
            'time/tlim=100.0',
        ])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    _run_variant('cell', 'false')
    _run_variant('face', 'true')
    _run_variant('band', 'false', [
        'antenna_driving/mode_set=anisotropic_band',
        'antenna_driving/nperp_min=1',
        'antenna_driving/nperp_max=2',
        'antenna_driving/nparallel_min=1',
        'antenna_driving/nparallel_max=2',
        'antenna_driving/current_envelope=band',
        'antenna_driving/frequency_model=alfven_parallel',
    ])


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
    for layout in ('cell', 'face', 'band'):
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
        if history_ref.shape[1] != 43 or history_rst.shape[1] != 43:
            logger.warning('%s antenna restart histories have invalid shapes', layout)
            success = False
        elif not np.allclose(history_ref[-1], history_rst[-1],
                             rtol=0.0, atol=3.0e-13):
            logger.warning('%s final restart history differs by %.3e', layout,
                           np.max(np.abs(history_ref[-1] - history_rst[-1])))
            success = False
    reference = 'rsrmhd_antenna_restart_cell_reference'
    legacy = 'rsrmhd_antenna_restart_cell_v2'
    for output_id in ('state', 'electric', 'antenna'):
        state_ref = _final_binary(reference, output_id)
        state_v2 = _final_binary(legacy, output_id)
        if not np.allclose(state_ref, state_v2, rtol=0.0, atol=3.0e-13):
            logger.warning('v2 %s restart differs by %.3e', output_id,
                           np.max(np.abs(state_ref - state_v2)))
            success = False
    history_ref = np.atleast_2d(np.loadtxt(
        'build/src/' + reference + '.user.hst'))
    history_v2 = np.atleast_2d(np.loadtxt(
        'build/src/' + legacy + '.user.hst'))
    if not np.allclose(history_ref[-1], history_v2[-1],
                       rtol=0.0, atol=3.0e-13):
        logger.warning('v2 final restart history differs by %.3e',
                       np.max(np.abs(history_ref[-1] - history_v2[-1])))
        success = False
    return success
