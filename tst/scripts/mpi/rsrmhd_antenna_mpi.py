"""MPI current exchange and rank-changing restart for the SRRMHD antenna."""

import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'vis/python'))
import athena_read  # noqa: E402
import bin_convert  # noqa: E402

logger = logging.getLogger('athena' + __name__[7:])


def _remove(pattern):
    for filename in glob.glob(pattern):
        os.remove(filename)


def _clean(basename):
    _remove('build/src/' + basename + '*')
    _remove('build/src/bin/' + basename + '*')
    _remove('build/src/rst/' + basename + '.*.rst')


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    field_one = 'rsrmhd_antenna_mpi_field_one'
    field_two = 'rsrmhd_antenna_mpi_field_two'
    field_four = 'rsrmhd_antenna_mpi_field_four'
    reference = 'rsrmhd_antenna_mpi_reference'
    split = 'rsrmhd_antenna_mpi_split'
    restarted = 'rsrmhd_antenna_mpi_restarted'
    reference_four = 'rsrmhd_antenna_mpi_reference_four'
    split_four = 'rsrmhd_antenna_mpi_split_four'
    restarted_four = 'rsrmhd_antenna_mpi_restarted_four'
    coupled = 'rsrmhd_antenna_mpi_coupled_reference'
    coupled_split = 'rsrmhd_antenna_mpi_coupled_split'
    coupled_restarted = 'rsrmhd_antenna_mpi_coupled_restarted'
    basenames = (
        field_one, field_two, field_four, reference, split, restarted,
        reference_four, split_four, restarted_four, coupled, coupled_split,
        coupled_restarted,
    )
    for basename in basenames:
        _clean(basename)

    field_arguments = [
        'mhd/electric_ct=true',
        'antenna_driving/mode_set=anisotropic_band',
        'antenna_driving/nperp_min=1',
        'antenna_driving/nperp_max=2',
        'antenna_driving/nparallel_min=1',
        'antenna_driving/nparallel_max=2',
        'antenna_driving/current_envelope=band',
        'antenna_driving/frequency_model=alfven_parallel',
    ]
    athena.mpirun(1, 'tests/rsrmhd_antenna_field.athinput', [
        'job/basename=' + field_one,
    ] + field_arguments)
    athena.mpirun(2, 'tests/rsrmhd_antenna_field.athinput', [
        'job/basename=' + field_two,
    ] + field_arguments)
    athena.mpirun(4, 'tests/rsrmhd_antenna_field.athinput', [
        'job/basename=' + field_four,
    ] + field_arguments)

    common = field_arguments.copy()
    athena.mpirun(2, 'tests/rsrmhd_antenna_source.athinput', [
        'job/basename=' + reference,
        'problem/profile_name=' + reference,
    ] + common)
    athena.mpirun(1, 'tests/rsrmhd_antenna_source.athinput', [
        'job/basename=' + split,
        'problem/profile_name=' + split,
        'time/nlim=4',
    ] + common)

    checkpoints = sorted(glob.glob('build/src/rst/' + split + '.*.rst'))
    if not checkpoints:
        raise RuntimeError('No rank-changing antenna checkpoint was written')
    checkpoint = os.path.relpath(checkpoints[-1], 'build/src')
    athena.mpirestart(2, checkpoint, [
        'job/basename=' + restarted,
        'problem/profile_name=' + restarted,
        'time/nlim=8',
        'time/tlim=100.0',
    ])

    athena.mpirun(4, 'tests/rsrmhd_antenna_source.athinput', [
        'job/basename=' + reference_four,
        'problem/profile_name=' + reference_four,
    ] + common)
    athena.mpirun(4, 'tests/rsrmhd_antenna_source.athinput', [
        'job/basename=' + split_four,
        'problem/profile_name=' + split_four,
        'time/nlim=4',
    ] + common)
    checkpoints_four = sorted(glob.glob(
        'build/src/rst/' + split_four + '.*.rst'))
    if not checkpoints_four:
        raise RuntimeError('No four-rank antenna checkpoint was written')
    checkpoint_four = os.path.relpath(checkpoints_four[-1], 'build/src')
    athena.mpirestart(4, checkpoint_four, [
        'job/basename=' + restarted_four,
        'problem/profile_name=' + restarted_four,
        'time/nlim=8',
        'time/tlim=100.0',
    ])
    athena.mpirun(4, 'tests/rsrmhd_antenna_coupled.athinput', [
        'job/basename=' + coupled,
        'problem/profile_name=' + coupled,
        'problem/restart_state_name=' + coupled + '_state',
    ])
    athena.mpirun(4, 'tests/rsrmhd_antenna_coupled.athinput', [
        'job/basename=' + coupled_split,
        'problem/profile_name=' + coupled_split,
        'problem/restart_state_name=' + coupled_split + '_state',
        'time/nlim=2',
        'output2/dcycle=2',
    ])
    coupled_checkpoints = sorted(glob.glob(
        'build/src/rst/' + coupled_split + '.*.rst'))
    if not coupled_checkpoints:
        raise RuntimeError('No four-rank coupled checkpoint was written')
    coupled_checkpoint = os.path.relpath(
        coupled_checkpoints[-1], 'build/src')
    athena.mpirestart(4, coupled_checkpoint, [
        'job/basename=' + coupled_restarted,
        'problem/profile_name=' + coupled_restarted,
        'problem/restart_state_name=' + coupled_restarted + '_state',
        'time/nlim=4',
        'time/tlim=10.0',
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


def _final_state(basename):
    paths = sorted(glob.glob('build/src/' + basename + '_state*.dat'))
    if not paths:
        raise RuntimeError('Missing final state for ' + basename)
    state = np.vstack([np.atleast_2d(np.loadtxt(path)) for path in paths])
    order = np.lexsort((state[:, 2], state[:, 1], state[:, 0]))
    return state[order]


def analyze():
    logger.debug('Analyzing test ' + __name__)
    success = True
    field_one = 'rsrmhd_antenna_mpi_field_one'
    field_runs = (
        ('two-rank', 'rsrmhd_antenna_mpi_field_two'),
        ('four-rank', 'rsrmhd_antenna_mpi_field_four'),
    )
    for output_id in ('antenna', 'antenna_applied'):
        one = _final_binary(field_one, output_id)
        for rank_label, basename in field_runs:
            other = _final_binary(basename, output_id)
            if not np.array_equal(one, other):
                logger.warning('%s one-rank/%s current differs by %.3e',
                               output_id, rank_label,
                               np.max(np.abs(one - other)))
                success = False

    reference = 'rsrmhd_antenna_mpi_reference'
    restarted = 'rsrmhd_antenna_mpi_restarted'
    for output_id in ('state', 'electric', 'antenna'):
        continuous = _final_binary(reference, output_id)
        resumed = _final_binary(restarted, output_id)
        if not np.allclose(continuous, resumed, rtol=0.0, atol=8.0e-13):
            logger.warning('%s rank-changing restart differs by %.3e',
                           output_id, np.max(np.abs(continuous - resumed)))
            success = False

    history_ref = np.atleast_2d(np.loadtxt(
        'build/src/' + reference + '.user.hst'))
    history_rst = np.atleast_2d(np.loadtxt(
        'build/src/' + restarted + '.user.hst'))
    if history_ref.shape[1] != 43 or history_rst.shape[1] != 43:
        logger.warning('Antenna MPI histories have invalid shapes: %s %s',
                       history_ref.shape, history_rst.shape)
        success = False
    elif not np.allclose(history_ref[-1], history_rst[-1],
                         rtol=0.0, atol=2.0e-12):
        logger.warning('Antenna MPI restart history differs by %.3e',
                       np.max(np.abs(history_ref[-1] - history_rst[-1])))
        success = False
    named_history = athena_read.hst('build/src/' + reference + '.user.hst')
    expected_labels = {'jant_filt', 'jant_app', 'jant_comp'}
    missing_labels = expected_labels - set(named_history)
    if missing_labels:
        logger.warning('Antenna history labels are missing: %s', missing_labels)
        success = False

    reference_four = 'rsrmhd_antenna_mpi_reference_four'
    restarted_four = 'rsrmhd_antenna_mpi_restarted_four'
    for output_id in ('state', 'electric', 'antenna'):
        continuous = _final_binary(reference_four, output_id)
        resumed = _final_binary(restarted_four, output_id)
        if not np.allclose(continuous, resumed, rtol=0.0, atol=8.0e-13):
            logger.warning('%s four-rank restart differs by %.3e',
                           output_id, np.max(np.abs(continuous - resumed)))
            success = False

    history_ref_four = np.atleast_2d(np.loadtxt(
        'build/src/' + reference_four + '.user.hst'))
    history_rst_four = np.atleast_2d(np.loadtxt(
        'build/src/' + restarted_four + '.user.hst'))
    if (history_ref_four.shape[1] != 43
            or history_rst_four.shape[1] != 43):
        logger.warning('Four-rank antenna histories have invalid shapes: %s %s',
                       history_ref_four.shape, history_rst_four.shape)
        success = False
    elif not np.allclose(history_ref_four[-1], history_rst_four[-1],
                         rtol=0.0, atol=2.0e-12):
        logger.warning('Four-rank antenna restart history differs by %.3e',
                       np.max(np.abs(history_ref_four[-1]
                                     - history_rst_four[-1])))
        success = False

    coupled_name = 'rsrmhd_antenna_mpi_coupled_reference'
    coupled_restarted_name = 'rsrmhd_antenna_mpi_coupled_restarted'
    coupled = np.atleast_2d(np.loadtxt(
        'build/src/' + coupled_name + '.user.hst'))
    coupled = coupled[np.concatenate(([True], np.diff(coupled[:, 0]) != 0.0))]
    if coupled.shape != (5, 64) or not np.all(np.isfinite(coupled)):
        logger.warning('Coupled antenna MPI history is invalid: %s', coupled.shape)
        success = False
    else:
        if np.max(coupled[:, 13]) <= 0.0 or np.max(coupled[:, 49]) <= 0.0:
            logger.warning('Coupled mechanical or antenna forcing was inactive')
            success = False
        if np.max(np.abs(coupled[:, 50])) > 2.0e-12:
            logger.warning('Coupled MPI antenna divergence = %.3e',
                           np.max(np.abs(coupled[:, 50])))
            success = False
        if np.max(np.abs(coupled[:, 61])) <= 2.0e-12:
            logger.warning('Coupled FC-E antenna layout filter was inactive')
            success = False
        if np.max(np.abs(coupled[:, 63])) > 2.0e-12:
            logger.warning('Coupled MPI face/cell compatibility residual = %.3e',
                           np.max(np.abs(coupled[:, 63])))
            success = False

    coupled_state = _final_state(coupled_name)
    coupled_restarted_state = _final_state(coupled_restarted_name)
    if (coupled_state.shape != (512, 29)
            or coupled_restarted_state.shape != coupled_state.shape):
        logger.warning('Four-rank coupled states have invalid shapes: %s %s',
                       coupled_state.shape, coupled_restarted_state.shape)
        success = False
    elif not np.allclose(coupled_state, coupled_restarted_state,
                         rtol=0.0, atol=8.0e-12):
        logger.warning('Four-rank coupled restart state differs by %.3e',
                       np.max(np.abs(coupled_state
                                     - coupled_restarted_state)))
        success = False

    coupled_restarted = np.atleast_2d(np.loadtxt(
        'build/src/' + coupled_restarted_name + '.user.hst'))
    if coupled_restarted.shape[1] != 64:
        logger.warning('Four-rank coupled restart history is invalid: %s',
                       coupled_restarted.shape)
        success = False
    elif not np.allclose(coupled[-1], coupled_restarted[-1],
                         rtol=0.0, atol=1.0e-11):
        logger.warning('Four-rank coupled restart history differs by %.3e',
                       np.max(np.abs(coupled[-1] - coupled_restarted[-1])))
        success = False
    return success
