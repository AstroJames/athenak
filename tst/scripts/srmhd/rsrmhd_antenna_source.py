"""Conservative-source regression for the SRRMHD antenna driver."""

import logging

import numpy as np

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for layout, electric_ct in (('cell', 'false'), ('face', 'true')):
        athena.run('tests/rsrmhd_antenna_source.athinput', [
            'job/basename=rsrmhd_antenna_source_' + layout,
            'problem/profile_name=rsrmhd_antenna_source_' + layout,
            'mhd/electric_ct=' + electric_ct,
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    success = True
    histories = {}
    for layout in ('cell', 'face'):
        history = np.atleast_2d(np.loadtxt(
            f'build/src/rsrmhd_antenna_source_{layout}.user.hst'))
        history = history[np.concatenate(([True], np.diff(history[:, 0]) != 0.0))]
        histories[layout] = history
        if history.shape != (9, 41) or not np.all(np.isfinite(history)):
            logger.warning('%s antenna history is invalid: shape=%s',
                           layout, history.shape)
            success = False
            continue

        energy_residual = history[:, 28] - history[0, 28] - history[:, 18]
        momentum_residual = (
            history[:, 25:28] - history[0, 25:28] - history[:, 19:22]
        )
        if np.max(np.abs(energy_residual)) > 3.0e-11:
            logger.warning('%s antenna energy audit failed: %s',
                           layout, energy_residual)
            success = False
        if np.max(np.abs(momentum_residual)) > 3.0e-11:
            logger.warning('%s antenna momentum audit failed: %s',
                           layout, momentum_residual)
            success = False
        if history[-1, 18] <= 0.0 or np.max(history[:, 13]) <= 0.0:
            logger.warning('%s antenna source was inactive', layout)
            success = False
        if np.max(np.abs(history[:, 14])) > 2.0e-12:
            logger.warning('%s antenna current divergence = %.3e',
                           layout, np.max(np.abs(history[:, 14])))
            success = False
        if np.max(np.abs(history[:, 40])) > 2.0e-12:
            logger.warning('%s antenna face/cell mismatch = %.3e',
                           layout, np.max(np.abs(history[:, 40])))
            success = False

    if all(layout in histories and histories[layout].shape == (9, 41)
           for layout in ('cell', 'face')):
        # The compatible face representation averages to the CC current exactly.
        if not np.array_equal(histories['cell'][:, 13:15],
                              histories['face'][:, 13:15]):
            logger.warning('CC-E and FC-E current diagnostics differ')
            success = False
    return success
