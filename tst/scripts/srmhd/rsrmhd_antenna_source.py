"""Conservative-source regression for the SRRMHD antenna driver."""

import logging
from pathlib import Path

import numpy as np

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for layout, electric_ct in (('cell', 'false'), ('face', 'true')):
        Path(f'build/src/rsrmhd_antenna_source_{layout}.user.hst').unlink(
            missing_ok=True)
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
        if history.shape != (9, 43) or not np.all(np.isfinite(history)):
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
        layout_filter = np.max(np.abs(history[:, 40]))
        if layout == 'cell' and layout_filter > 2.0e-12:
            logger.warning('CC-E antenna layout filter = %.3e', layout_filter)
            success = False
        if layout == 'face' and layout_filter <= 2.0e-12:
            logger.warning('FC-E antenna did not apply the compatible layout filter')
            success = False
        applied_rms = history[:, 41]
        expected_ratio = (1.0 if layout == 'cell'
                          else np.cos(np.pi/8.0)**2)
        if not np.allclose(applied_rms, expected_ratio*history[:, 13],
                           rtol=0.0, atol=2.0e-12):
            logger.warning('%s applied-current RMS has the wrong layout filter',
                           layout)
            success = False
        if np.max(np.abs(history[:, 42])) > 2.0e-12:
            logger.warning('%s face/cell compatibility residual = %.3e',
                           layout, np.max(np.abs(history[:, 42])))
            success = False

    if all(layout in histories and histories[layout].shape == (9, 43)
           for layout in ('cell', 'face')):
        # The canonical current is common to both layouts; only the actual FC source
        # receives the expected A_i^T A_i filter.
        if not np.allclose(histories['cell'][:, 13:15],
                           histories['face'][:, 13:15],
                           rtol=0.0, atol=2.0e-12):
            logger.warning('CC-E and FC-E current diagnostics differ')
            success = False
    return success
