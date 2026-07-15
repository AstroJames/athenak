"""Statistical regression for the isotropic Fourier forcing spectrum."""

import glob
import logging
import sys
from pathlib import Path

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'vis/python'))
import bin_convert  # noqa

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_forcing_spectrum.athinput', [])


def _merge_force(path):
    data = bin_convert.read_binary(path)
    shape = (data['Nx3'], data['Nx2'], data['Nx1'])
    block_shape = (data['nx3_out_mb'], data['nx2_out_mb'],
                   data['nx1_out_mb'])
    force = [np.empty(shape) for _ in range(3)]
    for block, logical in enumerate(data['mb_logical']):
        block_x, block_y, block_z, level = logical
        if level != 0:
            raise ValueError('Forcing-spectrum test requires a uniform mesh')
        xs = slice(block_x*block_shape[2], (block_x + 1)*block_shape[2])
        ys = slice(block_y*block_shape[1], (block_y + 1)*block_shape[1])
        zs = slice(block_z*block_shape[0], (block_z + 1)*block_shape[0])
        for component in range(3):
            force[component][zs, ys, xs] = data['mb_data'][
                f'force{component + 1}'][block]
    return force


def analyze():
    logger.debug('Analyzing test ' + __name__)
    paths = sorted(glob.glob(
        'build/src/bin/rsrmhd_forcing_spectrum.force.*.bin'))
    if len(paths) < 100:
        logger.warning('Too few forcing realizations: %d', len(paths))
        return False

    shell_counts = {1: 6, 2: 12, 3: 8, 4: 6,
                    5: 24, 6: 24, 8: 12, 9: 30}
    expected = np.array(list(shell_counts.values()), dtype=float)
    expected /= np.sum(expected)
    fractions = []
    max_leakage = 0.0

    for path in paths:
        force = _merge_force(path)
        nz, ny, nx = force[0].shape
        nx_mode = np.rint(np.fft.fftfreq(nx, d=1.0/nx)).astype(int)
        ny_mode = np.rint(np.fft.fftfreq(ny, d=1.0/ny)).astype(int)
        nz_mode = np.rint(np.fft.fftfreq(nz, d=1.0/nz)).astype(int)
        nz_grid, ny_grid, nx_grid = np.meshgrid(
            nz_mode, ny_mode, nx_mode, indexing='ij')
        mode2 = nx_grid**2 + ny_grid**2 + nz_grid**2
        power = sum(np.abs(np.fft.fftn(component))**2
                    for component in force)
        driven = (mode2 >= 1) & (mode2 <= 9)
        driven_power = np.sum(power[driven])
        total_power = np.sum(power)
        if driven_power <= 0.0 or total_power <= 0.0:
            continue
        max_leakage = max(max_leakage,
                          1.0 - driven_power/total_power)
        fractions.append([
            np.sum(power[mode2 == shell])/driven_power
            for shell in shell_counts
        ])

    if len(fractions) < 100:
        logger.warning('Too few nonzero forcing realizations: %d', len(fractions))
        return False
    if max_leakage > 1.0e-6:
        logger.warning('Forcing power leaked outside the driven shell: %g',
                       max_leakage)
        return False

    measured = np.mean(fractions, axis=0)
    error = np.max(np.abs(measured - expected))
    if error > 2.0e-2:
        logger.warning('Shell power does not follow signed-mode degeneracy: '
                       'measured=%s expected=%s', measured, expected)
        return False
    return True
