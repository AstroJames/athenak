"""Field-only regression for the Zhdankin eight-mode antenna."""

import glob
import logging
import sys
from pathlib import Path

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'vis/python'))
import bin_convert  # noqa: E402

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for layout, electric_ct in (('cell', 'false'), ('face', 'true')):
        athena.run('tests/rsrmhd_antenna_field.athinput', [
            'job/basename=rsrmhd_antenna_field_' + layout,
            'mhd/electric_ct=' + electric_ct,
        ])


def _merge_current(path):
    data = bin_convert.read_binary(path)
    shape = (data['Nx3'], data['Nx2'], data['Nx1'])
    block_shape = (data['nx3_out_mb'], data['nx2_out_mb'],
                   data['nx1_out_mb'])
    current = [np.empty(shape) for _ in range(3)]
    for block, logical in enumerate(data['mb_logical']):
        block_x, block_y, block_z, level = logical
        if level != 0:
            raise ValueError('Antenna field test requires a uniform mesh')
        xs = slice(block_x*block_shape[2], (block_x + 1)*block_shape[2])
        ys = slice(block_y*block_shape[1], (block_y + 1)*block_shape[1])
        zs = slice(block_z*block_shape[0], (block_z + 1)*block_shape[0])
        for component in range(3):
            current[component][zs, ys, xs] = data['mb_data'][
                f'jant{component + 1}'][block]
    return float(data['time']), current


def _mode_arrays(shape):
    nz, ny, nx = shape
    mx = np.rint(np.fft.fftfreq(nx, d=1.0/nx)).astype(int)
    my = np.rint(np.fft.fftfreq(ny, d=1.0/ny)).astype(int)
    mz = np.rint(np.fft.fftfreq(nz, d=1.0/nz)).astype(int)
    return np.meshgrid(mz, my, mx, indexing='ij')


def analyze():
    logger.debug('Analyzing test ' + __name__)
    success = True
    layout_series = {}
    allowed = {
        (1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1),
        (-1, 0, -1), (0, -1, -1), (1, 0, -1), (0, 1, -1),
    }

    for layout in ('cell', 'face'):
        paths = sorted(glob.glob(
            f'build/src/bin/rsrmhd_antenna_field_{layout}.antenna.*.bin'))
        samples = [_merge_current(path) for path in paths]
        samples = [sample for sample in samples
                   if sum(np.linalg.norm(v) for v in sample[1]) > 0.0]
        samples = [sample for index, sample in enumerate(samples)
                   if index == 0 or sample[0] != samples[index - 1][0]]
        if len(samples) != 8:
            logger.warning('%s antenna produced %d nonzero samples',
                           layout, len(samples))
            success = False
            continue
        layout_series[layout] = samples

        _, current = samples[-1]
        transforms = [np.fft.fftn(component) for component in current]
        mz, my, mx = _mode_arrays(current[0].shape)
        power = sum(np.abs(transform)**2 for transform in transforms)
        allowed_mask = np.zeros(current[0].shape, dtype=bool)
        for mode in allowed:
            allowed_mask |= ((mx == mode[0]) & (my == mode[1])
                             & (mz == mode[2]))
        total_power = np.sum(power)
        leakage = np.sum(power[~allowed_mask])/total_power
        if leakage > 2.0e-14:
            logger.warning('%s antenna spectral leakage = %.3e', layout, leakage)
            success = False

        dx = 1.0/current[0].shape[2]
        qx = np.sin(2.0*np.pi*mx*dx)/dx
        qy = np.sin(2.0*np.pi*my*dx)/dx
        qz = np.sin(2.0*np.pi*mz*dx)/dx
        divergence = qx*transforms[0] + qy*transforms[1] + qz*transforms[2]
        divergence_error = np.max(np.abs(divergence))/np.sqrt(total_power)
        if divergence_error > 8.0e-8:
            logger.warning('%s discrete divergence error = %.3e',
                           layout, divergence_error)
            success = False

        mode_index = np.where((mx == 1) & (my == 0) & (mz == 1))
        coefficients = []
        times = []
        for time, sample_current in samples:
            transform = np.fft.fftn(sample_current[2])
            coefficients.append(complex(transform[mode_index][0]))
            times.append(time)
        coefficients = np.asarray(coefficients)
        times = np.asarray(times)
        omega = 0.6*2.0*np.pi*0.5/np.sqrt(3.0)
        measured = coefficients[1:]/coefficients[:-1]
        expected = np.exp(-1j*omega*np.diff(times))
        rotation_error = np.max(np.abs(measured - expected))
        if rotation_error > 5.0e-8:
            logger.warning('%s exact rotation error = %.3e',
                           layout, rotation_error)
            success = False

    if set(layout_series) == {'cell', 'face'}:
        for cell_sample, face_sample in zip(layout_series['cell'],
                                            layout_series['face']):
            if cell_sample[0] != face_sample[0]:
                success = False
            for cell_component, face_component in zip(cell_sample[1],
                                                       face_sample[1]):
                if not np.array_equal(cell_component, face_component):
                    logger.warning('CC-E and FC-E antenna fields differ')
                    success = False
                    break
    return success
