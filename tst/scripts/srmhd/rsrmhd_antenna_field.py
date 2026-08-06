"""Field-only regression for the Zhdankin eight-mode antenna."""

import glob
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'vis/python'))
import bin_convert  # noqa: E402

logger = logging.getLogger('athena' + __name__[7:])


def _expect_input_failure(arguments, expected_message):
    root = Path(__file__).resolve().parents[3]
    executable = root / 'tst/build/src/athena'
    input_file = root / 'inputs/tests/rsrmhd_antenna_field.athinput'
    result = subprocess.run(
        [str(executable), '-i', str(input_file)] + arguments,
        cwd=root / 'tst/build/src', capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode == 0:
        raise RuntimeError('Invalid antenna input unexpectedly succeeded: '
                           + ' '.join(arguments))
    if expected_message not in output:
        raise RuntimeError(
            f'Invalid antenna input did not report {expected_message!r}: {output}')


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for layout, electric_ct in (('cell', 'false'), ('face', 'true')):
        athena.run('tests/rsrmhd_antenna_field.athinput', [
            'job/basename=rsrmhd_antenna_field_' + layout,
            'mhd/electric_ct=' + electric_ct,
        ])
    athena.run('tests/rsrmhd_antenna_field.athinput', [
        'job/basename=rsrmhd_antenna_field_alfven',
        'mhd/electric_ct=false',
        'antenna_driving/frequency_model=alfven_parallel',
    ])
    for axis in ('x', 'y'):
        athena.run('tests/rsrmhd_antenna_field.athinput', [
            f'job/basename=rsrmhd_antenna_field_guide_{axis}',
            'mhd/electric_ct=false',
            f'problem/magnetic_configuration=uniform_{axis}',
            f'antenna_driving/guide_axis={axis}',
        ])
    athena.run('tests/rsrmhd_antenna_field.athinput', [
        'job/basename=rsrmhd_antenna_field_band',
        'mhd/electric_ct=false',
        'antenna_driving/mode_set=anisotropic_band',
        'antenna_driving/nperp_min=1',
        'antenna_driving/nperp_max=2',
        'antenna_driving/nparallel_min=1',
        'antenna_driving/nparallel_max=2',
        'antenna_driving/current_envelope=band',
        'antenna_driving/frequency_model=alfven_parallel',
    ])
    athena.run('tests/rsrmhd_antenna_field.athinput', [
        'job/basename=rsrmhd_antenna_field_band_face',
        'mhd/electric_ct=true',
        'antenna_driving/mode_set=anisotropic_band',
        'antenna_driving/nperp_min=1',
        'antenna_driving/nperp_max=2',
        'antenna_driving/nparallel_min=1',
        'antenna_driving/nparallel_max=2',
        'antenna_driving/current_envelope=band',
        'antenna_driving/frequency_model=alfven_parallel',
    ])
    athena.run('tests/rsrmhd_antenna_field.athinput', [
        'job/basename=rsrmhd_antenna_field_powerlaw',
        'mhd/electric_ct=false',
        'antenna_driving/mode_set=anisotropic_band',
        'antenna_driving/nperp_min=1',
        'antenna_driving/nperp_max=2',
        'antenna_driving/nparallel_min=1',
        'antenna_driving/nparallel_max=2',
        'antenna_driving/current_envelope=powerlaw',
        'antenna_driving/current_exponent_perp=1.0',
        'antenna_driving/current_exponent_parallel=0.0',
        'antenna_driving/frequency_model=alfven_parallel',
    ])
    for extent in (1, 3):
        athena.run('tests/rsrmhd_antenna_field.athinput', [
            f'job/basename=rsrmhd_antenna_field_parabolic_{extent}',
            'mhd/electric_ct=false',
            'antenna_driving/mode_set=anisotropic_band',
            'antenna_driving/nperp_min=1',
            f'antenna_driving/nperp_max={extent}',
            'antenna_driving/nparallel_min=1',
            'antenna_driving/nparallel_max=1',
            'antenna_driving/current_envelope=parabolic',
            'antenna_driving/current_parabola_peak=1.0',
            'antenna_driving/current_parabola_width=0.1',
            'antenna_driving/frequency_model=alfven_parallel',
        ])

    _expect_input_failure([
        'job/basename=rsrmhd_antenna_field_invalid_frequency',
        'antenna_driving/frequency_factor=nan',
    ], 'must be finite')
    _expect_input_failure([
        'job/basename=rsrmhd_antenna_field_invalid_envelope',
        'antenna_driving/mode_set=anisotropic_band',
        'antenna_driving/current_envelope=powerlaw',
        'antenna_driving/current_exponent_perp=nan',
    ], 'must be finite')
    _expect_input_failure([
        'job/basename=rsrmhd_antenna_field_invalid_alfven',
        'antenna_driving/va_reference=fixed',
        'antenna_driving/alfven_speed=nan',
    ], 'must be finite')
    _expect_input_failure([
        'job/basename=rsrmhd_antenna_field_noncubic_small',
        'mesh/x1max=1.0e-16',
        'mesh/x2max=2.0e-16',
        'mesh/x3max=1.0e-16',
    ], 'requires a cubic domain')
    _expect_input_failure([
        'job/basename=rsrmhd_antenna_field_misaligned_guide',
        'antenna_driving/guide_axis=x',
    ], 'positive mean field')


def _merge_current(path, applied=False):
    data = bin_convert.read_binary(path)
    shape = (data['Nx3'], data['Nx2'], data['Nx1'])
    block_shape = (data['nx3_out_mb'], data['nx2_out_mb'],
                   data['nx1_out_mb'])
    current = [np.empty(shape) for _ in range(3)]
    prefix = 'jant_applied' if applied else 'jant'
    for block, logical in enumerate(data['mb_logical']):
        block_x, block_y, block_z, level = logical
        if level != 0:
            raise ValueError('Antenna field test requires a uniform mesh')
        xs = slice(block_x*block_shape[2], (block_x + 1)*block_shape[2])
        ys = slice(block_y*block_shape[1], (block_y + 1)*block_shape[1])
        zs = slice(block_z*block_shape[0], (block_z + 1)*block_shape[0])
        for component in range(3):
            current[component][zs, ys, xs] = data['mb_data'][
                f'{prefix}{component + 1}'][block]
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
    zhdankin_modes = {
        (1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1),
        (-1, 0, -1), (0, -1, -1), (1, 0, -1), (0, 1, -1),
    }
    band_half = {
        (nx, ny, nz)
        for nz in range(1, 3)
        for nx in range(-2, 3)
        for ny in range(-2, 3)
        if 1 <= nx*nx + ny*ny <= 4
    }
    band_modes = band_half | {tuple(-value for value in mode)
                              for mode in band_half}
    parabolic_half = {
        (nx, ny, 1)
        for nx in range(-1, 2)
        for ny in range(-1, 2)
        if nx*nx + ny*ny == 1
    }
    parabolic_modes = parabolic_half | {
        tuple(-value for value in mode) for mode in parabolic_half}
    guide_x_half = {(1, 1, 0), (1, 0, 1), (1, -1, 0), (1, 0, -1)}
    guide_x_modes = guide_x_half | {
        tuple(-value for value in mode) for mode in guide_x_half}
    guide_y_half = {(1, 1, 0), (0, 1, 1), (-1, 1, 0), (0, 1, -1)}
    guide_y_modes = guide_y_half | {
        tuple(-value for value in mode) for mode in guide_y_half}

    layouts = ('cell', 'face', 'alfven', 'guide_x', 'guide_y', 'band',
               'band_face', 'powerlaw', 'parabolic_1', 'parabolic_3')
    for layout in layouts:
        if layout in ('band', 'band_face', 'powerlaw'):
            allowed = band_modes
        elif layout == 'guide_x':
            allowed = guide_x_modes
        elif layout == 'guide_y':
            allowed = guide_y_modes
        elif layout.startswith('parabolic_'):
            allowed = parabolic_modes
        else:
            allowed = zhdankin_modes
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

        applied_paths = sorted(glob.glob(
            f'build/src/bin/rsrmhd_antenna_field_{layout}.antenna_applied.*.bin'))
        applied_samples = [_merge_current(path, applied=True)
                           for path in applied_paths]
        applied_samples = [sample for sample in applied_samples
                           if sum(np.linalg.norm(v) for v in sample[1]) > 0.0]
        applied_samples = [sample for index, sample in enumerate(applied_samples)
                           if index == 0
                           or sample[0] != applied_samples[index - 1][0]]
        if len(applied_samples) != len(samples):
            logger.warning('%s antenna applied-current sample count differs', layout)
            success = False
            continue
        if not np.array_equal([sample[0] for sample in samples],
                              [sample[0] for sample in applied_samples]):
            logger.warning('%s raw/applied antenna sample times differ', layout)
            success = False
            continue

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
        minimum_mode_fraction = min(float(power[np.where(
            (mx == mode[0]) & (my == mode[1]) & (mz == mode[2]))][0]
            / total_power) for mode in allowed)
        if minimum_mode_fraction <= 1.0e-10:
            logger.warning('%s antenna mode set is incomplete: %.3e',
                           layout, minimum_mode_fraction)
            success = False

        applied_transforms = [np.fft.fftn(component)
                              for component in applied_samples[-1][1]]
        filter_error = 0.0
        for component, (raw_transform, applied_transform) in enumerate(
                zip(transforms, applied_transforms)):
            for mode in allowed:
                mode_index = np.where((mx == mode[0]) & (my == mode[1])
                                      & (mz == mode[2]))
                raw = complex(raw_transform[mode_index][0])
                # Binary output is single precision in this regression.  Components
                # whose analytic double-curl polarization vanishes retain FFT noise
                # at roughly 1e-9 of the total transform norm.
                if abs(raw) <= 1.0e-8*np.sqrt(total_power):
                    continue
                expected_filter = 1.0
                if layout in ('face', 'band_face'):
                    n_component = mode[component]
                    n_cells = current[component].shape[2 - component]
                    expected_filter = np.cos(np.pi*n_component/n_cells)**2
                measured_filter = complex(applied_transform[mode_index][0])/raw
                filter_error = max(filter_error,
                                   abs(measured_filter - expected_filter))
        if filter_error > 3.0e-7:
            logger.warning('%s antenna layout-filter error = %.3e',
                           layout, filter_error)
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

        tracked_modes = [((1, 0, 1), 1, 2)]
        if layout in ('guide_x', 'guide_y'):
            tracked_modes = [((1, 1, 0), 1, 0)]
        if layout in ('band', 'band_face', 'powerlaw'):
            tracked_modes.append(((1, 0, 2), 2, 2))
        for tracked_mode, parallel_index, tracked_component in tracked_modes:
            mode_index = np.where((mx == tracked_mode[0])
                                  & (my == tracked_mode[1])
                                  & (mz == tracked_mode[2]))
            coefficients = []
            times = []
            for time, sample_current in samples:
                transform = np.fft.fftn(sample_current[tracked_component])
                coefficients.append(complex(transform[mode_index][0]))
                times.append(time)
            coefficients = np.asarray(coefficients)
            times = np.asarray(times)
            omega_reference = 2.0*np.pi*0.5*parallel_index
            if (layout not in ('alfven', 'band', 'band_face', 'powerlaw')
                    and not layout.startswith('parabolic_')):
                omega_reference /= np.sqrt(3.0)
            omega = 0.6*omega_reference
            measured = coefficients[1:]/coefficients[:-1]
            expected = np.exp(-1j*omega*np.diff(times))
            rotation_error = np.max(np.abs(measured - expected))
            if rotation_error > 1.0e-7:
                logger.warning('%s mode %s exact rotation error = %.3e',
                               layout, tracked_mode, rotation_error)
                success = False

    if {'cell', 'face'} <= set(layout_series):
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
    if {'parabolic_1', 'parabolic_3'} <= set(layout_series):
        for narrow_sample, wide_sample in zip(layout_series['parabolic_1'],
                                              layout_series['parabolic_3']):
            if narrow_sample[0] != wide_sample[0]:
                success = False
            for narrow_component, wide_component in zip(narrow_sample[1],
                                                        wide_sample[1]):
                if not np.array_equal(narrow_component, wide_component):
                    logger.warning(
                        'Zero-envelope modes changed the active antenna field')
                    success = False
                    break
    if {'band', 'powerlaw'} <= set(layout_series):
        envelope_ratios = []
        for mode in ((1, 0, 1), (2, 0, 1)):
            band_current = np.fft.fftn(layout_series['band'][-1][1][2])
            powerlaw_current = np.fft.fftn(
                layout_series['powerlaw'][-1][1][2])
            mode_index = tuple(value % size for value, size in zip(
                mode[::-1], band_current.shape))
            envelope_ratios.append(
                powerlaw_current[mode_index]/band_current[mode_index])
        measured_envelope_ratio = envelope_ratios[0]/envelope_ratios[1]
        if abs(measured_envelope_ratio - 2.0) > 1.0e-6:
            logger.warning('Power-law antenna envelope ratio = %s',
                           measured_envelope_ratio)
            success = False
    return success
