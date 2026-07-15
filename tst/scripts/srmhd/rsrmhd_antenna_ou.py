"""Stationary variance and complex correlation of the exact antenna OU update."""

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
    athena.run('tests/rsrmhd_antenna_field.athinput', [
        'job/basename=rsrmhd_antenna_ou',
        'time/nlim=1024',
        'time/tlim=10000.0',
        'time/ndiag=1024',
        'time/cfl_number=0.8',
        'antenna_driving/decorrelation_factor=0.5',
        'antenna_driving/amplitude_normalization=zhdankin',
        'antenna_driving/amplitude_fraction_plus=1.0',
        'antenna_driving/amplitude_fraction_minus=0.0',
    ])


def _merge_j3(path):
    data = bin_convert.read_binary(path)
    shape = (data['Nx3'], data['Nx2'], data['Nx1'])
    block_shape = (data['nx3_out_mb'], data['nx2_out_mb'],
                   data['nx1_out_mb'])
    current3 = np.empty(shape)
    for block, logical in enumerate(data['mb_logical']):
        block_x, block_y, block_z, level = logical
        if level != 0:
            raise ValueError('Antenna OU test requires a uniform mesh')
        xs = slice(block_x*block_shape[2], (block_x + 1)*block_shape[2])
        ys = slice(block_y*block_shape[1], (block_y + 1)*block_shape[1])
        zs = slice(block_z*block_shape[0], (block_z + 1)*block_shape[0])
        current3[zs, ys, xs] = data['mb_data']['jant3'][block]
    return float(data['time']), current3


def analyze():
    logger.debug('Analyzing test ' + __name__)
    samples = [_merge_j3(path) for path in sorted(glob.glob(
        'build/src/bin/rsrmhd_antenna_ou.antenna.*.bin'))]
    samples = [sample for sample in samples if np.linalg.norm(sample[1]) > 0.0]
    samples = [sample for index, sample in enumerate(samples)
               if index == 0 or sample[0] != samples[index - 1][0]]
    if len(samples) != 1024:
        logger.warning('OU test produced %d unique samples', len(samples))
        return False

    modes = ((1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1))
    coefficients = np.empty((len(samples), len(modes)), dtype=complex)
    times = np.asarray([sample[0] for sample in samples])
    for sample_index, (_, current3) in enumerate(samples):
        transform = np.fft.fftn(current3)
        nz, ny, nx = current3.shape
        for mode_index, (kx, ky, kz) in enumerate(modes):
            coefficients[sample_index, mode_index] = transform[
                kz % nz, ky % ny, kx % nx]

    # For beta=1 and P=1, AthenaK's rationalized-unit guide field is sqrt(2).
    # Zhdankin's Gaussian-unit current B0_G/(4 L) converts to pi B0/L in
    # AthenaK. A real mode's positive Fourier coefficient contributes Ncells/2.
    coefficient_rms = samples[0][1].size*np.pi*np.sqrt(2.0)/2.0
    expected_variance = coefficient_rms**2
    measured_variance = np.mean(np.abs(coefficients)**2)
    variance_ratio = measured_variance/expected_variance
    if abs(variance_ratio - 1.0) > 0.20:
        logger.warning('OU stationary variance ratio = %.6f', variance_ratio)
        return False

    dt = np.mean(np.diff(times))
    omega_a = 2.0*np.pi*0.5/np.sqrt(3.0)
    expected_correlation = np.exp(
        -(0.5 + 0.6j)*omega_a*dt)
    measured_correlation = np.mean(
        coefficients[1:]*np.conjugate(coefficients[:-1])
    )/np.mean(np.abs(coefficients[:-1])**2)
    correlation_error = abs(measured_correlation - expected_correlation)
    if correlation_error > 0.035:
        logger.warning('OU lag-one correlation error = %.6f', correlation_error)
        return False

    normalized_mean = abs(np.mean(coefficients))/np.sqrt(expected_variance)
    if normalized_mean > 0.15:
        logger.warning('OU normalized sample mean = %.6f', normalized_mean)
        return False
    return True
