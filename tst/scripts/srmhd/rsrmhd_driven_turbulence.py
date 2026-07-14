"""Regressions for mechanically driven coupled SRMHD turbulence."""

import glob
import logging
import sys

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import bin_convert  # noqa

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/rsrmhd_driven_turbulence.athinput', [
        'mesh/nx1=32',
        'mesh/nx2=32',
        'meshblock/nx1=16',
        'meshblock/nx2=16',
        'time/tlim=0.2',
        'time/ndiag=10',
        'output2/dt=0.2',
        'output3/dt=0.2',
        'problem/profile_name=rsrmhd_driven_turbulence_test',
    ])
    athena.run('tests/rsrmhd_driven_turbulence.athinput', [
        'job/basename=rsrmhd_driven_turbulence_3d',
        'mesh/nx1=8',
        'mesh/nx2=8',
        'mesh/nx3=8',
        'meshblock/nx1=8',
        'meshblock/nx2=8',
        'meshblock/nx3=8',
        'time/tlim=0.1',
        'time/ndiag=5',
        'output2/dt=0.1',
        'output3/dt=0.1',
        'problem/velocity_rms=0.0',
        'problem/magnetic_configuration=uniform_z',
        'problem/plasma_beta=1.0',
        'problem/profile_name=rsrmhd_driven_turbulence_3d_test',
    ])


def _merge_force(path):
    data = bin_convert.read_binary(path)
    nx1 = data['Nx1']
    nx2 = data['Nx2']
    nx1_mb = data['nx1_out_mb']
    nx2_mb = data['nx2_out_mb']
    force1 = np.empty((nx2, nx1))
    force2 = np.empty((nx2, nx1))
    for block, logical in enumerate(data['mb_logical']):
        block_x, block_y, block_z, level = logical
        if block_z != 0 or level != 0:
            raise ValueError('Driven-turbulence test requires a uniform 2D mesh')
        xs = slice(block_x*nx1_mb, (block_x + 1)*nx1_mb)
        ys = slice(block_y*nx2_mb, (block_y + 1)*nx2_mb)
        force1[ys, xs] = data['mb_data']['force1'][block][0]
        force2[ys, xs] = data['mb_data']['force2'][block][0]
    return force1, force2


def _merge_force_3d(path):
    data = bin_convert.read_binary(path)
    shape = (data['Nx3'], data['Nx2'], data['Nx1'])
    block_shape = (data['nx3_out_mb'], data['nx2_out_mb'],
                   data['nx1_out_mb'])
    force = [np.empty(shape) for _ in range(3)]
    for block, logical in enumerate(data['mb_logical']):
        block_x, block_y, block_z, level = logical
        if level != 0:
            raise ValueError('Driven-turbulence test requires a uniform mesh')
        xs = slice(block_x*block_shape[2], (block_x + 1)*block_shape[2])
        ys = slice(block_y*block_shape[1], (block_y + 1)*block_shape[1])
        zs = slice(block_z*block_shape[0], (block_z + 1)*block_shape[0])
        for component in range(3):
            force[component][zs, ys, xs] = data['mb_data'][
                f'force{component + 1}'][block]
    return force


def analyze():
    logger.debug('Analyzing test ' + __name__)
    history = np.atleast_2d(np.loadtxt(
        'build/src/rsrmhd_driven_turbulence.user.hst'))
    if history.shape != (5, 38) or not np.all(np.isfinite(history)):
        logger.warning('Driven-turbulence history is invalid: shape=%s',
                       history.shape)
        return False

    target_rms = 0.25
    if np.max(np.abs(history[1:, 13]/target_rms - 1.0)) > 2.0e-12:
        logger.warning('Mechanical acceleration RMS missed its target: %s',
                       history[:, 13])
        return False
    if np.max(np.abs(history[1:, 14:17])) > 2.0e-11:
        logger.warning('Mechanical source has nonzero net momentum: %s',
                       history[:, 14:17])
        return False

    energy_residual = history[:, 20] - history[0, 20] - history[:, 21]
    if np.max(np.abs(energy_residual)) > 2.0e-10:
        logger.warning('Driven total-energy audit failed: %s', energy_residual)
        return False
    momentum_residual = history[:, 17:20] - history[0, 17:20] - history[:, 22:25]
    if np.max(np.abs(momentum_residual)) > 2.0e-10:
        logger.warning('Driven total-momentum audit failed: %s', momentum_residual)
        return False
    if history[-1, 21] <= 0.0 or history[-1, 3] <= history[0, 3]:
        logger.warning('The driver did not grow turbulent kinetic energy: %g %g %g',
                       history[-1, 21], history[0, 3], history[-1, 3])
        return False
    if np.max(history[:, 8]) <= 0.0 or np.max(history[:, 7]) <= 0.0:
        logger.warning('The coupled run did not develop shear/current structure')
        return False
    if np.max(history[:, 11]) > 1.0e-20:
        logger.warning('The driven run violates div(B)=0: %g',
                       np.max(history[:, 11]))
        return False
    mass_error = np.max(np.abs(history[:, 26]/history[0, 26] - 1.0))
    if mass_error > 2.0e-11:
        logger.warning('Driven rest-mass conservation failed: %g', mass_error)
        return False
    if np.any(history[:, 35:37] < -1.0e-14):
        logger.warning('Driven heating estimators are negative: %s',
                       history[:, 35:37])
        return False

    diagnostics = np.loadtxt(
        'build/src/rsrmhd_driven_turbulence_test-forcing.dat')
    if diagnostics.shape != (10,) or not np.all(np.isfinite(diagnostics)):
        logger.warning('Driven final diagnostics are invalid: %s', diagnostics)
        return False
    if diagnostics[0] <= 0.0 or diagnostics[1] <= 0.0 or diagnostics[2] >= 2.0:
        logger.warning('Driven final primitive bounds are invalid: %s', diagnostics[:3])
        return False
    if np.any(diagnostics[3:8] != 0):
        logger.warning('Driven run used FOFC/floors or failed recovery: %s',
                       diagnostics[3:8])
        return False

    force_files = sorted(glob.glob(
        'build/src/bin/rsrmhd_driven_turbulence.force.*.bin'))
    if len(force_files) != 2:
        logger.warning('Expected initial and final 2D force dumps: %s', force_files)
        return False
    force1, force2 = _merge_force(force_files[-1])
    nx2, nx1 = force1.shape
    kx = 2.0*np.pi*np.fft.fftfreq(nx1, d=1.0/nx1)
    ky = 2.0*np.pi*np.fft.fftfreq(nx2, d=1.0/nx2)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    force1_hat = np.fft.fft2(force1)
    force2_hat = np.fft.fft2(force2)
    divergence_hat = 1j*(kx_grid*force1_hat + ky_grid*force2_hat)
    gradient_norm = np.sqrt(np.sum(
        (kx_grid**2 + ky_grid**2)
        * (np.abs(force1_hat)**2 + np.abs(force2_hat)**2)))
    divergence_ratio = np.sqrt(np.sum(np.abs(divergence_hat)**2))/gradient_norm
    if divergence_ratio > 2.0e-5:
        logger.warning('The 2D solenoidal force is not divergence-free: %g',
                       divergence_ratio)
        return False

    history_3d = np.atleast_2d(np.loadtxt(
        'build/src/rsrmhd_driven_turbulence_3d.user.hst'))
    if history_3d.shape != (3, 38) or not np.all(np.isfinite(history_3d)):
        logger.warning('3D driven history is invalid: shape=%s', history_3d.shape)
        return False
    initial_expected = {
        'kinetic energy': (history_3d[0, 3], 0.0),
        'magnetic energy': (history_3d[0, 4], 1.0),
        'electric energy': (history_3d[0, 5], 0.0),
        'density integral': (history_3d[0, 9], 1.0),
        'pressure integral': (history_3d[0, 10], 1.0),
        'velocity variance': (history_3d[0, 28], 0.0),
        'Lorentz-factor integral': (history_3d[0, 31], 1.0),
        'magnetization': (history_3d[0, 33], 4.0/7.0),
        'Alfven speed': (history_3d[0, 37], np.sqrt(4.0/11.0)),
    }
    for label, (actual, expected) in initial_expected.items():
        if not np.isclose(actual, expected, rtol=0.0, atol=2.0e-13):
            logger.warning('Incorrect initial 3D %s: %.17g != %.17g',
                           label, actual, expected)
            return False
    energy_residual = (
        history_3d[:, 20] - history_3d[0, 20] - history_3d[:, 21]
    )
    if np.max(np.abs(energy_residual)) > 2.0e-10:
        logger.warning('3D total-energy audit failed: %s', energy_residual)
        return False
    if np.max(history_3d[:, 11]) > 1.0e-20:
        logger.warning('The 3D run violates div(B)=0: %g',
                       np.max(history_3d[:, 11]))
        return False
    if np.max(history_3d[:, 6]) <= 0.0 or np.max(history_3d[:, 7]) <= 0.0:
        logger.warning('The 3D run did not develop curl structure')
        return False

    diagnostics_3d = np.loadtxt(
        'build/src/rsrmhd_driven_turbulence_3d_test-forcing.dat')
    if diagnostics_3d.shape != (10,) or np.any(diagnostics_3d[3:8] != 0):
        logger.warning('3D final diagnostics are invalid: %s', diagnostics_3d)
        return False

    force_files_3d = sorted(glob.glob(
        'build/src/bin/rsrmhd_driven_turbulence_3d.force.*.bin'))
    if len(force_files_3d) != 2:
        logger.warning('Expected initial and final 3D force dumps: %s',
                       force_files_3d)
        return False
    force1, force2, force3 = _merge_force_3d(force_files_3d[-1])
    nz, ny, nx = force1.shape
    kx = 2.0*np.pi*np.fft.fftfreq(nx, d=1.0/nx)
    ky = 2.0*np.pi*np.fft.fftfreq(ny, d=1.0/ny)
    kz = 2.0*np.pi*np.fft.fftfreq(nz, d=1.0/nz)
    kz_grid, ky_grid, kx_grid = np.meshgrid(kz, ky, kx, indexing='ij')
    force_hat = [np.fft.fftn(component)
                 for component in (force1, force2, force3)]
    divergence_hat = 1j*(kx_grid*force_hat[0] + ky_grid*force_hat[1]
                         + kz_grid*force_hat[2])
    k2 = kx_grid**2 + ky_grid**2 + kz_grid**2
    gradient_norm = np.sqrt(np.sum(
        k2*sum(np.abs(component)**2 for component in force_hat)))
    divergence_ratio = np.sqrt(np.sum(np.abs(divergence_hat)**2))/gradient_norm
    if divergence_ratio > 2.0e-5:
        logger.warning('The 3D solenoidal force is not divergence-free: %g',
                       divergence_ratio)
        return False
    return True
