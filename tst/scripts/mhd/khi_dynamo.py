"""KHI dynamo setup checks. Build with --cmake '-DPROBLEM=khi_dynamo'."""

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


def run(**kwargs):
    for field in ('uniform_control', 'spectral'):
        for block in (32, 16):
            name = f'khi_dynamo_{field}_b{block}'
            for filename in glob.glob('build/src/bin/' + name + '.*.bin'):
                os.remove(filename)
            for filename in glob.glob('build/src/' + name + '.*.hst'):
                os.remove(filename)
            athena.run('mhd/khi_dynamo.athinput', [
                'job/basename=' + name,
                # Unit tests deliberately override the 1000-shear-time run default.
                'time/nlim=4', 'time/tlim=0.4', 'time/dt_max=1.0',
                'output7/dt=0', 'output8/dt=0',
                'output1/dt=0.1', 'output2/dt=0.1', 'output3/dt=0.1',
                'problem/initial_field=' + field,
                'problem/phase_2=0.4',
                'problem/sim_vz=0.03',
                'problem/sim_bx=0.01',
                'problem/sim_by=0.02',
                'problem/sim_bz=-0.03',
                f'meshblock/nx1={block}',
                f'meshblock/nx2={block}',
                f'meshblock/nx3={block}',
            ])
    name = 'khi_dynamo_box_units'
    for filename in (glob.glob('build/src/bin/' + name + '.*.bin')
                     + glob.glob('build/src/' + name + '.*.hst')):
        os.remove(filename)
    rho_ref = 55/18
    athena.run('mhd/khi_dynamo.athinput', [
        'job/basename=' + name, 'time/nlim=4', 'time/tlim=0.01',
        'time/dt_max=0.025', 'output7/dt=0', 'output8/dt=0',
        'output1/dt=0.0025', 'output2/dt=0.0025', 'output3/dt=0.0025',
        'mesh/x1min=-0.5', 'mesh/x1max=0.5',
        'mesh/x2min=-0.5', 'mesh/x2max=0.5',
        'mesh/x3min=-0.5', 'mesh/x3max=0.5',
        'problem/sim_layer_thickness=0.025', 'problem/sim_smoothing_size=0.025',
        'problem/x_tilde_1=0.25', 'problem/x_tilde_2=-0.25',
        'output10/slice_x1=-0.25', 'output11/slice_x1=0.25',
        'problem/sim_dens=1.6666666666666667', 'problem/sim_pres=4.0',
        'problem/phase_2=0.4', 'problem/sim_vz=0.03',
        f'problem/sim_bx={0.01*np.sqrt(rho_ref)}',
        f'problem/sim_by={0.02*np.sqrt(rho_ref)}',
        f'problem/sim_bz={-0.03*np.sqrt(rho_ref)}',
        f'problem/magnetic_rms={np.sqrt(9.086e-7)}',
        'mhd/viscosity=0.0000244140625', 'mhd/ohmic_resistivity=0.0000244140625',
    ])


def _read(name, product, number):
    return bin_convert.read_binary(
        f'build/src/bin/{name}.{product}.{number:05d}.bin')


def _assemble(data, variable):
    result = np.empty((32, 32, 32))
    for m, logical in enumerate(data['mb_logical']):
        value = data['mb_data'][variable][m]
        n = value.shape[0]
        x, y, z, level = logical
        assert level == 0
        result[z*n:(z+1)*n, y*n:(y+1)*n, x*n:(x+1)*n] = value
    return result


def analyze():
    try:
        center = -0.5 + (np.arange(32) + 0.5)/32
        z, y, x = np.meshgrid(center, center, center, indexing='ij')
        layer = np.tanh((x + 0.25)/0.025) - np.tanh((x - 0.25)/0.025)
        expected = {
            'dens': (6/11)*(1 + 0.5*(5/3)*layer),
            'velx': 0.05*np.sin(4*np.pi*y + 0.4)
                    * (np.exp(-((x + 0.25)/0.025)**2)
                       + np.exp(-((x - 0.25)/0.025)**2)),
            'vely': 0.5*(layer - 1),
            'velz': 0.03*np.sin(4*np.pi*z),
            'eint': (72/55)/(5/3 - 1),
        }
        for field in ('uniform_control', 'spectral'):
            reference = None
            for block in (32, 16):
                name = f'khi_dynamo_{field}_b{block}'
                initial = _read(name, 'prim', 0)
                fields = {v: _assemble(initial, v) for v in initial['var_names']}
                for var, value in expected.items():
                    np.testing.assert_allclose(fields[var], value,
                                               rtol=2e-6, atol=1e-8)
                fluctuation2 = 0.0
                for component, mean in zip(('bcc1', 'bcc2', 'bcc3'),
                                           (0.01, 0.02, -0.03)):
                    np.testing.assert_allclose(fields[component].mean(), mean,
                                               rtol=0, atol=2e-9)
                    fluctuation2 += np.mean((fields[component] - mean)**2)
                target = np.sqrt(9.086e-7/(55/18)) if field == 'spectral' else 0.0
                np.testing.assert_allclose(np.sqrt(fluctuation2), target,
                                           rtol=2e-6, atol=2e-9)
                if reference is not None:
                    for var in fields:
                        np.testing.assert_allclose(fields[var], reference[var],
                                                   rtol=2e-6, atol=1e-8)
                reference = fields
                divergence_files = glob.glob(f'build/src/bin/{name}.divb.*.bin')
                assert divergence_files
                for filename in divergence_files:
                    div = bin_convert.read_binary(filename)
                    for values in div['mb_data'].values():
                        assert np.max(np.abs(values)) < 1e-12
                snapshots = sorted(glob.glob(f'build/src/bin/{name}.prim.*.bin'))
                final = bin_convert.read_binary(snapshots[-1])
                assert final['cycle'] > 0 and final['time'] > 0
                for var in final['var_names']:
                    assert np.all(np.isfinite(final['mb_data'][var]))
                assert np.min(final['mb_data']['dens']) > 0
                assert np.min(final['mb_data']['eint']) > 0
                hist = np.loadtxt(f'build/src/{name}.mhd.hst')
                user = np.loadtxt(f'build/src/{name}.user.hst')
                # Zero momenta are scaled by total mass times Delta U=1.
                scale = np.array([hist[0, 2]]*4 + [hist[0, 6]])
                drift = (hist[:, 2:7] - hist[0, 2:7])/scale
                assert np.max(np.abs(drift)) < 1e-12
                bscale = np.sqrt(0.01**2 + 0.02**2 + 0.03**2 + target**2)
                assert np.max(np.abs(user[:, 2:5] - user[0, 2:5]))/bscale < 1e-12
                energy = hist[:, 7:10].sum(axis=1) + user[:, 5] + user[:, 6]
                np.testing.assert_allclose(energy, hist[:, 6], rtol=1e-12, atol=0)
        # Identical physical evolution expressed in box versus shear units.
        for suffix in ('00000', '00003'):
            box = bin_convert.read_binary(
                f'build/src/bin/khi_dynamo_box_units.prim.{suffix}.bin')
            shear = bin_convert.read_binary(
                f'build/src/bin/khi_dynamo_spectral_b16.prim.{suffix}.bin')
            np.testing.assert_allclose(box['time']/0.025, shear['time'], rtol=2e-6)
            for var in shear['var_names']:
                factor = 55/18 if var in ('dens', 'eint') else 1.0
                if var.startswith('bcc'):
                    factor = np.sqrt(55/18)
                np.testing.assert_allclose(_assemble(box, var)/factor,
                                           _assemble(shear, var), rtol=2e-6, atol=1e-8)
        return True
    except (AssertionError, OSError, ValueError) as error:
        logger.warning('KHI dynamo setup check failed: %s', error)
        return False
