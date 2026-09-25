"""Isothermal SSD initialization; build -DPROBLEM=newtonian_SSD.

Tests fresh seeds and zero-field driven spin-up followed by one-time restart
seeding, including exact preservation of the saved flow and forcing data.
"""
import logging
import os
from pathlib import Path
import re
import subprocess
import struct

import numpy as np
import scripts.utils.athena as athena
from scripts.mhd.force_free_ssd import assemble

logger = logging.getLogger('athena' + __name__[7:])
REPO = Path(__file__).resolve().parents[3]
INPUT = 'SSD_extreme_scale_sep/force_free_k10_32.athinput'
metrics = {}


def run(**kwargs):
    # Extra field diagnostics belong only to this regression, not the setup run.
    path = Path('build/src/NewtonianSSDTest.athinput').resolve()
    text = (REPO/'inputs'/INPUT).read_text()
    text = text.replace('fft_backend = kokkos\n', '')  # use the compiled backend
    text = text.replace('<mhd>', '<mhd>\nnscalars = 1')
    text = text.replace('eos = isothermal', 'eos = isothermal\ngamma = 1.6666666667')
    text = text.replace('id = magnetic', 'id = magnetic\nhistory_curl_peak = k_eta')
    text += ('\n<output2>\nfile_type = bin\nvariable = mhd_w_bcc\n'
             'id = prim\ndt = 1.0\n'
             '\n<output3>\nfile_type = bin\nvariable = mhd_divb\n'
             'id = divb\ndt = 1.0\n'
             '\n<problem>\nuser_hist = true\n'
             '\n<output5>\nfile_type = hst\ndt = 1.0\ndata_format = %24.16e\n'
             '\n<output6>\nfile_type = power_spectrum\nvariable = velocity\n'
             'id = velocity\ndt = 1.0\nhistory_curl_peak = k_nu\n')
    path.write_text(text)
    input_path = os.path.relpath(path, REPO/'inputs')
    for name, flag in (('force_free', 'true'), ('gaussian', 'false')):
        athena.run(input_path, [f'job/basename=NewtonianSSDTest_{name}',
                               f'spectral_ic/force_free={flag}',
                               'mhd/nscalars=1', 'time/nlim=0'])
    path.write_text(text.replace('history_curl_peak = k_eta', '')
                   .replace('history_curl_peak = k_nu', ''))
    athena.run(input_path, ['job/basename=NewtonianSSDTest_no_peaks', 'time/nlim=0'])
    path.write_text(text)
    athena.run(input_path, ['job/basename=NewtonianSSDTest_evolved',
                           'mhd/nscalars=1', 'time/nlim=4'])
    # Axial shell k=2 has zero normal magnetic derivatives. Its history can be
    # checked independently from the assembled cell-centered binary field.
    for name, flag in (('weak_ff', 'true'), ('weak_gaussian', 'false')):
        athena.run(input_path, [f'job/basename=NewtonianSSDTest_{name}',
                               f'spectral_ic/force_free={flag}',
                               'spectral_ic/nlow=2', 'spectral_ic/nhigh=2',
                               'spectral_ic/rms_b=1e-6', 'time/nlim=0'])
    athena.run(input_path, ['job/basename=NewtonianSSDTest_large_box',
                           'mesh/x1max=2', 'mesh/x2max=2', 'mesh/x3max=2',
                           'spectral_ic/nlow=2', 'spectral_ic/nhigh=2', 'time/nlim=0'])
    for override in ('mhd/eos=ideal', 'mhd/iso_sound_speed=0',
                     'spectral_ic/rms_b=-1', 'spectral_ic/rms_b=nan',
                     'spectral_ic/rms_b=inf', 'spectral_ic/seed_on_restart=true'):
        result = subprocess.run(['./athena', '-i', str(path), override, 'time/nlim=0'],
                                cwd='build/src', capture_output=True, text=True)
        assert result.returncode != 0, override
        assert ('newtonian_SSD requires' in result.stderr
                or 'Invalid spectral seed' in result.stderr), result.stderr
    for override, message in (('output1/variable=density', 'velocity or magnetic'),
                               ('output5/dt=0', 'hst output'),
                               ('output6/history_curl_peak=k_eta', 'Duplicate')):
        result = subprocess.run(['./athena', '-i', str(path), override, 'time/nlim=0',
                                 'job/basename=NewtonianSSDTest_invalid'],
                                cwd='build/src', capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr, result.stderr

    # A short driven run tests the workflow, not fully developed turbulence.
    text += ('\n<turb_driving>\ndriving_geometry = isotropic\n'
             'driving_profile = band\nnlow = 1\nnhigh = 2\n'
             'dedt = 0.1\ntcorr = 0.1\nsol_weight = 1.0\n'
             '\n<output4>\nfile_type = rst\ndt = 1.0\n')
    path.write_text(text)
    athena.run(input_path, ['job/basename=NewtonianSSDTest_spinup',
                           'spectral_ic/rms_b=0', 'time/nlim=8',
                           'spectral_ic/nlow=2', 'spectral_ic/nhigh=2'])
    spinup = latest_restart('spinup')
    # Match an uninterrupted trajectory after resuming the correlated forcing.
    athena.run(input_path, ['job/basename=NewtonianSSDTest_uninterrupted',
                           'spectral_ic/rms_b=0', 'time/nlim=12',
                           'spectral_ic/nlow=2', 'spectral_ic/nhigh=2'])
    restart(spinup, 'split', ['time/nlim=12'])
    # Old checkpoints remain readable, but explicitly lack OU correlation history.
    legacy = legacy_checkpoint(spinup)
    restart(legacy, 'legacy', [])
    legacy.unlink()
    for name, overrides in (
            ('injected', ['spectral_ic/seed_on_restart=true', 'spectral_ic/rms_b=0.02']),
            ('unseeded', ['spectral_ic/rms_b=0.02'])):
        restart(spinup, name, overrides)
    injected = latest_restart('injected')
    # The consumed flag must not be needed again, and changed seed controls must
    # not alter the evolved field on an ordinary restart.
    restart(injected, 'preserved', ['spectral_ic/rms_b=0.5',
                                  'spectral_ic/iseed=210990'])
    restart(injected, 'continued', ['time/nlim=12'])
    restart(latest_restart('continued'), 'continued_preserved', [])
    for name in ('longitudinal', 'mixed_modes', 'nyquist'):
        manufactured_velocity(spinup, name)
    for checkpoint, overrides, message in (
            (spinup, ['spectral_ic/seed_on_restart=true'], 'positive'),
            (injected, ['spectral_ic/seed_on_restart=true'], 'zero magnetic field')):
        result = subprocess.run(['./athena', '-r', str(checkpoint.resolve()),
                                 *overrides, 'time/nlim=0'], cwd='build/src',
                                capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr, result.stderr
    # Distinct correlated components must each survive restart, not just their sum.
    text += '\n<turb_driving>\nnum_components = 2\n'
    for c in range(2):
        text += (f'\n<turb_driving/component{c}>\n'
                 f'nlow = {c+1}\nnhigh = {c+2}\n'
                 'driving_profile = band\n'
                 f'dedt = {0.03*(c+1)}\ntcorr = {0.1*(c+1)}\nsol_weight = 1\n')
    path.write_text(text)
    for name, cycles in (('multi_full', 12), ('multi_first', 8)):
        athena.run(input_path, [f'job/basename=NewtonianSSDTest_{name}',
                               'spectral_ic/rms_b=0', f'time/nlim={cycles}'])
    restart(latest_restart('multi_first'), 'multi_split', ['time/nlim=12'])
    path.unlink()


def latest_restart(name):
    return sorted(Path('build/src/rst').glob(f'NewtonianSSDTest_{name}.*.rst'))[-1]


def restart(checkpoint, name, overrides):
    subprocess.run(['./athena', '-r', str(checkpoint.resolve()),
                    f'job/basename=NewtonianSSDTest_{name}', 'time/nlim=0', *overrides],
                   cwd='build/src', check=True)


def read_checkpoint(path):
    """Read this test's double-precision 3-D MHD + scalar + forcing checkpoint."""
    blob = path.read_bytes()
    offset = blob.index(b'<par_end>\n') + len(b'<par_end>\n')
    header = blob[:offset].decode()
    nmb, = struct.unpack_from('=i', blob, offset)
    # Two ints, RegionSize (9 Real), and two RegionIndcs (19 int each).
    block = struct.unpack_from('=19i', blob, offset+8+9*8+19*4)
    ng, nx, ny, nz = block[:4]
    offset += 8+9*8+2*19*4
    time, _, cycle = struct.unpack_from('=ddi', blob, offset)
    offset += 20+nmb*(4*4+4)  # time/dt/cycle, logical locations and costs
    # RNG_State: 35 int64, one int, padding to double alignment, one double.
    rng = blob[offset:offset+struct.calcsize('@35qid')]
    offset += struct.calcsize('@35qid')
    data_size, = struct.unpack_from('=Q', blob, offset)
    offset += 8
    shape = (nz+2*ng, ny+2*ng, nx+2*ng)
    nc = int(np.prod(shape))
    nf = sum((n+1)*nc//n for n in shape)
    count = re.search(r'^restart_num_components\s*=\s*(\d+)', header, re.MULTILINE)
    nforce = 3*(1+int(count[1])) if count else 3
    assert data_size == (5*nc+nf+nforce*nc)*8
    assert len(blob) == offset+nmb*data_size
    payload = np.frombuffer(blob, dtype='=f8', offset=offset).reshape(nmb, -1)
    u = payload[:, :5*nc].reshape(nmb, 5, *shape)
    force = payload[:, 5*nc+nf:].reshape(nmb, nforce, *shape)
    active = (slice(None), slice(None), slice(ng, ng+nz),
              slice(ng, ng+ny), slice(ng, ng+nx))
    return dict(header=header, time=time, cycle=cycle, rng=rng, u=u[active],
                force=force[active], faces=payload[:, 5*nc:5*nc+nf])


def legacy_checkpoint(path):
    """Drop OU component arrays to exercise the pre-extension restart format."""
    blob = path.read_bytes()
    offset = blob.index(b'<par_end>\n')+len(b'<par_end>\n')
    header = re.sub(rb'^restart_num_components[^\n]*\n', b'', blob[:offset],
                    flags=re.MULTILINE)
    nmb, = struct.unpack_from('=i', blob, offset)
    ng, nx, ny, nz = struct.unpack_from('=4i', blob, offset+8+9*8+19*4)
    start = offset+8+9*8+2*19*4+20+20*nmb+struct.calcsize('@35qid')
    size, = struct.unpack_from('=Q', blob, start)
    nc = (nx+2*ng)*(ny+2*ng)*(nz+2*ng)
    new_size = size-3*nc*8
    payload = b''.join(blob[start+8+m*size:start+8+m*size+new_size]
                       for m in range(nmb))
    legacy = path.with_name('NewtonianSSDTest_legacy_input.rst')
    legacy.write_bytes(header+blob[offset:start]+struct.pack('=Q', new_size)+payload)
    return legacy


def magnetic_history(name):
    path = Path(f'build/src/NewtonianSSDTest_{name}.user.hst')
    header = path.read_text().splitlines()[1]
    assert all(key in header for key in ('k_parallel', 'k_BxJ', 'k_BdotJ'))
    data = np.loadtxt(path, ndmin=2)
    assert data.shape[1] == 7 and np.all(np.isfinite(data))
    assert np.all(data[:, 2:] >= 0.0)
    return data[:, 2:5]


def manufactured_velocity(checkpoint, name):
    """Set analytic velocities in a disposable copy of the spin-up checkpoint."""
    blob = bytearray(checkpoint.read_bytes())
    offset = blob.index(b'<par_end>\n')+len(b'<par_end>\n')
    nmb, = struct.unpack_from('=i', blob, offset)
    ng, nx, ny, nz = struct.unpack_from('=4i', blob, offset+8+9*8+19*4)
    offset += 8+9*8+2*19*4+20
    locations = np.frombuffer(blob, dtype='=i4', count=4*nmb, offset=offset)
    locations = locations.reshape(nmb, 4)
    offset += 20*nmb+struct.calcsize('@35qid')+8
    shape = (nz+2*ng, ny+2*ng, nx+2*ng)
    nc = int(np.prod(shape))
    payload = np.frombuffer(blob, dtype='=f8', offset=offset).reshape(nmb, -1)
    fluid = payload[:, :5*nc].reshape(nmb, 5, *shape)
    for m, loc in enumerate(locations):
        x, y, z = [(loc[a]*n+np.arange(n+2*ng)-ng+0.5)/32
                   for a, n in enumerate((nx, ny, nz))]
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        velocity = np.zeros((3, *shape))
        if name == 'longitudinal':
            velocity[0] = np.sin(2*np.pi*4*xx)
        elif name == 'nyquist':
            velocity[1] = (-1.0)**(loc[0]*nx+np.arange(nx+2*ng)-ng)
        else:
            # Strong compressive shell 5; weaker transverse shell 3. All curl
            # components are nonzero. Total velocity and curl peak differently.
            longitudinal = np.sin(2*np.pi*(3*xx+4*yy)+0.3)
            transverse = np.cos(2*np.pi*(xx+2*yy+2*zz)-0.7)
            velocity[0] = 3*longitudinal+0.06*transverse
            velocity[1] = 4*longitudinal-0.03*transverse
        fluid[m, 0] = 1.0
        fluid[m, 1:4] = velocity
    path = Path(f'build/src/NewtonianSSDTest_{name}.rst')
    path.write_bytes(blob)
    restart(path, name, [])
    path.unlink()


def spectral_history(name, length=1.0):
    """Compare the full curl spectra with independent FFTs and every hst peak."""
    root = Path('build/src')
    prefix = f'NewtonianSSDTest_{name}'
    history_path = root/f'{prefix}.user.hst'
    labels = [item.split('=')[1] for item in history_path.read_text().splitlines()[1]
              .split() if '=' in item]
    history = np.loadtxt(history_path, ndmin=2)
    for variable, label, components in (('magnetic', 'k_eta', ('bcc1', 'bcc2', 'bcc3')),
                                         ('velocity', 'k_nu', ('velx', 'vely', 'velz'))):
        peaks = {}
        paths = sorted(root.glob(f'{prefix}.{variable}.*.spec'))
        for path in paths:
            timestamp = float(path.read_text().splitlines()[0].split()[1].split('=')[1])
            data = np.loadtxt(path)
            assert data.shape[1] == 3 and np.all(data[:, 1:] >= 0.0)
            peaks[timestamp] = (data[np.argmax(data[:, 2]), 0]*2*np.pi/length
                                if np.any(data[:, 2]) else 0.0)
        for row in history:
            np.testing.assert_allclose(row[labels.index(label)], peaks[row[0]],
                                       rtol=2e-14)
        # Last spectrum and binary both contain the final state, including restarts.
        fields = assemble(sorted((root/'bin').glob(f'{prefix}.prim.*.bin'))[-1])
        field = np.array([fields[c] for c in components])
        nz, ny, nx = field.shape[1:]
        transform = np.fft.fftn(field, axes=(1, 2, 3))/(nx*ny*nz)
        nzs, nys, nxs = [np.fft.fftfreq(n)*n for n in (nz, ny, nx)]
        kz, ky, kx = np.meshgrid(nzs, nys, nxs, indexing='ij')
        shells = np.floor(np.sqrt(kx*kx+ky*ky+kz*kz)).astype(int)
        wave = np.array([kx, ky, kz])*2*np.pi/length
        for a, n in enumerate((nx, ny, nz)):
            wave[a][abs((kx, ky, kz)[a]) == n/2] = 0.0
        curl = np.cross(wave, transform, axisa=0, axisb=0, axisc=0)
        power = np.sum(abs(curl)**2, axis=0)
        expected = np.bincount(shells.ravel(), weights=power.ravel())
        # Binary dumps have float32 fields; negligible FFT noise is expected.
        scale = max(float(np.sum(expected)), 1e-24)
        np.testing.assert_allclose(data[:, 2], expected[1:len(data)+1],
                                   rtol=3e-6, atol=3e-6*scale)
    return dict(zip(labels, history[-1]))


def reference_magnetic_scales(fields):
    """Independent periodic differences for this test's axial k=2 fields."""
    b = np.array([fields[f'bcc{a}'] for a in (1, 2, 3)])
    grad = np.array([[(np.roll(component, -1, 2-d)-np.roll(component, 1, 2-d))
                       * component.shape[2-d]/2 for d in range(3)] for component in b])
    tension = np.einsum('dzyx,adzyx->azyx', b, grad)
    current = np.array([grad[2, 1]-grad[1, 2], grad[0, 2]-grad[2, 0],
                        grad[1, 0]-grad[0, 1]])
    cross = np.cross(b, current, axisa=0, axisb=0, axisc=0)
    dot = np.sum(b*current, axis=0)
    b4 = np.mean(np.sum(b*b, axis=0)**2)
    return np.sqrt(np.array([np.mean(np.sum(tension*tension, axis=0)),
                              np.mean(np.sum(cross*cross, axis=0)),
                              np.mean(dot*dot)])/b4)


def analyze():
    root = Path('build/src')
    try:
        seeds = {}
        for name in ('force_free', 'gaussian'):
            prefix = 'NewtonianSSDTest_'+name
            fields = assemble(root/f'bin/{prefix}.prim.00000.bin')
            assert 'eint' not in fields
            np.testing.assert_array_equal(fields['dens'], 1.0)
            for key, values in fields.items():
                if key.startswith(('vel', 's_')):
                    np.testing.assert_array_equal(values, 0.0)
            assert any(key.startswith('s_') for key in fields)
            b = np.array([fields[f'bcc{axis}'] for axis in (1, 2, 3)])
            seeds[name] = b
            b2 = np.mean(np.sum(b*b, axis=0))
            np.testing.assert_allclose(b2, 0.01**2, rtol=1e-7)
            div = assemble(root/f'bin/{prefix}.divb.00000.bin')
            max_div = max(np.max(abs(a)) for a in div.values())
            assert max_div < 1e-12
            spectrum = np.loadtxt(root/f'{prefix}.magnetic.00000.spec')
            np.testing.assert_allclose(spectrum[:, 1].sum(), b2, rtol=1e-7)
            outside = spectrum[spectrum[:, 0] != 10, 1].sum()/spectrum[:, 1].sum()
            assert outside < 1e-25
            metrics[name] = dict(rms_b=float(np.sqrt(b2)), max_div=float(max_div),
                                 outside_shell=float(outside))
        assert np.linalg.norm(seeds['force_free']-seeds['gaussian']) > 0.1
        old_format = np.loadtxt(root/'NewtonianSSDTest_no_peaks.magnetic.00000.spec')
        enabled = np.loadtxt(root/'NewtonianSSDTest_force_free.magnetic.00000.spec')
        assert old_format.shape[1] == 2
        np.testing.assert_allclose(old_format, enabled[:, :2], rtol=2e-14, atol=1e-30)
        paths = sorted((root/'bin').glob('NewtonianSSDTest_evolved.prim.*.bin'))
        assert len(paths) == 2
        final = assemble(paths[-1])
        assert all(np.all(np.isfinite(a)) for a in final.values())
        assert np.min(final['dens']) > 0.0
        for key, values in final.items():
            if key.startswith('s_'):
                np.testing.assert_array_equal(values, 0.0)

        states = {name: read_checkpoint(latest_restart(name)) for name in
                  ('spinup', 'injected', 'unseeded', 'preserved', 'continued',
                   'continued_preserved', 'uninterrupted', 'split', 'legacy',
                   'multi_full', 'multi_split')}
        for full, split in (('uninterrupted', 'split'), ('multi_full', 'multi_split')):
            for key in ('time', 'cycle'):
                assert states[full][key] == states[split][key], key
            # Compare the uniform generator's state; trailing Gaussian fields
            # and struct padding are unused by this driver.
            assert states[full]['rng'][:35*8] == states[split]['rng'][:35*8]
            for key in ('u', 'force', 'faces'):
                np.testing.assert_array_equal(states[full][key], states[split][key])
        for key in ('u', 'faces'):
            np.testing.assert_array_equal(states['legacy'][key], states['spinup'][key])
        np.testing.assert_array_equal(states['legacy']['force'][:, :3],
                                      states['spinup']['force'][:, :3])
        np.testing.assert_array_equal(states['legacy']['force'][:, 3:], 0.0)
        spinup, injected = states['spinup'], states['injected']
        assert np.std(spinup['u'][:, 0]) > 1e-6
        assert np.max(abs(spinup['u'][:, 1:4])) > 1e-3
        assert np.max(abs(spinup['force'])) > 0.0
        np.testing.assert_array_equal(spinup['faces'], 0.0)
        np.testing.assert_array_equal(states['unseeded']['faces'], 0.0)
        for before, after in ((spinup, injected), (spinup, states['unseeded']),
                              (injected, states['preserved']),
                              (states['continued'], states['continued_preserved'])):
            assert before['time'] == after['time'] and before['cycle'] == after['cycle']
            np.testing.assert_array_equal(before['u'], after['u'])
            np.testing.assert_array_equal(before['force'], after['force'])
            assert before['rng'] == after['rng']
        for before, after in ((injected, states['preserved']),
                              (states['continued'], states['continued_preserved'])):
            np.testing.assert_array_equal(before['faces'], after['faces'])
        seed_block = injected['header'].split('<spectral_ic>')[1].split('<')[0]
        flag = next(line for line in seed_block.splitlines() if 'seed_on_restart' in line)
        assert flag.split('=')[1].split('#')[0].strip() in ('false', '0')
        spectrum_path = sorted(root.glob('NewtonianSSDTest_injected.magnetic.*.spec'))[-1]
        spectrum = np.loadtxt(spectrum_path)
        np.testing.assert_allclose(spectrum[:, 1].sum(), 0.02**2, rtol=1e-12)
        outside = spectrum[spectrum[:, 0] != 2, 1].sum()/spectrum[:, 1].sum()
        assert outside < 1e-25
        div_path = sorted((root/'bin').glob('NewtonianSSDTest_injected.divb.*.bin'))[-1]
        max_div = max(np.max(abs(a)) for a in assemble(div_path).values())
        assert max_div < 1e-12
        assert states['continued']['cycle'] == 12
        assert states['continued']['time'] > injected['time']
        assert np.all(np.isfinite(states['continued']['u']))
        assert np.min(states['continued']['u'][:, 0]) > 0.0
        assert np.linalg.norm(states['continued']['faces']-injected['faces']) > 1e-5
        for name in ('spinup', 'unseeded'):
            np.testing.assert_array_equal(magnetic_history(name), 0.0)
        for name in ('injected', 'weak_ff', 'weak_gaussian'):
            paths = sorted((root/'bin').glob(f'NewtonianSSDTest_{name}.prim.*.bin'))
            expected = reference_magnetic_scales(assemble(paths[-1]))
            np.testing.assert_allclose(magnetic_history(name)[-1], expected,
                                       rtol=2e-6, atol=2e-5)
        # For the axial force-free shell, centered curl has this exact eigenvalue.
        curl_eigenvalue = 32*np.sin(2*np.pi*2/32)
        ff_scales = magnetic_history('injected')[-1]
        assert ff_scales[0] > 0.0 and ff_scales[1] < 1e-10
        np.testing.assert_allclose(ff_scales[2], curl_eigenvalue, rtol=2e-12)
        np.testing.assert_allclose(magnetic_history('weak_ff')[-1], ff_scales,
                                   rtol=2e-12, atol=1e-11)
        assert magnetic_history('weak_gaussian')[-1, 1] > 1e-3
        for before, after in (('injected', 'preserved'),
                              ('continued', 'continued_preserved')):
            np.testing.assert_allclose(magnetic_history(before)[-1],
                                       magnetic_history(after)[-1],
                                       rtol=2e-12, atol=1e-11)
        metrics['magnetic_scales'] = dict(k_parallel=float(ff_scales[0]),
                                          k_BxJ=float(ff_scales[1]),
                                          k_BdotJ=float(ff_scales[2]),
                                          weak_seed_amplitude_independent=True)
        peak_metrics = {}
        for name in ('force_free', 'gaussian', 'evolved', 'weak_ff', 'weak_gaussian',
                     'spinup', 'injected', 'preserved', 'continued',
                     'continued_preserved', 'longitudinal', 'mixed_modes', 'nyquist'):
            peak_metrics[name] = spectral_history(name)
        np.testing.assert_allclose(peak_metrics['weak_ff']['k_eta'], 4*np.pi)
        np.testing.assert_allclose(spectral_history('large_box', 2.0)['k_eta'], 2*np.pi)
        for name in ('force_free', 'longitudinal', 'nyquist'):
            assert peak_metrics[name]['k_nu'] == 0.0
        assert peak_metrics['spinup']['k_eta'] == 0.0
        np.testing.assert_allclose(peak_metrics['mixed_modes']['k_nu'], 6*np.pi)
        mixed_path = sorted(root.glob('NewtonianSSDTest_mixed_modes.velocity.*.spec'))[-1]
        mixed = np.loadtxt(mixed_path)
        assert mixed[np.argmax(mixed[:, 1]), 0] == 5
        # Analytic transverse mode amplitude squared is .06^2+.03^2, |n|=3.
        np.testing.assert_allclose(mixed[2, 2], 0.5*(0.06**2+0.03**2)*(6*np.pi)**2,
                                   rtol=1e-12)
        metrics['spectral_peaks'] = peak_metrics
        metrics['restart'] = dict(flow_preserved_exactly=True,
                                  saved_forcing_preserved=True,
                                  uninterrupted_matches_split_exactly=True,
                                  multiple_components_and_legacy_checked=True,
                                  rms_b=0.02, max_div=float(max_div),
                                  outside_shell=float(outside))
        logger.info('Isothermal SSD validation: %s', metrics)
    except (AssertionError, OSError, ValueError, KeyError) as error:
        logger.warning('Isothermal SSD regression failed: %s', error, exc_info=True)
        return False
    for directory in (root, root/'bin', root/'rst'):
        for path in directory.glob('NewtonianSSDTest*'):
            if path.is_file():
                path.unlink()
    return True
