"""Isothermal SSD initialization; build -DPROBLEM=newtonian_SSD.

Tests fresh seeds and zero-field driven spin-up followed by one-time restart
seeding, including exact preservation of the saved flow and forcing data.
"""
import logging
import os
from pathlib import Path
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
    text = text.replace('<mhd>', '<mhd>\nnscalars = 1')
    text = text.replace('eos = isothermal', 'eos = isothermal\ngamma = 1.6666666667')
    text += ('\n<output2>\nfile_type = bin\nvariable = mhd_w_bcc\n'
             'id = prim\ndt = 1.0\n'
             '\n<output3>\nfile_type = bin\nvariable = mhd_divb\n'
             'id = divb\ndt = 1.0\n')
    path.write_text(text)
    input_path = os.path.relpath(path, REPO/'inputs')
    for name, flag in (('force_free', 'true'), ('gaussian', 'false')):
        athena.run(input_path, [f'job/basename=NewtonianSSDTest_{name}',
                               f'spectral_ic/force_free={flag}',
                               'mhd/nscalars=1', 'time/nlim=0'])
    athena.run(input_path, ['job/basename=NewtonianSSDTest_evolved',
                           'mhd/nscalars=1', 'time/nlim=4'])
    for override in ('mhd/eos=ideal', 'mhd/iso_sound_speed=0',
                     'spectral_ic/rms_b=-1', 'spectral_ic/rms_b=nan',
                     'spectral_ic/rms_b=inf', 'spectral_ic/seed_on_restart=true'):
        result = subprocess.run(['./athena', '-i', str(path), override, 'time/nlim=0'],
                                cwd='build/src', capture_output=True, text=True)
        assert result.returncode != 0, override
        assert ('newtonian_SSD requires' in result.stderr
                or 'Invalid spectral seed' in result.stderr), result.stderr

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
    for checkpoint, overrides, message in (
            (spinup, ['spectral_ic/seed_on_restart=true'], 'positive'),
            (injected, ['spectral_ic/seed_on_restart=true'], 'zero magnetic field')):
        result = subprocess.run(['./athena', '-r', str(checkpoint.resolve()),
                                 *overrides, 'time/nlim=0'], cwd='build/src',
                                capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr, result.stderr
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
    assert data_size == (5*nc+nf+3*nc)*8
    assert len(blob) == offset+nmb*data_size
    payload = np.frombuffer(blob, dtype='=f8', offset=offset).reshape(nmb, -1)
    u = payload[:, :5*nc].reshape(nmb, 5, *shape)
    force = payload[:, 5*nc+nf:].reshape(nmb, 3, *shape)
    active = (slice(None), slice(None), slice(ng, ng+nz),
              slice(ng, ng+ny), slice(ng, ng+nx))
    return dict(header=header, time=time, cycle=cycle, rng=rng, u=u[active],
                force=force[active], faces=payload[:, 5*nc:5*nc+nf])


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
                   'continued_preserved')}
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
        metrics['restart'] = dict(flow_preserved_exactly=True,
                                  saved_forcing_preserved=True,
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
