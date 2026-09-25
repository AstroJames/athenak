"""Correlated forcing must follow the same trajectory across a restart.

Use built_in_pgens and existing linear-wave inputs for both hydro and MHD,
with one or two forcing components and no passive scalars.
"""
import logging
import os
from pathlib import Path
import re
import struct
import subprocess

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])
REPO = Path(__file__).resolve().parents[3]
PREFIX = 'ForcingRestart'


def latest(name):
    return sorted(Path('build/src/rst').glob(f'{PREFIX}_{name}.*.rst'))[-1]


def run(**kwargs):
    path = Path(f'build/src/{PREFIX}.athinput').resolve()
    for fluid in ('hydro', 'mhd'):
        text = (REPO/f'inputs/tests/linear_wave_{fluid}.athinput').read_text()
        text = text.split('<output1>')[0]
        text = text.replace(f'<{fluid}>', f'<{fluid}>\niso_sound_speed = 1\nnscalars = 0')
        text += '\n<output1>\nfile_type = rst\ndt = 1.0\n'
        for components in (1, 2):
            driving = f'\n<turb_driving>\nnum_components = {components}\n'
            for c in range(components):
                block = ('turb_driving' if components == 1
                         else f'turb_driving/component{c}')
                driving += (f'\n<{block}>\ndriving_geometry = isotropic\n'
                            'driving_profile = parabola\nparabola_peak = 2\n'
                            'parabola_width = 0.5\nnlow = 1\nnhigh = 3\n'
                            f'tcorr = {0.1*(c+1)}\ndedt = {0.03*(c+1)}\n'
                            'sol_weight = 1\n')
            path.write_text(text+driving)
            name = f'{fluid}_{components}'
            common = [f'{fluid}/eos=isothermal', f'{fluid}/iso_sound_speed=1',
                      f'{fluid}/nscalars=0', 'mesh/nghost=4', 'time/integrator=rk3',
                      'time/tlim=1', 'problem/amp=0.01', 'problem/vflow=0.2']
            for axis in (1, 2, 3):
                common += [f'mesh/nx{axis}=16', f'meshblock/nx{axis}=8']
            for part, cycles in (('full', 11), ('first', 5)):
                athena.run(os.path.relpath(path, REPO/'inputs'),
                           [f'job/basename={PREFIX}_{name}_{part}',
                            f'time/nlim={cycles}', *common])
            subprocess.run(['./athena', '-r', str(latest(name+'_first').resolve()),
                            f'job/basename={PREFIX}_{name}_split', 'time/nlim=11'],
                           cwd='build/src', check=True)
    path.unlink()


def state(path, fluid):
    blob = path.read_bytes()
    start = blob.index(b'<par_end>\n')+len(b'<par_end>\n')
    header = blob[:start].decode()
    nmb, = struct.unpack_from('=i', blob, start)
    ng, nx, ny, nz = struct.unpack_from('=4i', blob, start+8+9*8+19*4)
    offset = start+8+9*8+2*19*4
    time, _, cycle = struct.unpack_from('=ddi', blob, offset)
    offset += 20+20*nmb
    rng = blob[offset:offset+35*8]  # uniform RNG fields; omit padding/unused fields
    offset += struct.calcsize('@35qid')
    size, = struct.unpack_from('=Q', blob, offset)
    offset += 8
    assert len(blob) == offset+nmb*size
    shape = (nz+2*ng, ny+2*ng, nx+2*ng)
    nc = int(np.prod(shape))
    nf = sum(nc*(n+1)//n for n in shape) if fluid == 'mhd' else 0
    count = re.search(r'^restart_num_components\s*=\s*(\d+)', header, re.MULTILINE)
    nforce = 3*(1+int(count[1])) if count else 3
    assert size == (4*nc+nf+nforce*nc)*8
    payload = np.frombuffer(blob, dtype='=f8', offset=offset).reshape(nmb, -1)
    active = (slice(None), slice(None), slice(ng, ng+nz),
              slice(ng, ng+ny), slice(ng, ng+nx))
    fields = payload[:, :4*nc].reshape(nmb, 4, *shape)[active]
    force = payload[:, 4*nc+nf:].reshape(nmb, nforce, *shape)[active]
    assert np.all(np.isfinite(fields)) and np.all(np.isfinite(force))
    return dict(time=time, cycle=cycle, rng=rng, fields=fields, force=force,
                faces=payload[:, 4*nc:4*nc+nf])


def analyze():
    try:
        for fluid in ('hydro', 'mhd'):
            for components in (1, 2):
                name = f'{fluid}_{components}'
                full, split = [state(latest(name+'_'+part), fluid)
                               for part in ('full', 'split')]
                assert full['cycle'] == split['cycle'] == 11
                assert full['time'] == split['time'] > 0
                assert full['rng'] == split['rng']
                for field in ('fields', 'force', 'faces'):
                    np.testing.assert_array_equal(full[field], split[field],
                                                  err_msg=f'{name}: {field}')
                assert np.max(abs(full['force'])) > 0
    except (AssertionError, OSError, ValueError) as error:
        logger.warning('Forcing restart failed: %s', error, exc_info=True)
        return False
    for path in Path('build/src/rst').glob(PREFIX+'*'):
        path.unlink()
    return True
