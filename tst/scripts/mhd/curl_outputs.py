"""Check standard hydro/MHD curl outputs on periodic 1-D, 2-D and 3-D grids.

Reconstruct the entire mesh before taking independent periodic differences, so
MeshBlock interfaces are tested too. Binary fields are stored as float32.
Uses built_in_pgens; no KHI problem or diagnostic code is required.
"""
import logging
import os
from pathlib import Path
import sys

import numpy as np
import scripts.utils.athena as athena

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'vis/python'))
import bin_convert  # noqa: E402

logger = logging.getLogger('athena' + __name__[7:])


def cases():
    for fluid in ('hydro', 'mhd'):
        for dim in (1, 2, 3):
            for block in (16, 8):
                yield fluid, dim, block, f'Curl_{fluid}_{dim}_{block}'


def run(**kwargs):
    for fluid, dim, block, name in cases():
        text = (REPO / f'inputs/tests/linear_wave_{fluid}.athinput').read_text()
        text = text.split('<output1>')[0]
        variables = [fluid + ('_w_bcc' if fluid == 'mhd' else '_w'),
                     fluid + '_wz', fluid + '_w2']
        if fluid == 'mhd':
            variables += ['mhd_jz', 'mhd_j2']
        for n, var in enumerate(variables, 1):
            text += (f'\n<output{n}>\nfile_type = bin\nvariable = {var}\n'
                     f'id = {var}\ndt = 1.0\n')
        inp = Path('build/src/curl_outputs.athinput').resolve()
        inp.write_text(text)
        args = [f'job/basename={name}', 'time/nlim=2', 'time/tlim=1.0',
                'time/integrator=rk3', 'problem/amp=0.05',
                'problem/wave_flag=1', 'problem/vflow=0.2']
        for axis, length in enumerate((3.0, 2.0, 1.0), 1):
            nx = 16 if axis <= dim else 1
            nb = block if axis <= dim else 1
            args += [f'mesh/nx{axis}={nx}', f'meshblock/nx{axis}={nb}',
                     f'mesh/x{axis}min=0.0', f'mesh/x{axis}max={length}']
        athena.run(os.path.relpath(inp, REPO / 'inputs'), args)
    inp.unlink()


def assemble(path):
    data = bin_convert.read_binary(str(path))
    shape = tuple(data[f'Nx{a}'] for a in (3, 2, 1))
    fields = {v: np.empty(shape) for v in data['var_names']}
    for m, (x, y, z, level) in enumerate(data['mb_logical']):
        assert level == 0
        for key in fields:
            value = data['mb_data'][key][m]
            nz, ny, nx = value.shape
            fields[key][z*nz:(z+1)*nz, y*ny:(y+1)*ny, x*nx:(x+1)*nx] = value
    return fields


def curl(vector):
    def derivative(field, axis):
        h = (3.0, 2.0, 1.0)[axis] / field.shape[2-axis]
        return (np.roll(field, -1, 2-axis) - np.roll(field, 1, 2-axis)) / (2*h)
    return np.array([derivative(vector[2], 1)-derivative(vector[1], 2),
                     derivative(vector[0], 2)-derivative(vector[2], 0),
                     derivative(vector[1], 0)-derivative(vector[0], 1)])


def analyze():
    root = Path('build/src/bin')
    try:
        for fluid, dim, block, name in cases():
            primitive = fluid + ('_w_bcc' if fluid == 'mhd' else '_w')
            paths = sorted(root.glob(f'{name}.{primitive}.*.bin'))
            assert len(paths) == 2, (name, 'initial and final output required')
            for path in paths:
                number = path.name.split('.')[-2]
                fields = assemble(path)
                omega = curl([fields['vel'+a] for a in 'xyz'])
                expected = {fluid+'_wz': omega[2], fluid+'_w2': (omega**2).sum(0)}
                if fluid == 'mhd':
                    current = curl([fields[f'bcc{a}'] for a in (1, 2, 3)])
                    expected.update(mhd_jz=current[2], mhd_j2=(current**2).sum(0))
                for product, reference in expected.items():
                    scale = np.max(abs(reference))
                    assert scale > 1e-8, (name, product, 'trivial test field')
                    actual = assemble(root / f'{name}.{product}.{number}.bin')
                    # Float32 primitives lose digits when differencing a background B.
                    np.testing.assert_allclose(next(iter(actual.values())), reference,
                                               rtol=2e-5, atol=2e-5*scale,
                                               err_msg=f'{name}/{product}/{number}')
    except (AssertionError, OSError, ValueError, KeyError) as error:
        logger.warning('Curl output regression failed: %s', error)
        return False
    for path in root.glob('Curl_*.bin'):
        path.unlink()
    return True
