"""KHI diagnostics against independent field reductions and NumPy FFTs.

Build with -DPROBLEM=khi_dynamo and an enabled FFT backend.
"""
import glob
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


def run(**kwargs):
    text = (REPO / 'inputs/mhd/khi_dynamo.athinput').read_text()
    text = text.replace('tlim = 1000.0', 'tlim = 0.2')
    for interval in ('100.0', '1.0', '0.5'):
        text = text.replace('dt = ' + interval, 'dt = 0.1')
    for n, var in enumerate(('mhd_wz', 'mhd_w2', 'mhd_jz', 'mhd_j2'), 12):
        text += (f'\n<output{n}>\nfile_type = bin\nvariable = {var}\n'
                 f'id = {var}\ndt = 0.1\n')
    text += ('\n<output16>\nfile_type = bin\nvariable = mhd_khi\n'
             'id = khi_y0\nslice_x2 = 0.0\ndt = 0.1\n')
    inp = Path('build/src/khi_diag_test.athinput').resolve()
    inp.write_text(text)
    for block in (32, 16):
        name = f'KhiDiagB{block}'
        for pattern in (f'build/src/{name}.*', f'build/src/bin/{name}.*'):
            for filename in glob.glob(pattern):
                os.remove(filename)
        athena.run(os.path.relpath(inp, REPO / 'inputs'), [
            f'job/basename={name}', 'problem/phase_2=0.4', 'problem/sim_vz=0.03',
            f'meshblock/nx1={block}', f'meshblock/nx2={block}', f'meshblock/nx3={block}',
        ])


def table(path, prefix):
    lines = Path(path).read_text().splitlines()
    names = next(line[2:].split() for line in lines if line.startswith('# ' + prefix))
    return dict(zip(names, np.loadtxt(path, ndmin=2).T))


def assemble(data):
    result = {v: np.empty((32, 32, 32)) for v in data['var_names']}
    for m, (x, y, z, level) in enumerate(data['mb_logical']):
        assert level == 0
        for v in result:
            value = data['mb_data'][v][m]
            nz, ny, nx = value.shape
            result[v][z*nz:(z+1)*nz, y*ny:(y+1)*ny, x*nx:(x+1)*nx] = value
    return result


def restart_fields(path):
    """Read this test's double-precision, uniform Newtonian-MHD restart layout."""
    content = Path(path).read_bytes()
    pos = content.index(b'<par_end>\n') + len(b'<par_end>\n')

    def read(dtype, count):
        nonlocal pos
        value = np.frombuffer(content, dtype=dtype, count=count, offset=pos)
        pos += value.nbytes
        return value

    nmb, level = read('i4', 2)
    read('f8', 9)  # RegionSize
    mesh = read('i4', 19)
    block = read('i4', 19)
    read('f8', 2)  # time, dt
    read('i4', 1)  # cycle
    logical = read('i4', int(nmb)*4).reshape(-1, 4)
    read('f4', int(nmb))  # costs
    block_bytes = int(read('u8', 1)[0])
    ng, nx, ny, nz = map(int, block[:4])
    assert tuple(mesh[1:4]) == (32, 32, 32)
    shape = (nz+2*ng, ny+2*ng, nx+2*ng)
    names = ('dens', 'velx', 'vely', 'velz', 'eint', 'bcc1', 'bcc2', 'bcc3')
    result = {v: np.empty((32, 32, 32)) for v in names}
    base = pos
    for m, (x, y, z, lev) in enumerate(logical):
        assert lev == level
        pos = base + m*block_bytes
        u = read('f8', 5*np.prod(shape)).reshape((5,) + shape)
        fields = []
        for axis in (2, 1, 0):
            face_shape = list(shape)
            face_shape[axis] += 1
            face = read('f8', np.prod(face_shape)).reshape(face_shape)
            lo = [slice(None)]*3
            hi = [slice(None)]*3
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            fields.append(.5*(face[tuple(lo)] + face[tuple(hi)]))
        vel = u[1:4]/u[0]
        eint = u[4] - .5*np.sum(u[1:4]*vel, axis=0)
        eint -= .5*np.sum(np.array(fields)**2, axis=0)
        values = [u[0], *vel, eint, *fields]
        for name, value in zip(names, values):
            result[name][z*nz:(z+1)*nz, y*ny:(y+1)*ny, x*nx:(x+1)*nx] = (
                value[ng:ng+nz, ng:ng+ny, ng:ng+nx])
    assert pos == len(content)
    return result


def derivative(a, axis):
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis))/(2*1.25)


def reference(f):
    rho = f['dens']
    u = np.array([f['velx'], f['vely'], f['velz']])
    b = np.array([f['bcc1'], f['bcc2'], f['bcc3']])
    du = np.array([[derivative(u[a], 2-j) for j in range(3)] for a in range(3)])
    db = np.array([[derivative(b[a], 2-j) for j in range(3)] for a in range(3)])
    omega = np.array([du[2, 1]-du[1, 2], du[0, 2]-du[2, 0], du[1, 0]-du[0, 1]])
    current = np.array([db[2, 1]-db[1, 2], db[0, 2]-db[2, 0], db[1, 0]-db[0, 1]])
    div = np.trace(du, axis1=0, axis2=1)
    strain = (du + du.swapaxes(0, 1))/2
    for a in range(3):
        strain[a, a] -= div/3
    q = {'rho': rho, 'Eint': f['eint'], 'pressure': (2/3)*f['eint'],
         'Etot': f['eint'] + .5*(rho*np.sum(u*u, axis=0) + np.sum(b*b, axis=0)),
         'omega2': np.sum(omega*omega, axis=0), 'J2': np.sum(current*current, axis=0),
         'u_dot_omega': np.sum(u*omega, axis=0), 'B_dot_J': np.sum(b*current, axis=0),
         'u_dot_B': np.sum(u*b, axis=0),
         'stretch': np.einsum('i...,j...,ij...->...', b, b, du),
         'compression': -.5*np.sum(b*b, axis=0)*div,
         'ohmic_heat': np.sum(current*current, axis=0)/1024,
         'visc_heat': 2*rho*np.sum(strain*strain, axis=(0, 1))/1024, 'div_u': div}
    for a, axis in enumerate('xyz'):
        q['mom_' + axis] = rho*u[a]
        q['u' + axis] = u[a]
        q['B' + axis] = b[a]
        q['u' + axis + '2'] = u[a]**2
        q['B' + axis + '2'] = b[a]**2
        q['K' + axis] = .5*rho*u[a]**2
        q['omega_' + axis] = omega[a]
        q['J' + axis] = current[a]
    for a, c in ((1, 2), (2, 1), (2, 0), (0, 2), (0, 1), (1, 0)):
        q['u' + 'xyz'[a] + 'B' + 'xyz'[c]] = u[a]*b[c]
    return q


def extras(q):
    ke = sum(q['K' + a] for a in 'xyz')
    me = .5*sum(q['B' + a + '2'] for a in 'xyz')
    mean_me = .5*sum(q['B' + a]**2 for a in 'xyz')
    return {
        'emf_x': q['uyBz']-q['uzBy']-(q['uy']*q['Bz']-q['uz']*q['By']),
        'emf_y': q['uzBx']-q['uxBz']-(q['uz']*q['Bx']-q['ux']*q['Bz']),
        'emf_z': q['uxBy']-q['uyBx']-(q['ux']*q['By']-q['uy']*q['Bx']),
        'Ekin': ke, 'Emag': me, 'Emag_mean': mean_me, 'Emag_fluct': me-mean_me,
        'Kfluct_vol': ke-sum(q['u'+a]*q['mom_'+a] for a in 'xyz')
        + .5*q['rho']*sum(q['u'+a]**2 for a in 'xyz'),
        'Kfluct_Favre': ke-.5*sum(q['mom_'+a]**2 for a in 'xyz')/q['rho'],
    }


def check_case(root, name):
    root = Path(root)
    max_error = 0.0
    for number in (0, 1, 2):
        data = bin_convert.read_binary(str(root / f'bin/{name}.prim.{number:05d}.bin'))
        fields = restart_fields(root / f'rst/{name}.{number:05d}.rst')
        raw = reference(fields)
        profiles = table(root / f'{name}.khi_profiles.{number:05d}.tab', 'region axis')
        for r, region in enumerate(('global', 'lower', 'upper')):
            selection = (slice(None) if r == 0 else
                         (slice(0, 16) if r == 1 else slice(16, 32)))
            local = {v: a[:, :, selection] for v, a in raw.items()}
            means = {v: a.mean() for v, a in local.items()}
            expected = dict(means, **extras(means))
            expected['volume'] = local['rho'].size*1.25**3
            hst = table(root / f'{name}.khi_{region}.hst', 'time dt')
            record = np.argmin(abs(hst['time'] - data['time']))
            np.testing.assert_allclose(hst['time'][record], data['time'], atol=1e-7)
            assert np.max(np.diff(hst['time'])) <= .05+1e-12
            for axis, plane in enumerate(('yz', 'xz', 'xy')):
                avg = {v: a.mean(axis=tuple(d for d in range(3) if d != 2-axis))
                       for v, a in local.items()}
                derived = extras(avg)
                mask = (profiles['region'] == r) & (profiles['axis'] == axis)
                expected_coord = -20 + (np.arange(32) + .5)*1.25
                if axis == 0:
                    expected_coord = expected_coord[selection]
                np.testing.assert_allclose(profiles['coordinate'][mask], expected_coord)
                for key, value in dict(avg, **derived).items():
                    # Double-precision restarts retain the weak correlations.
                    scale = max(float(np.max(abs(raw.get(key, raw['Etot'])))), 1e-12)
                    np.testing.assert_allclose(profiles[key][mask], value,
                                               atol=2e-11*scale + 2e-15,
                                               rtol=2e-10, err_msg=key)
                for c, component in enumerate('xyz', 1):
                    expected[f'Bmean2_{plane}_{c}'] = np.mean(avg['B'+component]**2)
                for key in ('Kfluct_vol', 'Kfluct_Favre'):
                    expected[key+'_'+plane] = derived[key].mean()
            for key, value in expected.items():
                scale = max(float(np.max(abs(raw.get(key, raw['Etot'])))), 1e-12)
                error = abs(hst[key][record]-value)/scale
                max_error = max(max_error, error)
                np.testing.assert_allclose(hst[key][record], value,
                                           atol=2e-11*scale + 2e-15,
                                               rtol=2e-10, err_msg=key)
        # Independent complete Fourier transform and spherical shell binning.
        k = np.fft.fftfreq(32)*32
        kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
        shell = np.floor(np.sqrt(kx*kx+ky*ky+kz*kz)).astype(int)
        for product, variables in (('velocity', ('velx', 'vely', 'velz')),
                                   ('magnetic', ('bcc1', 'bcc2', 'bcc3'))):
            power = sum(abs(np.fft.fftn(fields[v])/32**3)**2 for v in variables)
            expected = np.bincount(shell.ravel(), weights=power.ravel())[1:17]
            actual = np.loadtxt(root / f'{name}.{product}.{number:05d}.spec')[:, 1]
            np.testing.assert_allclose(actual, expected, rtol=2e-10,
                                       atol=2e-12*max(expected), err_msg=product)
        # Slices use the native cell containing the requested plane, not interpolation.
        mapping = {'rho': 'rho', 'eint': 'Eint'}
        for product, axis, index in (('khi_z0', 0, 16), ('khi_xlower', 2, 8),
                                     ('khi_xupper', 2, 24), ('khi_y0', 1, 16)):
            path = root / f'bin/{name}.{product}.{number:05d}.bin'
            d = bin_convert.read_binary(str(path))
            for m, logical in enumerate(d['mb_logical']):
                x, y, z = logical[:3]
                # Checks use 16-cell blocks (the 32-cell case has a single block).
                n = 32 if data['n_mbs'] == 1 else 16
                location = [slice(z*n, (z+1)*n), slice(y*n, (y+1)*n),
                            slice(x*n, (x+1)*n)]
                location[axis] = slice(index, index+1)
                for key in d['var_names']:
                    value = raw[mapping.get(key, key)][tuple(location)]
                    scale = max(float(np.max(abs(raw[mapping.get(key, key)]))), 1e-12)
                    np.testing.assert_allclose(d['mb_data'][key][m], value,
                                               rtol=3e-6, atol=2e-6*scale, err_msg=key)
        for product, key in (('mhd_wz', 'omega_z'), ('mhd_w2', 'omega2'),
                             ('mhd_jz', 'Jz'), ('mhd_j2', 'J2')):
            path = root / f'bin/{name}.{product}.{number:05d}.bin'
            d = assemble(bin_convert.read_binary(str(path)))
            np.testing.assert_allclose(next(iter(d.values())), raw[key],
                                       rtol=3e-6,
                                       atol=2e-6*max(abs(raw[key]).max(), 1e-12))
    return max_error


def analyze():
    try:
        for block in (32, 16):
            check_case('build/src', f'KhiDiagB{block}')
        return True
    except (AssertionError, OSError, ValueError, KeyError) as error:
        logger.warning('KHI diagnostics failed: %s', error)
        return False
