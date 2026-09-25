"""Single-shell Gaussian force-free ICs; build -DPROBLEM=scale_separated_ssd.

Checks both helicities, two shells, seed reproducibility, MeshBlock independence,
CT divergence, Fourier support, the continuum Beltrami identity, grid convergence,
and a short driven evolution. Fourier diagnostics account for face averaging.
"""
import logging
import os
from pathlib import Path
import sys
import subprocess

import numpy as np
import scripts.utils.athena as athena

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'vis/python'))
import bin_convert  # noqa: E402

logger = logging.getLogger('athena' + __name__[7:])
INPUT = 'turbulence/scale_separated_ssd.athinput'
CASES = [('plus32', 32, 16, 1, 5, 210989),
         ('single32', 32, 32, 1, 5, 210989),
         ('minus32', 32, 16, -1, 5, 210989),
         ('plus64', 64, 32, 1, 5, 210989),
         ('seed32', 32, 16, 1, 5, 210990),
         ('shell3', 32, 16, 1, 3, 210989)]
metrics = {}


def run(**kwargs):
    for name, nx, block, helicity, shell, seed in CASES:
        args = [f'job/basename=SSDTest_{name}', 'time/nlim=0',
                f'spectral_ic/helicity={helicity}', f'spectral_ic/nlow={shell}',
                f'spectral_ic/nhigh={shell}', f'spectral_ic/iseed={seed}']
        for axis in (1, 2, 3):
            args += [f'mesh/nx{axis}={nx}', f'meshblock/nx{axis}={block}']
        athena.run(INPUT, args)
    athena.run(INPUT, ['job/basename=SSDTest_driven', 'time/nlim=4',
                      'mesh/nx1=32', 'mesh/nx2=32', 'mesh/nx3=32',
                      'meshblock/nx1=16', 'meshblock/nx2=16', 'meshblock/nx3=16'])
    control_args = ['time/nlim=0', 'mesh/nx1=32', 'mesh/nx2=32', 'mesh/nx3=32',
                    'meshblock/nx1=16', 'meshblock/nx2=16', 'meshblock/nx3=16']
    athena.run(INPUT, ['job/basename=SSDTest_gaussian', 'spectral_ic/force_free=false',
                      *control_args])
    original = (REPO / 'inputs' / INPUT).read_text()
    for name, removed in (('legacy', 'force_free = true\n'),
                          ('default_helicity', 'helicity = 1\n')):
        control = Path('build/src/SSDTest_control.athinput').resolve()
        control.write_text(original.replace(removed, ''))
        athena.run(os.path.relpath(control, REPO / 'inputs'),
                   [f'job/basename=SSDTest_{name}', *control_args])
        control.unlink()
    invalid = [('spectral_ic/helicity=0',), ('spectral_ic/rms_b=-1',),
               ('spectral_ic/nlow=4',), ('spectral_ic/nlow=0',),
               ('spectral_ic/b_mean_x=0.1',),
               ('spectral_ic/nlow=32', 'spectral_ic/nhigh=32'),
               ('mesh/x2max=2',),
               ('mesh/ix1_bc=outflow', 'mesh/ox1_bc=outflow'),
               ('mhd/eos=isothermal', 'mhd/iso_sound_speed=1')]
    invalid_input = Path('build/src/SSDTest_invalid.athinput').resolve()
    text = (REPO / 'inputs' / INPUT).read_text()
    text = text.replace('eos = ideal', 'eos = ideal\niso_sound_speed = 1.0')
    invalid_input.write_text(text.replace('<spectral_ic>',
                                          '<spectral_ic>\nb_mean_x = 0.0'))
    for overrides in invalid:
        result = subprocess.run(
            ['./athena', '-i', str(invalid_input),
             *overrides, 'time/nlim=0'],
            cwd='build/src', capture_output=True, text=True)
        assert result.returncode != 0, overrides
        assert ('spectral seed' in result.stderr or
                'Force-free ICs require' in result.stderr or
                'requires Newtonian ideal-gas MHD' in result.stderr), (
                    overrides, result.stdout, result.stderr)
    invalid_input.unlink()



def assemble(path):
    data = bin_convert.read_binary(str(path))
    shape = tuple(data[f'Nx{a}'] for a in (3, 2, 1))
    fields = {v: np.empty(shape) for v in data['var_names']}
    for m, (x, y, z, level) in enumerate(data['mb_logical']):
        assert level == 0
        for key in fields:
            a = data['mb_data'][key][m]
            nz, ny, nx = a.shape
            fields[key][z*nz:(z+1)*nz, y*ny:(y+1)*ny, x*nx:(x+1)*nx] = a
    return fields


def check_field(root, name, nx, helicity, shell):
    root = Path(root)
    fields = assemble(root / f'bin/{name}.prim.00000.bin')
    b = np.array([fields[f'bcc{a}'] for a in (1, 2, 3)])
    rms = np.sqrt(np.mean(np.sum(b*b, axis=0)))
    np.testing.assert_allclose(rms, 0.01, rtol=1e-7)
    np.testing.assert_allclose(fields['dens'], 1.0, atol=1e-7)
    np.testing.assert_allclose(fields['eint'], 1.5, atol=2e-7)
    for axis in 'xyz':
        assert np.max(abs(fields['vel'+axis])) == 0.0
    assert np.max(abs(b.mean(axis=(1, 2, 3)))) < 1e-9*rms
    div = assemble(root / f'bin/{name}.divb.00000.bin')
    max_div = np.max(abs(next(iter(div.values()))))
    assert max_div < 1e-11*rms*2*np.pi*shell
    n = np.fft.fftfreq(nx)*nx
    nz, ny, nx_grid = np.meshgrid(n, n, n, indexing='ij')
    wave = np.array([nx_grid, ny, nz])
    mask = np.sum(wave**2, axis=0) == shell**2
    bhat = np.fft.fftn(b, axes=(1, 2, 3))/nx**3
    power = np.sum(abs(bhat)**2, axis=0)
    leakage = power[~mask].sum()/power.sum()
    assert leakage < 1e-13
    # Bcc is the arithmetic average of analytic face averages. Undo its known
    # component-dependent transfer function on the occupied shell ONLY.
    # This reconstructs the continuum coefficients, not a relaxed field.
    continuum = np.zeros_like(bhat)
    for a in range(3):
        transfer = np.cos(np.pi*wave[a]/nx)
        for axis in range(3):
            if axis != a:
                transfer *= np.sinc(wave[axis]/nx)
        continuum[a, mask] = bhat[a, mask]/transfer[mask]
    current = 1j*2*np.pi*np.cross(wave, continuum, axisa=0, axisb=0, axisc=0)
    alpha = helicity*2*np.pi*shell
    residual = np.linalg.norm(current-alpha*continuum)/np.linalg.norm(alpha*continuum)
    assert residual < 2e-7
    divergence = np.linalg.norm(np.sum(wave*continuum, axis=0))
    assert divergence < 2e-7*shell*np.linalg.norm(continuum)
    # Actual centered cell diagnostic: its Lorentz-force error must converge away.
    def derivative(a, axis):
        return (np.roll(a, -1, axis)-np.roll(a, 1, axis))*nx/2
    j = np.array([derivative(b[2], 1)-derivative(b[1], 0),
                  derivative(b[0], 0)-derivative(b[2], 2),
                  derivative(b[1], 2)-derivative(b[0], 1)])
    force = np.cross(j, b, axisa=0, axisb=0, axisc=0)
    force_error = np.sqrt(np.mean(np.sum(force**2, axis=0)))/(abs(alpha)*rms*rms)
    metrics[name] = dict(rms=float(rms), max_div=float(max_div),
                         shell_leakage=float(leakage), beltrami_error=float(residual),
                         centered_force_error=float(force_error))
    return b, force_error


def analyze():
    root = Path('build/src')
    try:
        fields, errors = {}, {}
        for name, nx, block, helicity, shell, seed in CASES:
            fields[name], errors[name] = check_field(root, 'SSDTest_'+name,
                                                     nx, helicity, shell)
        np.testing.assert_allclose(fields['plus32'], fields['single32'],
                                   rtol=0, atol=1e-12)
        controls = {}
        for name in ('gaussian', 'legacy', 'default_helicity'):
            d = assemble(root / f'bin/SSDTest_{name}.prim.00000.bin')
            controls[name] = np.array([d[f'bcc{a}'] for a in (1, 2, 3)])
        np.testing.assert_array_equal(controls['gaussian'], controls['legacy'])
        np.testing.assert_array_equal(controls['default_helicity'], fields['plus32'])
        assert np.linalg.norm(controls['gaussian']-fields['plus32']) > 0.1
        assert np.linalg.norm(fields['plus32']-fields['seed32']) > 0.1
        assert errors['plus64'] < 0.35*errors['plus32']
        files = sorted((root / 'bin').glob('SSDTest_driven.prim.*.bin'))
        assert len(files) >= 2
        final = assemble(files[-1])
        assert all(np.all(np.isfinite(a)) for a in final.values())
        assert np.min(final['dens']) > 0 and np.min(final['eint']) > 0
        assert sum(np.sum(final['vel'+a]**2) for a in 'xyz') > 0.0
        logger.info('Force-free seed validation: %s', metrics)
    except (AssertionError, OSError, ValueError, KeyError) as error:
        logger.warning('Force-free seed regression failed: %s', error)
        return False
    for directory in (root, root/'bin', root/'rst'):
        for path in directory.glob('SSDTest_*'):
            if path.is_file():
                path.unlink()
    return True
