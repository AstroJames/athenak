"""Numerical resistivity at the seed scale (optional, ~2 min on 4 ranks).

Build -DPROBLEM=newtonian_SSD. Optional: run explicitly with
`run_tests.py mhd/shell_seed_advection`.
A 64^3 tile with dx = 1/288 (the template resolution; L = 64/288) carries a non-helical
shell seed at 9 cells per wavelength (n0 = 64/9, shell_width = 0.49, k0 = 2 pi 32), with
B_rms = 1e-6 (kinematic) and the template numerics (teno5, HLLD, FOFC, RK3, CFL 0.3).
A uniform oblique flow u = 0.3 (1,2,3)/sqrt(14) is imposed by editing the t = 0 restart.
A uniform translation leaves every |b_n| unchanged, so per-mode face power gives
  eta_eff,m = -ln(P_m(T)/P_m(0)) / (2 kappa_m^2 T),
with kappa^2 = sum_a k_a^2 sinc^2(k_a dx/2), the discrete CT resistive eigenvalue.
Runs: eta = 0 (advected: eta_num), eta = 2.5e-5
(advected: eta + eta_num), and eta = 2.5e-5 static (control: eta_eff = eta exactly).
Reported: eta_num, eta_num/eta for eta = 2.5e-5 and for the template eta = 7.5e-5, the
spread over mode directions, and additivity of the explicit and numerical parts.
"""
import logging
from pathlib import Path
import struct

import numpy as np
import scripts.utils.athena as athena  # noqa: F401  (harness builds via this module)
from scripts.mhd.shell_seed import launcher, read_restart, wavenumbers
import subprocess

logger = logging.getLogger('athena' + __name__[7:])
NX, L = 64, 64.0/288.0
N0 = 64.0/9.0
U = 0.3*np.array([1.0, 2.0, 3.0])/np.sqrt(14.0)
ETA, ETA_TEMPLATE, TLIM = 2.5e-5, 7.5e-5, 0.2
DECK = 'build/src/ShellAdvectTest.athinput'
metrics = {}

TEXT = f"""<job>
basename = ShellAdvectTest
<mesh>
nx1 = {NX}
nx2 = {NX}
nx3 = {NX}
nghost = 4
x1min = 0.0
x1max = {L!r}
x2min = 0.0
x2max = {L!r}
x3min = 0.0
x3max = {L!r}
ix1_bc = periodic
ox1_bc = periodic
ix2_bc = periodic
ox2_bc = periodic
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = 32
nx2 = 32
nx3 = 32
<time>
evolution = dynamic
integrator = rk3
cfl_number = 0.3
tlim = {TLIM}
nlim = 0
ndiag = 1000
<mhd>
eos = isothermal
iso_sound_speed = 1.0
reconstruct = teno5
rsolver = hlld
fofc = true
ohmic_resistivity = 0.0
<spectral_ic>
seed_type = shell
n0 = {N0!r}
shell_width = 0.49
n_dirs = 512
helicity = 0
amplitude = equal
iseed = 210989
rms_b = 1.0e-6
seed_on_restart = false
<output1>
file_type = rst
dt = 100.0
<output2>
file_type = hst
dt = 0.005
data_format = %24.16e
"""


def athena_run(args, restart=None):
    source = ['-r', str(Path(restart).resolve())] if restart else ['-i', Path(DECK).name]
    subprocess.run([*launcher(), '-n', '4', './athena', *source, *args],
                   cwd='build/src', check=True, capture_output=True)


def set_uniform_velocity(path, velocity, out):
    """Copy a 4-variable isothermal-MHD restart with momentum rho*u everywhere."""
    blob = bytearray(Path(path).read_bytes())
    offset = blob.index(b'<par_end>\n')+len(b'<par_end>\n')
    nmb, = struct.unpack_from('=i', blob, offset)
    ng, nx, ny, nz = struct.unpack_from('=4i', blob, offset+8+9*8+19*4)
    offset += 8+9*8+2*19*4+20+20*nmb  # no RNG_State: no <turb_driving>
    size, = struct.unpack_from('=Q', blob, offset)
    offset += 8
    nc = (nx+2*ng)*(ny+2*ng)*(nz+2*ng)
    payload = np.frombuffer(blob, dtype='=f8', offset=offset).reshape(nmb, size//8)
    payload.setflags(write=True)
    for m in range(nmb):
        rho = payload[m, :nc]
        for a in range(3):
            payload[m, (a+1)*nc:(a+2)*nc] = rho*velocity[a]
    Path(out).write_bytes(blob)


def run(**kwargs):
    Path(DECK).write_text(TEXT)
    athena_run(['job/basename=ShellAdvectTest_init'])
    start = sorted(Path('build/src/rst').glob('ShellAdvectTest_init.*.rst'))[0]
    for name, eta, velocity in (('adv_eta0', 0.0, U), ('adv_eta', ETA, U),
                                ('static_eta', ETA, 0.0*U)):
        moving = Path(f'build/src/ShellAdvectTest_{name}_start.rst')
        set_uniform_velocity(start, velocity, moving)
        athena_run([f'job/basename=ShellAdvectTest_{name}', 'time/nlim=-1',
                    f'mhd/ohmic_resistivity={eta}'], restart=moving)
        moving.unlink()
    Path(DECK).unlink()


def mode_eta(name):
    root = Path('build/src')
    init = read_restart(sorted((root/'rst').glob('ShellAdvectTest_init.*.rst'))[0])
    final = read_restart(sorted((root/'rst').glob(f'ShellAdvectTest_{name}.*.rst'))[-1])
    assert final['time'] >= TLIM*(1-1e-12) and np.all(np.isfinite(final['faces']))
    wave = wavenumbers(NX)
    p0 = np.sum(abs(np.fft.fftn(init['faces'], axes=(1, 2, 3)))**2, axis=0)
    p1 = np.sum(abs(np.fft.fftn(final['faces'], axes=(1, 2, 3)))**2, axis=0)
    occupied = p0 > 1e-12*p0.max()
    kappa2 = np.sum((2*NX/L*np.sin(np.pi*wave/NX))**2, axis=0)
    k2 = np.sum((2*np.pi*wave/L)**2, axis=0)
    eta = -np.log(p1[occupied]/p0[occupied])/(2*kappa2[occupied]*final['time'])
    khat = wave[:, occupied]/np.sqrt(np.sum(wave[:, occupied]**2, axis=0))
    cosine = abs(U @ khat)/np.linalg.norm(U)
    weight = p0[occupied]/p0[occupied].sum()
    expected = U if name.startswith('adv') else 0.0*U
    velocity = final['u'][1:4]/final['u'][0]-expected[:, None, None, None]
    return dict(eta_mean=float(np.sum(weight*eta)), eta_min=float(eta.min()),
                eta_max=float(eta.max()), modes=int(occupied.sum()),
                kappa2_over_k2=float(np.sum(weight*kappa2[occupied]/k2[occupied])),
                corr_eta_vs_cos_k_u=float(np.corrcoef(eta, cosine)[0, 1])
                if eta.std() > 0 else 0.0,
                max_velocity_change=float(np.max(abs(velocity))),
                leakage=float(p1[~occupied].sum()/p1.sum()), time=float(final['time']))


def analyze():
    try:
        for name in ('adv_eta0', 'adv_eta', 'static_eta'):
            metrics[name] = mode_eta(name)
        static, eta0, eta = (metrics[n] for n in ('static_eta', 'adv_eta0', 'adv_eta'))
        # Control: the explicit CT operator alone gives eta_eff = eta for every mode.
        assert abs(static['eta_min']/ETA-1) < 1e-4 and abs(static['eta_max']/ETA-1) < 1e-4
        eta_num = eta0['eta_mean']
        # Numerical diffusion is dissipative and smaller than the explicit eta here.
        assert 0.0 < eta_num < ETA, eta_num
        additive = (eta['eta_mean']-ETA)/eta_num
        assert abs(additive-1) < 0.25, additive
        # Upwind reconstruction moves a little power out of the shell (reported).
        assert eta0['leakage'] < 1e-4 and eta0['max_velocity_change'] < 1e-6
        metrics['summary'] = dict(
            eta_num=eta_num, eta_num_over_eta_2p5e_5=eta_num/ETA,
            eta_num_over_eta_template_7p5e_5=eta_num/ETA_TEMPLATE,
            additivity_ratio=additive, eta_num_per_u_dx=eta_num/(np.linalg.norm(U)*L/NX),
            kappa2_over_k2=eta0['kappa2_over_k2'])
        logger.info('Shell seed numerical resistivity: %s', metrics)
    except (AssertionError, OSError, ValueError, KeyError, IndexError) as error:
        logger.warning('Shell seed advection test failed: %s', error, exc_info=True)
        return False
    for directory in (Path('build/src'), Path('build/src/rst')):
        for path in directory.glob('ShellAdvectTest*'):
            if path.is_file():
                path.unlink()
    return True
