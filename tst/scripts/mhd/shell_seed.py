"""Isotropic narrow-shell seed (spectral_ic/seed_type=shell).

Build with -DPROBLEM=newtonian_SSD.

Checks, from double-precision restart face fields with the known face-average
transfer function undone (exact for this analytic edge-averaged construction):
- div B at round-off, rms_b, zero mean, all power in ||n|-n0| <= shell_width;
- per-mode Beltrami residual (curl b = h|k| b for h=+/-1), relative helicity
  H/(2E/k) = +/-1, and |H| at round-off for h=0, mode by mode;
- weighted 2nd/4th angular moments D2, D4 (max-norm, models/kazantsev/seed_design.md)
  of the mode set, with weights recovered from the equal-amplitude mode powers, and the
  in-code Voronoi weights against the exact scipy.spatial.SphericalVoronoi areas;
- the full-lattice-shell fallback (<= 1.5 n_dirs/2 pairs in the shell) and the logged N_eff;
- the helicity_spectrum .spec column against an independent FFT of bcc;
- MeshBlock and MPI-rank independence, same-iseed determinism, a different iseed
  giving a different (rotated) mode set;
- restart injection (seed_on_restart): flow, forcing and RNG preserved bitwise and the
  injected field identical to a fresh start;
- unforced resistive decay: E_B(t) = sum_m P_m exp(-2 eta kappa_m^2 t), with the CT
  operator eigenvalue kappa^2 = sum_a k_a^2 sinc^2(k_a dx/2) per mode.
Isotropy tolerances follow project/handoffs/2026-11-11_theory-to-athenak_seed-checks.md:
D2 <= 1e-3, D4 <= 5e-3 at n_dirs = 512 (the default); 2.5e-3, 1.5e-2 at 256.
"""
import logging
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess

import numpy as np
import scripts.utils.athena as athena
from scripts.mhd.force_free_ssd import assemble

logger = logging.getLogger('athena' + __name__[7:])
REPO = Path(__file__).resolve().parents[3]
DECK = 'build/src/ShellSeedTest.athinput'
N0, WIDTH, NDIRS = 16.0, 0.5, 512
# (D2, D4) by 2 x modes: theory values for 512 and 256; the 128 tier is the theory
# table's 90th percentile at n_dirs = 128 (used for small full-lattice shells).
TOLERANCE = {512: (1e-3, 5e-3), 256: (2.5e-3, 1.5e-2), 128: (5e-3, 4e-2)}
metrics = {}

BASE = """<job>
basename = ShellSeedTest
<mesh>
nx1 = 64
nx2 = 64
nx3 = 64
nghost = 4
x1min = 0.0
x1max = 1.0
x2min = 0.0
x2max = 1.0
x3min = 0.0
x3max = 1.0
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
tlim = 1.0
nlim = 0
ndiag = 100
<mhd>
eos = isothermal
iso_sound_speed = 1.0
reconstruct = teno5
rsolver = hlld
fofc = true
<spectral_ic>
seed_type = shell
n0 = 16
shell_width = 0.5
n_dirs = 512
helicity = 1
amplitude = equal
iseed = 210989
rms_b = 0.01
seed_on_restart = false
<output1>
file_type = power_spectrum
variable = magnetic_field
id = magnetic
helicity_spectrum = true
dt = 1.0
<output2>
file_type = rst
dt = 1.0
<output3>
file_type = bin
variable = mhd_divb
id = divb
dt = 1.0
<output4>
file_type = hst
dt = 0.01
data_format = %24.16e
<output5>
file_type = bin
variable = mhd_w_bcc
id = prim
dt = 1.0
<output6>
file_type = power_spectrum
variable = velocity
id = velocity
helicity_spectrum = true
dt = 1.0
"""
DRIVING = """<turb_driving>
driving_geometry = isotropic
driving_profile = band
nlow = 1
nhigh = 2
dedt = 0.1
tcorr = 0.1
sol_weight = 1.0
"""
# name: overrides (all serial unless listed in MPI_CASES)
CASES = {
    'plus': [],
    'minus': ['spectral_ic/helicity=-1'],
    'zero': ['spectral_ic/helicity=0'],
    'gauss': ['spectral_ic/amplitude=gaussian'],
    'gauss_zero': ['spectral_ic/amplitude=gaussian', 'spectral_ic/helicity=0'],
    'blocks': ['meshblock/nx1=16', 'meshblock/nx2=16', 'meshblock/nx3=16'],
    'repeat': [],
    'seed2': ['spectral_ic/iseed=12345'],
    'dirs256': ['spectral_ic/n_dirs=256'],
    'fullshell': ['spectral_ic/n0=3.5'],
}
MPI_CASES = {'mpi4': []}
INVALID = [
    (['spectral_ic/n_dirs=201'], 'Shell seed requires'),
    (['spectral_ic/n_dirs=0'], 'Shell seed requires'),
    (['spectral_ic/helicity=2'], 'Shell seed requires'),
    (['spectral_ic/n0=31.6'], 'Shell seed requires'),
    (['spectral_ic/shell_width=0'], 'Shell seed requires'),
    (['spectral_ic/amplitude=lognormal'], "amplitude must be"),
    (['+force_free = true'], 'exclusive'),
    (['spectral_ic/seed_type=sphere'], 'Unknown spectral_ic/seed_type'),
    (['+b_mean_x = 0.1'], 'zero added mean'),
    (['spectral_ic/n0=1.2', 'spectral_ic/shell_width=0.1'], 'no lattice vector'),
    (['mesh/x2max=2'], 'Shell seed requires'),
    (['output1/variable=density'], 'helicity_spectrum requires'),
]
REQUIRED = ['n0', 'n_dirs', 'helicity', 'amplitude']
DECAY = dict(eta=1.0e-3, tlim=1.0)


def launcher():
    return shlex.split(os.environ.get('ATHENA_PACKING_MPI_LAUNCHER', 'mpiexec'))


def write_deck(text):
    path = Path(DECK).resolve()
    path.write_text(text)
    return os.path.relpath(path, REPO/'inputs')


def run_case(name, overrides, nproc=1, restart=None):
    """Run in build/src and keep stdout (the seed provenance lines) in a .log file."""
    command = [*launcher(), '-n', str(nproc)] if nproc > 1 else []
    source = ['-r', str(restart)] if restart else ['-i', Path(DECK).name]
    command += ['./athena', *source,
                f'job/basename=ShellSeedTest_{name}', *overrides]
    result = subprocess.run(command, cwd='build/src', capture_output=True, text=True)
    Path(f'build/src/ShellSeedTest_{name}.log').write_text(result.stdout+result.stderr)
    if result.returncode != 0:
        raise athena.AthenaError(f'{name}: return code {result.returncode}')


def run(**kwargs):
    write_deck(BASE)
    for name, overrides in CASES.items():
        run_case(name, overrides)
    for name, overrides in MPI_CASES.items():
        run_case(name, overrides, nproc=4)
    # Entries starting with '+' are new <spectral_ic> lines (the command line can
    # only override existing keys).
    for overrides, message in INVALID:
        added = '\n'.join(o[1:] for o in overrides if o.startswith('+'))
        Path(DECK).write_text(BASE.replace('<spectral_ic>', '<spectral_ic>\n'+added))
        result = subprocess.run(['./athena', '-i', Path(DECK).name,
                                 *[o for o in overrides if not o.startswith('+')],
                                 'job/basename=ShellSeedTest_invalid'],
                                cwd='build/src', capture_output=True, text=True)
        assert result.returncode != 0, overrides
        assert message in result.stderr+result.stdout, (overrides, result.stderr)
    for key in REQUIRED:
        text = re.sub(rf'^{key} = .*\n', '', BASE, flags=re.MULTILINE)
        Path(DECK).write_text(text)
        result = subprocess.run(['./athena', '-i', Path(DECK).name,
                                 'job/basename=ShellSeedTest_invalid'],
                                cwd='build/src', capture_output=True, text=True)
        assert result.returncode != 0, key
        assert key in result.stdout+result.stderr, (key, result.stdout)
    # Unforced resistive decay of a 32^3 shell (n0=3.5, 9.1 cells per wavelength).
    decay = BASE.replace('rms_b = 0.01', 'rms_b = 1.0e-6')
    decay = decay.replace('fofc = true',
                          f'fofc = true\nohmic_resistivity = {DECAY["eta"]}')
    write_deck(decay)
    for name, helicity in (('decay_zero', 0), ('decay_plus', 1)):
        run_case(name, ['time/nlim=-1',
                          f'time/tlim={DECAY["tlim"]}', 'output2/dt=100',
                          'output5/dt=100', 'output1/dt=100', 'output6/dt=100',
                          *[f'mesh/nx{a}=32' for a in (1, 2, 3)],
                          *[f'meshblock/nx{a}=16' for a in (1, 2, 3)],
                          'spectral_ic/n0=3.5', 'spectral_ic/n_dirs=60',
                          f'spectral_ic/helicity={helicity}'])
    # Driven spin-up with B = 0, then one-time shell injection on restart.
    write_deck(BASE+DRIVING)
    run_case('spinup', ['spectral_ic/rms_b=0', 'time/nlim=4'])
    run_case('injected', ['spectral_ic/seed_on_restart=true', 'spectral_ic/rms_b=0.01',
                          'time/nlim=4'], restart=latest_restart('spinup').resolve())
    # Curl-peak cadence: hst every cycle, velocity spectrum with k_nu every 3 cycles.
    cadence = (BASE+DRIVING).replace('<output4>\nfile_type = hst\ndt = 0.01',
                                     '<output4>\nfile_type = hst\ndcycle = 1')
    cadence = cadence.replace('id = velocity\nhelicity_spectrum = true\ndt = 1.0',
                              'id = velocity\nhelicity_spectrum = true\n'
                              'history_curl_peak = k_nu\ndcycle = 3')
    write_deck(cadence+'<problem>\nuser_hist = true\n')
    run_case('cadence', ['spectral_ic/rms_b=0', 'time/nlim=7', 'output1/dt=100'])
    Path(DECK).unlink()


def latest_restart(name):
    return sorted(Path('build/src/rst').glob(f'ShellSeedTest_{name}.*.rst'))[-1]


def read_restart(path, nvar=4):
    """Assemble double-precision MHD state from a uniform-grid restart file."""
    blob = path.read_bytes()
    offset = blob.index(b'<par_end>\n')+len(b'<par_end>\n')
    header = blob[:offset].decode()
    nmb, = struct.unpack_from('=i', blob, offset)
    mesh = struct.unpack_from('=19i', blob, offset+8+9*8)
    block = struct.unpack_from('=19i', blob, offset+8+9*8+19*4)
    ng, nx, ny, nz = block[:4]
    gx, gy, gz = mesh[1:4]
    offset += 8+9*8+2*19*4
    time, _, cycle = struct.unpack_from('=ddi', blob, offset)
    offset += 20
    loc = np.frombuffer(blob, dtype='=i4', count=4*nmb, offset=offset).reshape(nmb, 4)
    offset += 20*nmb
    rng = b''
    if '<turb_driving>' in header:
        rng = blob[offset:offset+struct.calcsize('@35qid')]
        offset += struct.calcsize('@35qid')
    size, = struct.unpack_from('=Q', blob, offset)
    offset += 8
    shape = (nz+2*ng, ny+2*ng, nx+2*ng)
    nc = int(np.prod(shape))
    nf = 3*nc+nc//shape[0]+nc//shape[1]+nc//shape[2]
    nforce = (size//8-nvar*nc-nf)//nc
    assert size == (nvar*nc+nf+nforce*nc)*8 and len(blob) == offset+nmb*size
    payload = np.frombuffer(blob, dtype='=f8', offset=offset).reshape(nmb, -1)
    u = np.empty((nvar, gz, gy, gx))
    force = np.empty((nforce, gz, gy, gx))
    faces = [np.empty((gz, gy, gx)) for _ in range(3)]
    act = (slice(ng, ng+nz), slice(ng, ng+ny), slice(ng, ng+nx))
    for m in range(nmb):
        sl = (slice(loc[m, 2]*nz, (loc[m, 2]+1)*nz),
              slice(loc[m, 1]*ny, (loc[m, 1]+1)*ny),
              slice(loc[m, 0]*nx, (loc[m, 0]+1)*nx))
        data = payload[m]
        u[(slice(None),)+sl] = data[:nvar*nc].reshape(nvar, *shape)[(slice(None),)+act]
        start = nvar*nc
        for a, extra in enumerate(((0, 0, 1), (0, 1, 0), (1, 0, 0))):
            fshape = tuple(s+e for s, e in zip(shape, extra))
            count = int(np.prod(fshape))
            faces[a][sl] = data[start:start+count].reshape(fshape)[act]
            start += count
        rest = data[start:]
        if nforce:
            force[(slice(None),)+sl] = rest.reshape(nforce, *shape)[(slice(None),)+act]
    return dict(header=header, time=time, cycle=cycle, rng=rng, u=u, force=force,
                faces=np.array(faces))


def wavenumbers(n):
    m = np.rint(np.fft.fftfreq(n)*n)
    kz, ky, kx = np.meshgrid(m, m, m, indexing='ij')
    return np.array([kx, ky, kz])


def continuum(faces):
    """Continuum Fourier coefficients b_n (per unit mode, sum_n |b_n|^2 = <B^2>)."""
    n = faces.shape[-1]
    wave = wavenumbers(n)
    theta = np.pi*wave/n  # k_a dx/2
    bhat = np.fft.fftn(faces, axes=(1, 2, 3))/n**3
    out = np.zeros_like(bhat)
    for a in range(3):
        # Face a sits at the left edge in a and at cell centres in the other axes:
        # FFT phase exp(i k.origin), and face averages carry sinc in the other axes.
        transfer = np.ones(wave.shape[1:], dtype=complex)
        for axis in range(3):
            if axis != a:
                transfer *= np.sinc(wave[axis]/n)*np.exp(1j*theta[axis])
        out[a] = bhat[a]/transfer
    return wave, out


def mode_set(wave, b, n0=N0, width=WIDTH):
    """Occupied modes, one per +/- pair (first non-zero component positive)."""
    power = np.sum(abs(b)**2, axis=0)
    norm = np.sqrt(np.sum(wave**2, axis=0))
    occupied = power > 1e-20*power.max()
    lead = np.where(wave[0] != 0, wave[0], np.where(wave[1] != 0, wave[1], wave[2]))
    keep = occupied & (lead > 0)
    shell = abs(norm-n0) <= width+1e-12
    leak = power[~shell].sum()/power.sum()
    return keep, leak


def helical_parts(n, b):
    """Project b (3, M) on the helical basis e_+/e_- of n (3, M); |c+|^2, |c-|^2."""
    khat = n/np.linalg.norm(n, axis=0)
    axis = np.argmin(abs(khat), axis=0)
    e1 = np.zeros_like(khat)
    e1[axis, np.arange(khat.shape[1])] = 1.0
    e1 -= khat*np.sum(e1*khat, axis=0)
    e1 /= np.linalg.norm(e1, axis=0)
    e2 = np.cross(khat, e1, axis=0)
    ep, em = (e1+1j*e2)/np.sqrt(2), (e1-1j*e2)/np.sqrt(2)
    cp = np.sum(np.conj(ep)*b, axis=0)
    cm = np.sum(np.conj(em)*b, axis=0)
    return abs(cp)**2, abs(cm)**2


def angular_moments(u, w):
    """Max-norm D2, D4 of the weighted antipodal set (theory seed_design.md section 2)."""
    full, weight = np.concatenate([u, -u], axis=1), np.concatenate([w, w])
    weight = 4*np.pi*weight/weight.sum()
    m2 = np.einsum('a,ia,ja->ij', weight, full, full)
    eye = np.eye(3)
    iso = 4*np.pi/15*(np.einsum('ij,kl->ijkl', eye, eye)
                      + np.einsum('ik,jl->ijkl', eye, eye)
                      + np.einsum('il,jk->ijkl', eye, eye))
    m4 = np.einsum('a,ia,ja,ka,la->ijkl', weight, full, full, full, full)
    return (np.max(abs(m2-4*np.pi/3*eye))/(4*np.pi/3),
            np.max(abs(m4-iso))/(4*np.pi/15))


def shell_pairs(n0, width=WIDTH):
    """All +/- lattice pairs with ||n|-n0| <= width (first non-zero component > 0)."""
    m = int(np.floor(n0+width))
    r = np.arange(-m, m+1)
    a, b, c = np.meshgrid(r, r, r, indexing='ij')
    lead = np.where(a != 0, a, np.where(b != 0, b, c))
    norm = np.sqrt(a*a+b*b+c*c)
    return int(np.sum((lead > 0) & (abs(norm-n0) <= width)))


def log_values(name):
    text = Path(f'build/src/ShellSeedTest_{name}.log').read_text()
    e2, e4 = re.search(r'moment errors D2=(\S+) D4=(\S+)', text).groups()
    modes = int(re.search(r'modes=(\d+)', text)[1])
    pairs = int(re.search(r'(\d+) pairs in shell', text)[1])
    neff = float(re.search(r'N_eff=([0-9.eE+-]+)', text)[1])
    digest = re.search(r'mode_table_hash=(0x[0-9a-f]+)', text)[1]
    return dict(e2=float(e2), e4=float(e4), modes=modes, pairs=pairs, neff=neff,
                full_shell='full lattice shell' in text, hash=digest)


def check_seed(name, helicity, equal=True, n0=N0, rms=0.01, ndirs=NDIRS):
    state = read_restart(latest_restart(name))
    faces = state['faces']
    n = faces.shape[-1]
    # Upper periodic faces equal lower faces; CT divergence per cell.
    div = sum(np.roll(faces[a], -1, axis=2-a)-faces[a] for a in range(3))*n
    bcc = np.array([0.5*(faces[a]+np.roll(faces[a], -1, axis=2-a)) for a in range(3)])
    b_rms = np.sqrt(np.mean(np.sum(bcc**2, axis=0)))
    np.testing.assert_allclose(b_rms, rms, rtol=1e-12)
    max_div = np.max(abs(div))/(2*np.pi*n0*rms)
    assert max_div < 1e-12, max_div
    mean = np.max(abs(faces.mean(axis=(1, 2, 3))))/rms
    assert mean < 1e-14, mean
    wave, b = continuum(faces)
    keep, leak = mode_set(wave, b, n0)
    assert leak < 1e-26, leak
    nvec, bm = wave[:, keep], b[:, keep]
    k = 2*np.pi*nvec
    kmag = np.linalg.norm(k, axis=0)
    transverse = np.max(abs(np.sum(k*bm, axis=0))/(kmag*np.linalg.norm(bm, axis=0)))
    assert transverse < 1e-12, transverse
    curl = 1j*np.cross(k, bm, axis=0)
    energy = np.sum(abs(bm)**2, axis=0)
    # Helicity per mode, Re(b^* . i k x b)/k^2; with E_n = |b_n|^2/2 per signed mode,
    # helicity/(2E/k) = Re(b^* . i k x b)/(k |b|^2).
    rel_h = np.real(np.sum(np.conj(bm)*curl, axis=0))/(kmag*energy)
    if helicity != 0:
        beltrami = np.linalg.norm(curl-helicity*kmag*bm)/np.linalg.norm(kmag*bm)
        assert beltrami < 1e-12, beltrami
        assert np.max(abs(rel_h-helicity)) < 1e-12
    else:
        beltrami = np.nan
        assert np.max(abs(rel_h)) < 1e-12, np.max(abs(rel_h))
    cp, cm = helical_parts(nvec, bm)
    if helicity == 0:
        np.testing.assert_allclose(cp, cm, rtol=1e-11)
    total_h = np.sum(np.real(np.sum(np.conj(bm)*curl, axis=0))/kmag**2)
    unit = nvec/np.linalg.norm(nvec, axis=0)
    info = log_values(name)
    assert keep.sum() == info['modes']
    assert info['pairs'] == shell_pairs(n0)
    assert info['full_shell'] == (2*info['pairs'] <= 3*(ndirs//2))  # 1.5 x n_dirs/2
    if info['full_shell']:
        assert info['modes'] == info['pairs']
    np.testing.assert_allclose(info['neff'], energy.sum()**2/np.sum(energy**2), rtol=1e-4)
    result = dict(rms=float(b_rms), max_div=float(max_div), mean=float(mean),
                  leakage=float(leak), modes=int(keep.sum()),
                  max_transverse=float(transverse), beltrami=float(beltrami),
                  max_rel_helicity_err=float(np.max(abs(rel_h-helicity))),
                  total_rel_helicity=float(total_h/np.sum(energy/kmag)),
                  log_D2=info['e2'], log_D4=info['e4'], N_eff=info['neff'],
                  pairs_in_shell=info['pairs'], full_shell=info['full_shell'],
                  hash=info['hash'])
    if equal:
        weight = energy/energy.sum()*2*np.pi  # sum over the +/- set = 4 pi
        e2, e4 = angular_moments(unit, weight)
        tier = max([t for t in TOLERANCE if 2*unit.shape[1] >= 0.9*t] or [128])
        d2_max, d4_max = TOLERANCE[tier]
        assert e2 <= d2_max and e4 <= d4_max, (e2, e4)
        np.testing.assert_allclose([e2, e4], [info['e2'], info['e4']], rtol=2e-3)
        from scipy.spatial import SphericalVoronoi
        # Collinear modes (same n/gcd) share one cell, split equally in the code.
        primitive = (nvec/np.gcd.reduce(abs(nvec).astype(int), axis=0)).T.astype(int)
        directions, group = np.unique(primitive, axis=0, return_inverse=True)
        group = group.ravel()
        cells = directions.T/np.linalg.norm(directions.T, axis=0)
        exact = SphericalVoronoi(np.concatenate([cells, -cells], axis=1).T)
        area = exact.calculate_areas()[:cells.shape[1]]
        members = np.bincount(group)
        voronoi_err = np.max(abs(weight/(area[group]/members[group])-1))
        assert np.all(weight > 0)
        assert voronoi_err < 1e-2, voronoi_err
        result.update(D2=float(e2), D4=float(e4), voronoi_max_rel_err=float(voronoi_err),
                      voronoi_rms_rel_err=float(np.sqrt(np.mean(
                          (weight/(area[group]/members[group])-1)**2))),
                      collinear_modes=int(np.sum(members[group] > 1)),
                      weight_min_max=[float(weight.min()*unit.shape[1]/(2*np.pi)),
                                      float(weight.max()*unit.shape[1]/(2*np.pi))])
    metrics[name] = result
    return state, wave, b, keep, bcc


def check_helicity_spectrum(name, bcc, field='magnetic', index=0):
    """The .spec helicity column against an independent double-precision FFT."""
    root = Path('build/src')
    spec = np.loadtxt(sorted(root.glob(f'ShellSeedTest_{name}.{field}.*.spec'))[index])
    assert spec.shape[1] == 3
    n = bcc.shape[-1]
    wave = wavenumbers(n)
    fhat = np.fft.fftn(bcc, axes=(1, 2, 3))/n**3
    k = 2*np.pi*wave
    for a in range(3):
        k[a][abs(wave[a]) == n/2] = 0.0
    hel = np.real(np.sum(np.conj(fhat)*1j*np.cross(k, fhat, axis=0), axis=0))
    k2 = np.sum(k*k, axis=0)
    if field == 'magnetic':
        hel = np.where(k2 > 0, hel/np.where(k2 > 0, k2, 1), 0.0)
    shells = np.floor(np.sqrt(np.sum(wave**2, axis=0))).astype(int)
    expected = np.bincount(shells.ravel(), weights=hel.ravel())[1:len(spec)+1]
    power = np.bincount(shells.ravel(), weights=np.sum(abs(fhat)**2, axis=0).ravel())
    np.testing.assert_allclose(spec[:, 1], power[1:len(spec)+1], rtol=1e-10,
                               atol=1e-14*power.sum())
    # Relative helicity of the binned field: sum H / sum (|f|^2/|k|) (magnetic) or
    # sum H / sum (|f|^2 |k|) (kinetic); +/-1 for a maximally helical field.
    kmag = np.sqrt(k2)
    f2 = np.sum(abs(fhat)**2, axis=0)
    inside = (shells >= 1) & (shells <= len(spec)) & (kmag > 0)
    norm = f2/np.where(kmag > 0, kmag, 1) if field == 'magnetic' else f2*kmag
    # Absolute tolerance from the realizability bound, so h=0 fields compare at round-off.
    scale = np.sum(norm[inside])+1e-300
    np.testing.assert_allclose(spec[:, 2], expected, rtol=1e-8, atol=1e-12*scale)
    return spec, float(np.sum(hel[inside])/scale)


def check_decay(name):
    root = Path('build/src')
    state = read_restart(sorted((root/'rst').glob(f'ShellSeedTest_{name}.*.rst'))[0])
    assert state['cycle'] == 0
    faces = state['faces']
    n = faces.shape[-1]
    wave = wavenumbers(n)
    # History ME is (1/4)(b_i^2+b_{i+1}^2) per face direction, i.e. face power per
    # mode (no cell-centring cos^2 factor); each mode then decays independently.
    power = np.sum(abs(np.fft.fftn(faces, axes=(1, 2, 3))/n**3)**2, axis=0)
    theta = np.pi*wave/n
    kappa2 = np.sum((2*n*np.sin(theta))**2, axis=0)  # CT resistive eigenvalue
    k2 = np.sum((2*np.pi*wave)**2, axis=0)
    occupied = power > 1e-20*power.max()
    hist = np.loadtxt(root/f'ShellSeedTest_{name}.mhd.hst')
    header = (root/f'ShellSeedTest_{name}.mhd.hst').read_text().splitlines()[1]
    labels = [item.split('=')[1] for item in header.split() if '=' in item]
    t = hist[:, labels.index('time')]
    me = sum(hist[:, labels.index(f'{a}-ME')] for a in (1, 2, 3))
    eta = DECAY['eta']
    p, kk, kc = power[occupied], kappa2[occupied], k2[occupied]
    predicted = np.array([np.sum(p*np.exp(-2*eta*kk*s)) for s in t])/p.sum()
    continuum_pred = np.array([np.sum(p*np.exp(-2*eta*kc*s)) for s in t])/p.sum()
    measured = me/me[0]
    err = np.max(abs(measured/predicted-1))
    assert t[-1] >= 0.99*DECAY['tlim'] and err < 1e-6, err
    rate = -np.polyfit(t, np.log(measured), 1)[0]
    ratio = rate/(-np.polyfit(t, np.log(continuum_pred), 1)[0])
    metrics[name] = dict(max_rel_err_vs_discrete=float(err), cycles=len(t),
                         fitted_rate=float(rate),
                         rate_over_continuum_2_eta_k2=float(ratio),
                         mean_kappa2_over_k2=float(np.sum(p*kk/kc)/p.sum()),
                         max_rel_err_vs_continuum=float(np.max(abs(measured/continuum_pred
                                                                    - 1))))


def check_cadence():
    """hst k_nu rows equal the peak of the latest velocity spectrum at or before them."""
    root = Path('build/src')
    spectra = []
    for path in sorted(root.glob('ShellSeedTest_cadence.velocity.*.spec')):
        head = path.read_text().splitlines()[0]
        time = float(re.search(r'time=(\S+)', head)[1])
        cycle = int(re.search(r'cycle=(\d+)', head)[1])
        data = np.loadtxt(path)
        peak = data[np.argmax(data[:, 2]), 0]*2*np.pi if np.any(data[:, 2]) else 0.0
        spectra.append((time, cycle, peak))
    assert [c for _, c, _ in spectra] == [0, 3, 6, 7], spectra
    hist_path = root/'ShellSeedTest_cadence.user.hst'
    labels = [x.split('=')[1] for x in hist_path.read_text().splitlines()[1].split()
              if '=' in x]
    rows = np.loadtxt(hist_path, ndmin=2)
    assert len(rows) >= 8
    for row in rows:
        latest = [peak for time, _, peak in spectra if time <= row[0]][-1]
        assert row[labels.index('k_nu')] == latest, (row, spectra)
    assert any(peak > 0 for _, _, peak in spectra[1:])
    metrics['curl_peak_cadence'] = dict(spectra=[c for _, c, _ in spectra],
                                        hst_rows=len(rows))


def analyze():
    root = Path('build/src')
    try:
        check_cadence()
        seeds = {}
        for name, helicity, equal in (('plus', 1, True), ('minus', -1, True),
                                      ('zero', 0, True), ('gauss', 1, False),
                                      ('gauss_zero', 0, False), ('blocks', 1, True),
                                      ('repeat', 1, True), ('seed2', 1, True),
                                      ('mpi4', 1, True)):
            seeds[name] = check_seed(name, helicity, equal)
        seeds['dirs256'] = check_seed('dirs256', 1, ndirs=256)
        check_seed('fullshell', 1, n0=3.5)
        assert metrics['fullshell']['full_shell'] and not metrics['plus']['full_shell']
        faces = {name: seeds[name][0]['faces'] for name in seeds}
        np.testing.assert_array_equal(faces['plus'], faces['repeat'])
        # Same mode table (hash below); other decompositions differ only by the global
        # NormalizeRmsB factor, whose sum order changes: a uniform ratio 1 + O(1e-15).
        decomposition = {}
        for name in ('blocks', 'mpi4'):
            big = abs(faces['plus']) > 1e-3*0.01
            ratio = faces[name][big]/faces['plus'][big]
            decomposition[name] = float(np.max(abs(ratio-1)))
            assert decomposition[name] < 1e-14 and np.ptp(ratio) < 1e-14, ratio
        mpi_diff = decomposition['mpi4']
        assert metrics['plus']['hash'] == metrics['blocks']['hash'] == \
            metrics['mpi4']['hash'] == metrics['repeat']['hash']
        assert metrics['plus']['hash'] != metrics['seed2']['hash']
        # Same iseed: identical mode sets for every helicity and amplitude choice.
        for name in ('minus', 'zero', 'gauss'):
            np.testing.assert_array_equal(seeds['plus'][3], seeds[name][3])
        overlap = np.sum(seeds['plus'][3] & seeds['seed2'][3])/seeds['plus'][3].sum()
        assert overlap < 0.8, overlap
        # Helicity spectra: per-shell ratio to P/k and its independent FFT.
        ratios = {}
        for name in ('plus', 'minus', 'zero', 'gauss', 'gauss_zero'):
            ratios[name] = check_helicity_spectrum(name, seeds[name][4])[1]
        metrics['bcc_relative_helicity_from_spec'] = ratios
        # bcc is not exactly helical (component-dependent cell-centring transfer).
        assert ratios['plus'] > 0.99 and ratios['minus'] < -0.99
        assert abs(ratios['zero']) < 0.01 and abs(ratios['gauss_zero']) < 0.01
        prim = assemble(root/'bin/ShellSeedTest_plus.prim.00000.bin')
        assert metrics['dirs256']['modes'] < metrics['plus']['modes']
        assert all(np.all(prim[f'vel{a}'] == 0.0) for a in 'xyz')
        spec, _ = check_helicity_spectrum('plus', np.zeros((3, 64, 64, 64)), 'velocity')
        np.testing.assert_array_equal(spec[:, 1:], 0.0)
        # Kinetic helicity of the driven flow at the end of the spin-up.
        spin = read_restart(latest_restart('spinup'))
        velocity = spin['u'][1:4]/spin['u'][0]
        vspec, vrel = check_helicity_spectrum('spinup', velocity, 'velocity', -1)
        assert np.max(abs(vspec[:, 2])) > 0.0
        metrics['spinup_kinetic_relative_helicity'] = vrel
        check_decay('decay_zero')
        check_decay('decay_plus')
        inj = read_restart(latest_restart('injected'))
        assert spin['time'] == inj['time'] and spin['cycle'] == inj['cycle']
        np.testing.assert_array_equal(spin['faces'], 0.0)
        for key in ('u', 'force'):
            np.testing.assert_array_equal(spin[key], inj[key])
        assert spin['rng'] == inj['rng'] and len(spin['rng']) > 0
        assert np.max(abs(spin['u'][1:4])) > 1e-4
        np.testing.assert_array_equal(inj['faces'], faces['plus'])
        seed_block = inj['header'].split('<spectral_ic>')[1].split('<')[0]
        flag = next(line for line in seed_block.splitlines() if 'seed_on_restart' in line)
        assert flag.split('=')[1].split('#')[0].strip() in ('false', '0')
        metrics['restart_injection'] = dict(flow_forcing_rng_preserved=True,
                                            field_equals_fresh_seed=True)
        metrics['decomposition_scale_diff'] = decomposition
        logger.info('Shell seed validation: %s', metrics)
    except (AssertionError, OSError, ValueError, KeyError, IndexError) as error:
        logger.warning('Shell seed regression failed: %s', error, exc_info=True)
        return False
    for directory in (root, root/'bin', root/'rst'):
        for path in directory.glob('ShellSeedTest*'):
            if path.is_file():
                path.unlink()
    return True
