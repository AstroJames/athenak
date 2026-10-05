"""Check MPI boundary exchange against one rank on uniform, static and moving AMR.

Uses built_in_pgens, with 1/2/4 ranks in MPI builds and one process otherwise.
ATHENA_PACKING_EXECUTABLE selects a prebuilt executable;
ATHENA_PACKING_MPI_LAUNCHER supplies launcher options.
"""
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

import numpy as np

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'vis/python'))
import bin_convert  # noqa: E402

logger = logging.getLogger('athena.test.rank_packing')
_passed = False


def run(**kwargs):
    global _passed
    _passed = False
    executable = Path(os.environ.get('ATHENA_PACKING_EXECUTABLE',
                                     'build/src/athena')).resolve(strict=True)
    config = subprocess.check_output([str(executable), '-c'], text=True)
    mpi = bool(re.search(r'MPI parallelism:\s+ON', config))
    counts = (1, 2, 4) if mpi else (1,)
    if not mpi:
        logger.info('Serial coverage only; enable MPI for cross-rank comparisons')
    launcher = shlex.split(os.environ.get('ATHENA_PACKING_MPI_LAUNCHER', 'mpiexec'))
    with tempfile.TemporaryDirectory(prefix='athenak-rank-packing-') as scratch:
        for fluid in ('hydro', 'mhd'):
            for layout in ('uniform', 'smr', 'amr'):
                suffix = '' if layout == 'uniform' else '_' + layout
                text = (SOURCE / ('inputs/tests/linear_wave_' + fluid + suffix +
                                  '.athinput')).read_text().split('<output1>')[0]
                variable = 'mhd_w_bcc' if fluid == 'mhd' else 'hydro_w'
                text += ('\n<output1>\nfile_type = bin\nvariable = ' + variable +
                         '\nid = state\ndt = 100.0\n')
                reference = None
                for ranks in counts:
                    case = Path(scratch) / f'{fluid}_{layout}_{ranks}'
                    case.mkdir()
                    inp = case / 'input.athinput'
                    inp.write_text(text)
                    grid, block = ((32, 32, 16), (16, 16, 16))
                    if layout == 'smr':
                        grid, block = (64, 32, 32), (16, 8, 8)
                    elif layout == 'amr':
                        grid, block = (64, 32, 1), (8, 8, 1)
                    args = ['job/basename=packing', 'time/nlim=60', 'time/tlim=10',
                            'time/integrator=rk3', 'time/ndiag=60',
                            'problem/amp=0.001', 'problem/wave_flag=0']
                    for axis in range(1, 4):
                        args += [f'mesh/nx{axis}={grid[axis-1]}',
                                 f'meshblock/nx{axis}={block[axis-1]}']
                    prefix = launcher + ['-n', str(ranks)] if mpi else []
                    result = subprocess.run(prefix + [str(executable), '-i', str(inp)]
                                            + args,
                                            cwd=case, text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, timeout=180)
                    if result.returncode:
                        raise RuntimeError(result.stdout[-4000:])
                    assert re.search(r'cycle\s*=\s*60', result.stdout), result.stdout
                    if layout == 'amr':
                        changes = re.search(r'(\d+) MeshBlocks created, (\d+) deleted',
                                            result.stdout)
                        assert changes and sum(map(int, changes.groups())) > 0
                    paths = sorted((case / 'bin').glob('*.bin'))
                    assert len(paths) == 2, 'Require initial and final fields'
                    data = bin_convert.read_binary(str(paths[-1]))
                    logical = data['mb_logical']
                    order = np.lexsort(logical.T)
                    fields = np.stack([np.asarray(data['mb_data'][v])[order]
                                       for v in data['var_names']])
                    assert np.isfinite(fields).all()
                    errors = np.loadtxt(case / 'packing-errs.dat')
                    assert np.isfinite(errors).all()
                    if reference is None:
                        reference = logical[order], fields, errors
                    else:
                        np.testing.assert_array_equal(logical[order], reference[0])
                        # Check every float32 output cell as well as reported error norms.
                        np.testing.assert_allclose(fields, reference[1],
                                                   rtol=2e-7, atol=1e-9)
                        np.testing.assert_allclose(errors[:11], reference[2][:11],
                                                   rtol=2e-11, atol=1e-13)
                        # Existing AMR magnetic error norms vary by about 5 ppm
                        # with rank count, also reproduced without rank packing.
                        magnetic_rtol = 1e-5 if layout == 'amr' else 2e-11
                        np.testing.assert_allclose(errors[11:], reference[2][11:],
                                                   rtol=magnetic_rtol, atol=1e-13)
                    logger.info('Boundary exchange PASS: %s %s ranks=%d',
                                fluid, layout, ranks)
    _passed = True


def analyze():
    return _passed
