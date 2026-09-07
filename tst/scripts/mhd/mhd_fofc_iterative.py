"""Production-kernel FOFC cascades, MPI boundaries, and bounded failure tests."""

import logging
import os
from pathlib import Path
import resource
import shlex
import subprocess

logger = logging.getLogger('athena' + __name__[7:])
_MODES = ('legacy', 'cascade', 'energy', 'nan', 'soft_floor', 'mask3d',
          'rk3', 'deep', 'limit', 'invalid', 'cfl3d_safe', 'cfl3d_unsafe')
_results = []


def run(**kwargs):
    """Use the normal Makefile harness, or a prebuilt test executable.

    ATHENA_FOFC_TEST_RANKS=1,2,8 enables MPI variants in an MPI build.
    ATHENA_FOFC_MPI_LAUNCHER can supply site-specific launcher options.
    ATHENA_FOFC_TEST_EXECUTABLE supports read-only source/build installations.
    """
    _results.clear()
    source = Path(__file__).resolve().parents[3]
    build = Path('build').resolve()
    override = os.environ.get('ATHENA_FOFC_TEST_EXECUTABLE')
    executable = Path(override) if override else build / 'src/fofc_cascade'
    if not override:
        subprocess.run(['bash', str(source / 'tst/unit/build_fofc_test.sh'),
                        str(build)], check=True)
    executable = executable.resolve(strict=True)
    ranks = [int(n) for n in
             os.environ.get('ATHENA_FOFC_TEST_RANKS', '1').split(',')]
    if any(n not in (1, 2, 4, 8) for n in ranks):
        raise ValueError('FOFC fixtures support 1, 2, 4, or 8 ranks')
    launcher = shlex.split(
        os.environ.get('ATHENA_FOFC_MPI_LAUNCHER', 'mpiexec'))
    # Expected aborts are part of the test, and must not produce core dumps.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for count in ranks:
        prefix = launcher + ['-n', str(count)] if count > 1 else []
        for mode in _MODES:
            result = subprocess.run(prefix + [str(executable), mode],
                                    cwd=build / 'src', universal_newlines=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=120)
            if mode in ('limit', 'invalid', 'cfl3d_unsafe'):
                passed = (result.returncode != 0 and
                          'FOFC_EXHAUSTED' in result.stdout)
            else:
                marker = ('FOFC_TEST mode={} ranks={} errors=0'
                          .format(mode, count))
                passed = result.returncode == 0 and marker in result.stdout
            _results.append(passed)
            logger.info('FOFC mode=%s ranks=%d passed=%s', mode, count, passed)
            if not passed:
                logger.warning('%s', result.stdout)


def analyze():
    return bool(_results) and all(_results)
