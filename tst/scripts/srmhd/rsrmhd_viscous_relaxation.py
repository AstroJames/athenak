"""Regression test for homogeneous Israel--Stewart shear relaxation."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

RESOLUTIONS = (32, 64, 128)
INITIAL_STRESS = np.array((0.004, -0.001, -0.003, 0.002, -0.0015, 0.01))


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for layout, electric_ct in (('cell', False), ('face', True)):
        for resolution in RESOLUTIONS:
            basename = 'rsrmhd_viscous_relaxation_{}_{}'.format(
                layout, resolution)
            filename = 'build/src/' + basename + '-errs.dat'
            if os.path.exists(filename):
                os.remove(filename)
            athena.run('tests/rsrmhd_viscous_relaxation.athinput', [
                'job/basename=' + basename,
                'mhd/electric_ct=' + str(electric_ct).lower(),
                'mesh/nx1=' + repr(resolution),
                'meshblock/nx1=' + repr(resolution),
                'problem/viscous_diagnostic_name=' + basename,
            ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    results = {}
    exact = INITIAL_STRESS*np.exp(-0.5)
    for layout in ('cell', 'face'):
        rows = []
        for resolution in RESOLUTIONS:
            name = 'rsrmhd_viscous_relaxation_{}_{}'.format(layout, resolution)
            row = np.loadtxt('build/src/' + name + '-errs.dat')
            if row.shape != (19,) or not np.all(np.isfinite(row)):
                logger.warning('%s diagnostics are invalid: %s', name, row)
                return False
            if row[0] != resolution or row[2] != 0.1 or row[3] > 2.0e-10:
                logger.warning('%s metadata/fluid state is wrong: %s', name, row[:4])
                return False
            if not np.allclose(row[4:10], exact, rtol=8.0e-3, atol=2.0e-14):
                logger.warning('%s relaxed shear state is wrong: %s', name, row[4:10])
                return False
            if np.any(row[16:19] != 0.0):
                logger.warning('%s generated a spurious electric field: %s',
                               name, row[16:19])
                return False
            rows.append(row)
        results[layout] = np.asarray(rows)
        max_errors = np.max(results[layout][:, 10:16], axis=1)
        rates = np.log2(max_errors[:-1]/max_errors[1:])
        if np.any(rates < 0.9):
            logger.warning('%s temporal relaxation convergence regressed: %s',
                           layout, rates)
            return False

    if not np.allclose(results['cell'][:, 4:10], results['face'][:, 4:10],
                       rtol=0.0, atol=5.0e-13):
        logger.warning('Cell/face relaxation states disagree: cell=%s face=%s',
                       results['cell'][:, 4:10], results['face'][:, 4:10])
        return False
    return True
