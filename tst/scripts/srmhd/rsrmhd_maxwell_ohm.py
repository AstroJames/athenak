"""Forced-equilibrium diagnostic using a linear Maxwell--Ohm slow mode."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_DT = 1.0/128.0
_H_VALUES = (1.0, 10.0, 60.0)
_INTEGRATORS = ('imex3', 'imex2', 'imex3_ars443')


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for integrator in _INTEGRATORS:
        for h in _H_VALUES:
            basename = 'rsrmhd_maxwell_ohm_{}_h{}'.format(
                integrator, int(h))
            filename = 'build/src/' + basename + '-errs.dat'
            if os.path.exists(filename):
                os.remove(filename)
            athena.run('tests/rsrmhd_maxwell_ohm.athinput', [
                'job/basename=' + basename,
                'time/integrator=' + integrator,
                'mhd/resistivity=' + repr(_DT/h),
            ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    passed = True
    electric_ratios = {}
    for integrator in _INTEGRATORS:
        electric_ratios[integrator] = {}
        for h in _H_VALUES:
            basename = 'rsrmhd_maxwell_ohm_{}_h{}'.format(
                integrator, int(h))
            row = np.loadtxt('build/src/' + basename + '-errs.dat')
            if row.shape != (18,) or not np.all(np.isfinite(row)):
                logger.warning('%s diagnostics are invalid: %s', basename, row)
                return False
            nx1, ncycle, time, eta, measured_h = row[:5]
            failures = row[17]
            if (nx1 != 64 or ncycle != 1 or abs(time - _DT) > 1.0e-14
                    or abs(eta - _DT/h) > 1.0e-16
                    or abs(measured_h - h) > 1.0e-12 or failures != 0):
                logger.warning('%s metadata are wrong: %s', basename, row)
                return False
            b_ratio, e_ratio = row[11:13]
            electric_ratios[integrator][h] = e_ratio
            if abs(b_ratio - 1.0) > 2.0e-4:
                logger.warning('%s magnetic amplitude misses the slow mode: %s',
                               basename, b_ratio)
                passed = False
            if max(row[14:17]) > 2.0e-8:
                logger.warning('%s fluid background is not linear/passive: %s',
                               basename, row[14:17])
                passed = False
            if integrator == 'imex3_ars443' and abs(e_ratio - 1.0) > 0.05:
                logger.warning('%s accepted E is off the Maxwell--Ohm slow mode: %s',
                               basename, e_ratio)
                passed = False

    logger.warning('Accepted electric-field ratios: %s', electric_ratios)
    return passed
