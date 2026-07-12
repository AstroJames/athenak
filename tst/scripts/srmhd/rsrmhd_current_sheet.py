"""Self-similar resistive-SRMHD current-sheet diffusion regression."""

import logging
import glob
import os
import sys
import numpy as np
import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import athena_read  # noqa

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    error_file = 'build/src/rsrmhd_current_sheet-errs.dat'
    if os.path.exists(error_file):
        os.remove(error_file)
    for resolution in (128, 256, 512):
        output_dt = '9.0' if resolution == 128 else '-1.0'
        athena.run('tests/rsrmhd_current_sheet.athinput', [
            'mesh/nx1=' + repr(resolution),
            'meshblock/nx1=' + repr(resolution),
            'output1/dt=' + output_dt,
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = np.loadtxt('build/src/rsrmhd_current_sheet-errs.dat', ndmin=2)
    if not np.all(np.isfinite(data)):
        logger.warning('Current-sheet test produced non-finite diagnostics: %s', data)
        return False
    if data.shape != (3, 10):
        logger.warning('Current-sheet error table has unexpected shape: %s', data.shape)
        return False
    nx1 = data[:, 0]
    ncycle = data[:, 1]
    l1_b = data[:, 2]
    linf_b = data[:, 3]
    l1_e = data[:, 4]
    linf_e = data[:, 5]
    failures = data[:, 6]
    dtnew = data[:, 8]
    final_time = data[:, 9]
    if not np.array_equal(nx1, (128, 256, 512)):
        logger.warning('Current-sheet resolutions are wrong: %s', nx1)
        return False
    if not np.array_equal(ncycle, (768, 1536, 3072)):
        logger.warning('Current-sheet cycle counts are wrong: %s', ncycle)
        return False
    if np.any(failures != 0):
        logger.warning('Current-sheet implicit recovery failures: %s', failures)
        return False
    if np.any(np.abs(final_time - 9.0) > 1.0e-12):
        logger.warning('Current-sheet runs ended at the wrong times: %s', final_time)
        return False
    dx1 = 3.0 / nx1
    if np.any(dtnew > dx1 * (1.0 + 2.0e-13)):
        logger.warning('Current-sheet timestep violates the Maxwell light bound')
        return False

    rates_b = np.log2(l1_b[:-1] / l1_b[1:])
    rates_e = np.log2(l1_e[:-1] / l1_e[1:])
    if np.any(rates_b < (0.40, 0.60)) or np.any(rates_e < (0.70, 0.80)):
        logger.warning('Current-sheet convergence rates are too low: B=%s E=%s',
                       rates_b, rates_e)
        return False
    if l1_b[-1] > 3.5e-3 or linf_b[-1] > 7.0e-3:
        logger.warning('Resolved magnetic errors are too large: %s %s',
                       l1_b[-1], linf_b[-1])
        return False
    if l1_e[-1] > 2.5e-4 or linf_e[-1] > 4.5e-4:
        logger.warning('Resolved electric errors are too large: %s %s',
                       l1_e[-1], linf_e[-1])
        return False

    output_files = sorted(glob.glob('build/src/tab/'
                                    'rsrmhd_current_sheet.electric.*.tab'))
    if not output_files:
        logger.warning('Named resistive electric-field output was not written')
        return False
    electric = athena_read.tab(output_files[-1])
    if not all(name in electric for name in ('e1', 'e2', 'e3')):
        logger.warning('Electric-field output labels are missing: %s', electric.keys())
        return False
    if np.max(np.abs(electric['e1'])) > 1.0e-14 or \
            np.max(np.abs(electric['e2'])) > 1.0e-14:
        logger.warning('Nominally zero electric components are nonzero')
        return False
    analytic_time = 1.0 + electric['time']
    e3_exact = np.sqrt(0.01 / (np.pi * analytic_time)) * np.exp(
        -electric['x1v']**2 / (4.0 * 0.01 * analytic_time))
    if np.mean(np.abs(electric['e3'] - e3_exact)) > 8.0e-4:
        logger.warning('Named E3 output does not match the evolved sheet')
        return False
    return True
