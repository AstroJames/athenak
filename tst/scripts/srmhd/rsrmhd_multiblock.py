"""MeshBlock-interface equivalence for the resistive-SRMHD current sheet."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for name, block_size in (('single', 512), ('multi', 128)):
        basename = 'rsrmhd_current_sheet_' + name + 'block'
        for suffix in ('-errs.dat', '-profile.dat'):
            filename = 'build/src/' + basename + suffix
            if os.path.exists(filename):
                os.remove(filename)
        athena.run('tests/rsrmhd_current_sheet.athinput', [
            'job/basename=' + basename,
            'mesh/nx1=512',
            'meshblock/nx1=' + repr(block_size),
            'time/tlim=1.0',
            'output1/dt=-1.0',
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    single = np.loadtxt('build/src/rsrmhd_current_sheet_singleblock-errs.dat')
    multi = np.loadtxt('build/src/rsrmhd_current_sheet_multiblock-errs.dat')
    if not np.all(np.isfinite((single, multi))):
        logger.warning('MeshBlock diagnostics contain non-finite values')
        return False
    if single[0] != 512 or multi[0] != 512:
        logger.warning('MeshBlock diagnostics do not report the global resolution')
        return False
    if single[1] != 342 or multi[1] != 342 or single[9] != 1.0 or multi[9] != 1.0:
        logger.warning('MeshBlock runs have wrong cycle counts or final times')
        return False
    if single[6] != 0 or multi[6] != 0:
        logger.warning('MeshBlock run had primitive-recovery failures')
        return False
    if not np.allclose(single[2:9], multi[2:9], rtol=0.0, atol=2.0e-14):
        logger.warning('Single- and four-MeshBlock norms differ: %s %s',
                       single[2:9], multi[2:9])
        return False

    single_profile = np.loadtxt(
        'build/src/rsrmhd_current_sheet_singleblock-profile.dat')
    multi_profile = np.loadtxt(
        'build/src/rsrmhd_current_sheet_multiblock-profile.dat')
    if single_profile.shape != (512, 5) or multi_profile.shape != (512, 5):
        logger.warning('MeshBlock profiles have wrong shapes')
        return False
    difference = np.max(np.abs(single_profile - multi_profile), axis=0)
    if np.any(difference > 2.0e-14):
        logger.warning('MeshBlock-interface profile mismatch: %s', difference)
        return False
    return True
