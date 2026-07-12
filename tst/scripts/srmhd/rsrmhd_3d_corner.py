"""Three-dimensional cyclic-axis validation of resistive-SRMHD corner fields."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for plane in ('xy', 'yz', 'zx'):
        basename = 'rsrmhd_charged_vortex_3d_' + plane
        error_file = 'build/src/' + basename + '-errs.dat'
        if os.path.exists(error_file):
            os.remove(error_file)
        athena.run('tests/rsrmhd_charged_vortex_3d.athinput', [
            'job/basename=' + basename,
            'problem/plane=' + plane,
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    rows = []
    for plane in ('xy', 'yz', 'zx'):
        filename = 'build/src/rsrmhd_charged_vortex_3d_' + plane + '-errs.dat'
        rows.append(np.loadtxt(filename))
    data = np.asarray(rows)
    if data.shape != (3, 11) or not np.all(np.isfinite(data)):
        logger.warning('3D corner diagnostics are invalid: shape=%s data=%s',
                       data.shape, data)
        return False
    if np.any(data[:, 0] != 64) or np.any(data[:, 10] != 1.0):
        logger.warning('3D corner runs have wrong resolution or final time: %s', data)
        return False
    if not np.all(data[:, 1] == data[0, 1]):
        logger.warning('Cyclic 3D runs used different cycle counts: %s', data[:, 1])
        return False
    if np.any(data[:, 6] > 2.0e-11):
        logger.warning('3D compatible Gauss balance failed: %s', data[:, 6])
        return False
    if np.any(data[:, 7:10] > 2.0e-12):
        logger.warning('3D divB, symmetry, or recovery check failed: %s', data[:, 7:10])
        return False

    # Cyclically rotating (x,y,z) and (E1,E2,E3) must leave all scalar error
    # diagnostics unchanged.  This simultaneously exercises each CT edge component.
    # The integrated Gauss residual is a roundoff-level cancellation over O(N^3)
    # terms and need not be bitwise rotation invariant.  Compare the physical error
    # norms here; Gauss and divB have independent absolute gates above.
    reference = data[0, 2:6]
    if not np.allclose(data[1:, 2:6], reference, rtol=2.0e-11, atol=2.0e-13):
        logger.warning('3D corner construction is not cyclically invariant: %s',
                       data[:, 2:6])
        return False
    return True
