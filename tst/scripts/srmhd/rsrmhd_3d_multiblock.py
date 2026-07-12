"""Three-dimensional MeshBlock-interface validation for resistive SRMHD."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for plane in ('xy', 'yz', 'zx'):
        basename = 'rsrmhd_charged_vortex_3d_multiblock_' + plane
        error_file = 'build/src/' + basename + '-errs.dat'
        if os.path.exists(error_file):
            os.remove(error_file)
        athena.run('tests/rsrmhd_charged_vortex_3d.athinput', [
            'job/basename=' + basename,
            'problem/plane=' + plane,
            'meshblock/nx1=32',
            'meshblock/nx2=32',
            'meshblock/nx3=32',
        ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    rows = []
    for plane in ('xy', 'yz', 'zx'):
        filename = ('build/src/rsrmhd_charged_vortex_3d_multiblock_'
                    + plane + '-errs.dat')
        rows.append(np.loadtxt(filename))
    data = np.asarray(rows)
    if data.shape != (3, 11) or not np.all(np.isfinite(data)):
        logger.warning('3D MeshBlock diagnostics are invalid: %s', data)
        return False
    if np.any(data[:, 0] != 64) or np.any(data[:, 1] != 7):
        logger.warning('3D MeshBlock runs have wrong resolution or cycle count: %s',
                       data[:, :2])
        return False
    if np.any(data[:, 10] != 1.0):
        logger.warning('3D MeshBlock runs have wrong final time: %s', data[:, 10])
        return False
    if np.any(data[:, 6] > 2.0e-11):
        logger.warning('3D MeshBlock compatible Gauss balance failed: %s', data[:, 6])
        return False
    if np.any(data[:, 7:10] > 2.0e-12):
        logger.warning('3D MeshBlock divB, symmetry, or recovery failed: %s',
                       data[:, 7:10])
        return False

    # These are the one-MeshBlock physical error norms at the same global resolution.
    # Matching them verifies that faces, edges, and corners introduced by the 2x2x2
    # decomposition do not alter the evolved solution.
    reference = np.array([2.1626193782817325e-3, 1.0438390241497808e-1,
                          2.7744929495405313e-4, 1.2079908827251074e-2])
    if not np.allclose(data[:, 2:6], reference, rtol=2.0e-11, atol=2.0e-13):
        logger.warning('3D MeshBlock solution differs from one-block reference: %s',
                       data[:, 2:6])
        return False
    return True
