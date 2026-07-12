"""Cyclic and MeshBlock-equivalence tests for three-dimensional dual CT."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for block_size in (16, 8):
        decomposition = 'single' if block_size == 16 else 'multi'
        for plane in ('xy', 'yz', 'zx'):
            basename = 'rsrmhd_ect_3d_' + decomposition + '_' + plane
            error_file = 'build/src/' + basename + '-errs.dat'
            if os.path.exists(error_file):
                os.remove(error_file)
            athena.run('tests/rsrmhd_charged_vortex_3d.athinput', [
                'job/basename=' + basename,
                'mhd/electric_ct=true',
                'problem/plane=' + plane,
                'mesh/nx1=16',
                'mesh/nx2=16',
                'mesh/nx3=16',
                'meshblock/nx1=' + repr(block_size),
                'meshblock/nx2=' + repr(block_size),
                'meshblock/nx3=' + repr(block_size),
                'time/nlim=-1',
                'time/tlim=1.0',
            ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    rows = []
    for decomposition in ('single', 'multi'):
        for plane in ('xy', 'yz', 'zx'):
            filename = ('build/src/rsrmhd_ect_3d_' + decomposition + '_'
                        + plane + '-errs.dat')
            rows.append(np.loadtxt(filename))
    data = np.asarray(rows)
    if data.shape != (6, 11) or not np.all(np.isfinite(data)):
        logger.warning('3D dual-CT diagnostics are invalid: %s', data)
        return False
    if np.any(data[:, 0] != 16) or np.any(data[:, 1] != 2) or \
            np.any(data[:, 10] != 1.0):
        logger.warning('3D dual-CT resolution, cycles, or time are wrong: %s', data)
        return False
    if np.any(data[:, 6] > 2.0e-12) or np.any(data[:, 7] > 1.0e-13) or \
            np.any(data[:, 8] > 1.0e-12) or np.any(data[:, 9] != 0.0):
        logger.warning('3D dual-CT constraints or recovery failed: %s', data[:, 6:10])
        return False

    for offset in (0, 3):
        if not np.allclose(data[offset:offset + 3, 2:6],
                           data[offset, 2:6], rtol=1.0e-11, atol=2.0e-13):
            logger.warning('3D dual CT is not cyclically invariant: %s',
                           data[offset:offset + 3, 2:6])
            return False
    if not np.allclose(data[:3, 2:6], data[3:, 2:6],
                       rtol=0.0, atol=2.0e-13):
        logger.warning('Single/multiblock 3D dual CT differs: %s', data[:, 2:6])
        return False

    # Reference after refreshing physical ghost primitives following every face-E
    # implicit stage.  This is also the state ordering required for restart equivalence.
    reference = np.array([1.0161650218613861e-2, 4.9517336431936981e-2,
                          2.4374489078990607e-3, 1.9235370757992976e-2])
    if not np.allclose(data[0, 2:6], reference, rtol=1.0e-12, atol=1.0e-14):
        logger.warning('3D dual-CT reference solution changed: %s', data[0, 2:6])
        return False
    return True
