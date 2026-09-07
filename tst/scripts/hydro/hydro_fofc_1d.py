# Regression test for first-order flux correction (FOFC) in strictly 1D hydro.
#
# Each reconstruction is run both on a true 1D mesh and on an equivalent 2D mesh
# whose solution is constant in x2.  The histories must be finite and identical.

import logging

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

_INPUT = 'hydro/sod.athinput'
_RECONSTRUCTIONS = ('plm', 'wenoz', 'teno5', 'teno5_opt')


def run(**kwargs):
    logger.debug('Running test ' + __name__)

    for reconstruction in _RECONSTRUCTIONS:
        common = [
            'time/nlim=1',
            'mesh/nghost=4',
            'hydro/reconstruct=' + reconstruction,
            'hydro/fofc=true',
            'output1/dt=-1.0',
            'output2/data_format=%24.16e',
        ]

        athena.run(_INPUT, common + [
            'job/basename=hydro_fofc_' + reconstruction + '_1d',
        ])
        athena.run(_INPUT, common + [
            'job/basename=hydro_fofc_' + reconstruction + '_2d',
            'mesh/nx2=4',
            'meshblock/nx2=4',
        ])


def analyze():
    analyze_passed = True

    for reconstruction in _RECONSTRUCTIONS:
        histories = []
        for dimension in ('1d', '2d'):
            filename = ('build/src/hydro_fofc_' + reconstruction + '_' +
                        dimension + '.hydro.hst')
            history = np.loadtxt(filename)
            histories.append(history)
            if history.shape[0] < 2 or not np.all(np.isfinite(history)):
                logger.warning('%s %s FOFC history is incomplete or non-finite',
                               reconstruction, dimension)
                analyze_passed = False

        if histories[0].shape != histories[1].shape or not np.allclose(
                histories[0], histories[1], rtol=0.0, atol=1.0e-14):
            logger.warning('%s FOFC histories differ between 1D and 2D',
                           reconstruction)
            analyze_passed = False

    return analyze_passed
