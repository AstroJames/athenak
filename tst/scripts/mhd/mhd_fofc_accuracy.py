"""Check that iterative FOFC preserves smooth 3D WENOZ/HLLD wave accuracy."""

import logging

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])
_LAYOUTS = ('single', 'multiblock')
_CAPS = (1, 8)


def run(**kwargs):
    for layout in _LAYOUTS:
        for cap in _CAPS:
            for resolution in (16, 32):
                block = resolution if layout == 'single' else resolution // 2
                arguments = [
                    'job/basename=mhd_fofc_wave_{}_cap{}'.format(layout, cap),
                    'time/tlim=1.0', 'time/nlim=1000',
                    'time/integrator=rk3', 'mesh/nghost=4',
                    'mesh/nx1=' + str(resolution),
                    'mesh/nx2=' + str(resolution // 2),
                    'mesh/nx3=' + str(resolution // 2),
                    'meshblock/nx1=' + str(block),
                    'meshblock/nx2=' + str(block // 2),
                    'meshblock/nx3=' + str(block // 2),
                    'mhd/reconstruct=wenoz', 'mhd/rsolver=hlld',
                    'mhd/fofc=true', 'mhd/fofc_max_iterations=' + str(cap),
                    'problem/amp=1.0e-6', 'problem/wave_flag=0',
                    'problem/vflow=0.0',
                ] + ['output{}/dt=-1.0'.format(n) for n in range(1, 6)]
                athena.run('tests/linear_wave_mhd.athinput', arguments)


def analyze():
    for layout in _LAYOUTS:
        results = []
        for cap in _CAPS:
            filename = ('build/src/mhd_fofc_wave_{}_cap{}-errs.dat'
                        .format(layout, cap))
            data = np.loadtxt(filename, ndmin=2)
            if data.shape[0] != 2 or not np.all(np.isfinite(data)):
                logger.warning('Invalid wave diagnostics in %s', filename)
                return False
            if data[1, 4] > 2.0e-8 or data[1, 4] / data[0, 4] > 0.15:
                logger.warning('FOFC wave accuracy/convergence failed: %s',
                               filename)
                return False
            results.append(data)
        if not np.array_equal(results[0], results[1]):
            logger.warning('Iterative FOFC changed a smooth wave: %s', layout)
            return False
    return True
