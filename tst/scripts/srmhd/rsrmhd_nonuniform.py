"""Frozen local nonuniform-resistivity regression for both electric layouts."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    model_file = 'build/src/rsrmhd_resistivity_model-errs.dat'
    if os.path.exists(model_file):
        os.remove(model_file)
    athena.run('tests/rsrmhd_roundtrip.athinput', [])

    for electric_ct in (False, True):
        layout = 'face' if electric_ct else 'cell'
        for block_size in (16, 8):
            decomposition = 'single' if block_size == 16 else 'multi'
            basename = 'rsrmhd_nonuniform_' + layout + '_' + decomposition
            for suffix in ('-errs.dat', '-eta.dat'):
                filename = 'build/src/' + basename + suffix
                if os.path.exists(filename):
                    os.remove(filename)
            athena.run('tests/rsrmhd_charged_vortex.athinput', [
                'job/basename=' + basename,
                'mhd/electric_ct=' + str(electric_ct).lower(),
                'mhd/resistivity_model=charge_starvation',
                'mhd/eta_floor=1.0e-8',
                'mhd/eta_scale=1.0e-2',
                'mhd/number_per_mass=1.0',
                'problem/nonideal_e_scale=1.05',
                'mesh/nx1=16',
                'mesh/nx2=16',
                'meshblock/nx1=' + repr(block_size),
                'meshblock/nx2=' + repr(block_size),
                'time/nlim=1',
                'time/tlim=0.01',
                'output1/dt=-1.0',
            ])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    model = np.loadtxt('build/src/rsrmhd_resistivity_model-errs.dat')
    if model.shape != (2,) or model[0] > 2.0e-15 or model[1] != 1.0e-8:
        logger.warning('Local resistivity evaluator failed: %s', model)
        return False

    for layout in ('cell', 'face'):
        diagnostics = []
        eta_ranges = []
        for decomposition in ('single', 'multi'):
            basename = 'build/src/rsrmhd_nonuniform_' + layout + '_' + decomposition
            diagnostics.append(np.loadtxt(basename + '-errs.dat'))
            eta_ranges.append(np.loadtxt(basename + '-eta.dat'))
        diagnostics = np.asarray(diagnostics)
        eta_ranges = np.asarray(eta_ranges)
        if diagnostics.shape != (2, 11) or eta_ranges.shape != (2, 2) or \
                not np.all(np.isfinite(diagnostics)) or \
                not np.all(np.isfinite(eta_ranges)):
            logger.warning('%s nonuniform diagnostics are invalid', layout)
            return False
        if np.any(diagnostics[:, 0] != 16) or np.any(diagnostics[:, 1] != 1) or \
                np.any(diagnostics[:, 9] != 0) or np.any(diagnostics[:, 10] != 0.01):
            logger.warning('%s run metadata or recovery failed: %s', layout, diagnostics)
            return False
        if np.any(eta_ranges[:, 0] < 1.0e-8) or np.any(eta_ranges[:, 1] <= 1.0e-7):
            logger.warning('%s local eta range is not positive and nonuniform: %s',
                           layout, eta_ranges)
            return False
        if not np.allclose(diagnostics[0, 2:10], diagnostics[1, 2:10],
                           rtol=0.0, atol=2.0e-13):
            logger.warning('%s single/multiblock solutions differ: %s',
                           layout, diagnostics[:, 2:10])
            return False
        if not np.allclose(eta_ranges[0], eta_ranges[1], rtol=0.0, atol=2.0e-14):
            logger.warning('%s single/multiblock eta ranges differ: %s',
                           layout, eta_ranges)
            return False
    return True
