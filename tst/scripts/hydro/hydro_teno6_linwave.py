# Regression tests for TENO6 and TENO6-opt in 3D Newtonian hydrodynamics.

import logging
import sys

import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import athena_read  # noqa

athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])

_RECONSTRUCTIONS = ('teno6', 'teno6_opt')
_FLUXES = ('llf', 'hlle', 'hllc', 'roe')
_WAVES = ('L-sound', 'R-sound', 'entropy')


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for recon in _RECONSTRUCTIONS:
        for flux in _FLUXES:
            for res in (16, 32):
                arguments = ['job/basename=hydro_teno6_lin_wave',
                             'time/tlim=1.0',
                             'time/nlim=1000',
                             'time/integrator=rk3',
                             'mesh/nghost=3',
                             'mesh/nx1=' + repr(res),
                             'mesh/nx2=' + repr(res//2),
                             'mesh/nx3=' + repr(res//2),
                             'meshblock/nx1=' + repr(res//4),
                             'meshblock/nx2=' + repr(res//4),
                             'meshblock/nx3=' + repr(res//4),
                             'hydro/reconstruct=' + recon,
                             'hydro/rsolver=' + flux,
                             'problem/amp=1.0e-6',
                             'output1/dt=-1.0',
                             'output2/dt=-1.0',
                             'output3/dt=-1.0']
                for wave_flag, vflow in ((0, 0.0), (4, 0.0), (3, 1.0)):
                    args = arguments + ['problem/wave_flag=' + repr(wave_flag),
                                        'problem/vflow=' + repr(vflow)]
                    athena.run('tests/linear_wave_hydro.athinput', args)


def analyze():
    logger.debug('Analyzing test ' + __name__)
    data = athena_read.error_dat('build/src/hydro_teno6_lin_wave-errs.dat')
    data = data.reshape([len(_RECONSTRUCTIONS), len(_FLUXES), 2,
                         len(_WAVES), data.shape[-1]])
    error_limit = (6.0e-9, 6.0e-9, 4.5e-9)
    status = True
    for ri, recon in enumerate(_RECONSTRUCTIONS):
        # Once the sixth/fifth-order spatial errors are suppressed, SSPRK(3,3)
        # time error dominates and approaches the expected 1/8 refinement ratio.
        sound_ratio_limit = 0.13 if recon == 'teno6' else 0.12
        ratio_limit = (sound_ratio_limit, sound_ratio_limit, 0.08)
        for fi, flux in enumerate(_FLUXES):
            for wi, wave in enumerate(_WAVES):
                error_n16 = data[ri][fi][0][wi][4]
                error_n32 = data[ri][fi][1][wi][4]
                ratio = error_n32/error_n16
                if error_n32 > error_limit[wi]:
                    logger.warning('%s error too large for rk3+%s+%s: %g > %g',
                                   wave, recon, flux, error_n32, error_limit[wi])
                    status = False
                if ratio > ratio_limit[wi]:
                    logger.warning('%s not converging for rk3+%s+%s: %g > %g',
                                   wave, recon, flux, ratio, ratio_limit[wi])
                    status = False
    return status
