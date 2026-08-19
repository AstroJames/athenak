# Regression tests for the TENO6 reconstruction methods in supported MHD dimensions.

import logging
import sys

import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import athena_read  # noqa

athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])

_RECONSTRUCTIONS = ('teno6', 'teno6_opt')


def _common_arguments(basename, recon, nx1, nx2, nx3, mb1, mb2, mb3):
    return ['job/basename=' + basename,
            'time/tlim=1.0',
            'time/nlim=1000',
            'time/integrator=rk3',
            'mesh/nghost=3',
            'mesh/nx1=' + repr(nx1),
            'mesh/nx2=' + repr(nx2),
            'mesh/nx3=' + repr(nx3),
            'meshblock/nx1=' + repr(mb1),
            'meshblock/nx2=' + repr(mb2),
            'meshblock/nx3=' + repr(mb3),
            'mhd/reconstruct=' + recon,
            'mhd/rsolver=hlld',
            'problem/amp=1.0e-6',
            'problem/wave_flag=2',
            'problem/vflow=0.0',
            'output1/dt=-1.0',
            'output2/dt=-1.0',
            'output3/dt=-1.0',
            'output4/dt=-1.0',
            'output5/dt=-1.0']


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for recon in _RECONSTRUCTIONS:
        for res in (16, 32):
            args = _common_arguments('mhd_teno6_1d_lin_wave', recon,
                                     res, 1, 1, res//4, 1, 1)
            args += ['problem/along_x1=true']
            athena.run('tests/linear_wave_mhd.athinput', args)

            args = _common_arguments('mhd_teno6_2d_lin_wave', recon,
                                     res, res//2, 1, res//4, res//4, 1)
            athena.run('tests/linear_wave_mhd.athinput', args)


def _check_convergence(filename, error_limit, ratio_limit, dimension):
    data = athena_read.error_dat('build/src/' + filename)
    data = data.reshape([len(_RECONSTRUCTIONS), 2, data.shape[-1]])
    status = True
    for ri, recon in enumerate(_RECONSTRUCTIONS):
        error_n16 = data[ri][0][4]
        error_n32 = data[ri][1][4]
        ratio = error_n32/error_n16
        if error_n32 > error_limit:
            logger.warning('%s slow-wave error too large for rk3+%s+hlld: '
                           '%g > %g', dimension, recon, error_n32, error_limit)
            status = False
        if ratio > ratio_limit:
            logger.warning('%s slow wave not converging for rk3+%s+hlld: '
                           '%g > %g', dimension, recon, ratio, ratio_limit)
            status = False
    return status


def analyze():
    logger.debug('Analyzing test ' + __name__)
    one_d = _check_convergence('mhd_teno6_1d_lin_wave-errs.dat',
                               1.0e-11, 0.05, '1D')
    # The multidimensional constrained-transport operator limits the coarse-grid
    # convergence rate; this check is intentionally separate from the historical
    # 3D matrix so none of its established thresholds are weakened.
    two_d = _check_convergence('mhd_teno6_2d_lin_wave-errs.dat',
                               1.6e-8, 0.24, '2D')
    return one_d and two_d
