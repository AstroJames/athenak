"""Pressure-balanced uniform/nonuniform SRRMHD reconnection regression."""

import glob
import logging
import os
import sys

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import bin_convert  # noqa

logger = logging.getLogger('athena' + __name__[7:])


def _remove_outputs(basename):
    patterns = [
        'build/src/' + basename + '.*.hst',
        'build/src/bin/' + basename + '.*.bin',
    ]
    for pattern in patterns:
        for filename in glob.glob(pattern):
            os.remove(filename)


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for model in ('uniform', 'nonuniform'):
        basename = 'rsrmhd_reconnection_' + model
        _remove_outputs(basename)
        arguments = ['job/basename=' + basename]
        if model == 'nonuniform':
            arguments.append('mhd/resistivity_model=charge_starvation')
        athena.run('tests/rsrmhd_reconnection.athinput', arguments)


def _initial_condition_error(path):
    data = bin_convert.read_binary(path)
    required = ('dens', 'velx', 'vely', 'velz', 'eint',
                'bcc1', 'bcc2', 'bcc3')
    if not all(name in data['mb_data'] for name in required):
        return np.inf

    maximum_error = 0.0
    for m in range(data['n_mbs']):
        x1min, x1max, x2min, x2max, _, _ = data['mb_geometry'][m]
        dx1 = (x1max - x1min)/data['nx1_mb']
        dx2 = (x2max - x2min)/data['nx2_mb']
        x = x1min + dx1*(np.arange(data['nx1_mb']) + 0.5)
        y = x2min + dx2*(np.arange(data['nx2_mb']) + 0.5)
        xx, yy = np.meshgrid(x, y)
        tanh_sheet = np.tanh(yy/0.02)
        sech2 = 1.0 - tanh_sheet**2
        along = np.tanh(200.0*xx - 10.0) + np.tanh(-10.0 - 200.0*xx)
        across = np.tanh(200.0*yy + 2.0) + np.tanh(2.0 - 200.0*yy)
        pinch = 1.0 + 0.15*along*across
        expected = {
            'dens': 0.1*(1.0 + 3.0*sech2),
            'eint': 3.0*(1.0e-3 + 0.5*sech2*pinch),
            'bcc1': tanh_sheet,
        }
        for name, values in expected.items():
            actual = data['mb_data'][name][m][0]
            maximum_error = max(maximum_error,
                                float(np.max(np.abs(actual - values))))
        for name in ('velx', 'vely', 'velz', 'bcc2', 'bcc3'):
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(data['mb_data'][name][m][0]))))
    return maximum_error


def _read_history(path):
    values = np.loadtxt(path)
    return np.atleast_2d(values)


def analyze():
    logger.debug('Analyzing test ' + __name__)
    for model in ('uniform', 'nonuniform'):
        basename = 'rsrmhd_reconnection_' + model
        primitive = sorted(glob.glob(
            'build/src/bin/' + basename + '.primitive.*.bin'))
        electric = sorted(glob.glob(
            'build/src/bin/' + basename + '.electric.*.bin'))
        eta_files = sorted(glob.glob(
            'build/src/bin/' + basename + '.eta.*.bin'))
        if len(primitive) != 2 or len(electric) != 2 or len(eta_files) != 2:
            logger.warning('%s outputs are incomplete', model)
            return False
        error = _initial_condition_error(primitive[0])
        if error > 2.0e-6:
            logger.warning('%s Harris-sheet initialization error: %s', model, error)
            return False

        initial_e = np.asarray(
            bin_convert.read_binary(electric[0])['mb_data']['e3'])
        if np.max(np.abs(initial_e)) > 1.0e-14:
            logger.warning('%s initial electric field is nonzero', model)
            return False
        initial_eta = np.asarray(
            bin_convert.read_binary(eta_files[0])['mb_data']['eta'])
        final_eta = np.asarray(
            bin_convert.read_binary(eta_files[1])['mb_data']['eta'])
        if not np.all(np.isfinite(final_eta)):
            logger.warning('%s final resistivity is invalid', model)
            return False
        if model == 'uniform':
            if np.max(np.abs(initial_eta - 1.0e-3)) > 1.0e-9 or \
                    np.max(np.abs(final_eta - 1.0e-3)) > 1.0e-9:
                logger.warning('Uniform resistivity is not constant')
                return False
        else:
            if np.max(np.abs(initial_eta - 1.0e-8)) > 1.0e-14 or \
                    np.max(final_eta) <= 1.0e-5 or np.min(final_eta) < 9.9e-9:
                logger.warning('Charge-starvation eta did not localize: %s %s',
                               np.min(final_eta), np.max(final_eta))
                return False

        user = _read_history('build/src/' + basename + '.user.hst')
        mhd = _read_history('build/src/' + basename + '.mhd.hst')
        if user.shape != (2, 10) or mhd.shape[0] != 2 or \
                not np.all(np.isfinite(user)) or not np.all(np.isfinite(mhd)):
            logger.warning('%s history diagnostics are invalid', model)
            return False
        if user[-1, 8] < 0.099 or user[-1, 8] > 0.101:
            logger.warning('%s upstream density changed unexpectedly', model)
            return False
        for column in (2, 6):
            scale = max(abs(mhd[0, column]), 1.0)
            if abs(mhd[-1, column] - mhd[0, column])/scale > 2.0e-11:
                logger.warning('%s conserved history drifted: column %s',
                               model, column)
                return False
    return True
