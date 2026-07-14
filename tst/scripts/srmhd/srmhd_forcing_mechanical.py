"""One-step runtime regression for relativistic mechanical forcing."""

import glob
import logging
import sys

import numpy as np

import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import athena_read  # noqa

logger = logging.getLogger('athena' + __name__[7:])


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.run('tests/srmhd_forcing_mechanical.athinput', [])
    athena.run('tests/rsrmhd_forcing_mechanical_coupled.athinput', [])


def analyze():
    logger.debug('Analyzing test ' + __name__)
    primitive_files = sorted(glob.glob(
        'build/src/tab/srmhd_forcing_mechanical.primitive.*.tab'))
    force_files = sorted(glob.glob(
        'build/src/tab/srmhd_forcing_mechanical.force.*.tab'))
    if len(primitive_files) != 2 or len(force_files) != 2:
        logger.warning('Expected initial and final primitive/force tables: %s %s',
                       primitive_files, force_files)
        return False

    initial = athena_read.tab(primitive_files[0])
    final = athena_read.tab(primitive_files[-1])
    final_force = athena_read.tab(force_files[-1])
    primitive_names = ('dens', 'velx', 'vely', 'velz', 'eint',
                       'bcc1', 'bcc2', 'bcc3')
    force_names = ('force1', 'force2', 'force3')
    if not all(name in initial and name in final for name in primitive_names):
        logger.warning('Mechanical-forcing primitive labels are missing')
        return False
    if not all(name in final_force for name in force_names):
        logger.warning('Mechanical-forcing force labels are missing')
        return False

    values = [initial[name] for name in primitive_names]
    values += [final[name] for name in primitive_names]
    values += [final_force[name] for name in force_names]
    if not all(np.all(np.isfinite(value)) for value in values):
        logger.warning('Mechanical-forcing run produced non-finite output')
        return False

    force_norm = np.sqrt(sum(final_force[name]**2 for name in force_names))
    if np.max(force_norm) < 1.0e-8:
        logger.warning('Mechanical-forcing field was not initialized')
        return False

    gamma = 5.0/3.0
    enthalpy = final['dens'] + gamma*final['eint']
    acceleration = np.stack([final_force[name] for name in force_names])
    weighted_mean = np.sum(enthalpy*acceleration, axis=1)/np.sum(enthalpy)
    measured_rms = np.sqrt(
        np.sum(enthalpy*np.sum(acceleration**2, axis=0))/np.sum(enthalpy))
    target_rms = 0.05
    if np.max(np.abs(weighted_mean)) > 1.0e-10:
        logger.warning('Enthalpy-weighted acceleration mean is not zero: %s',
                       weighted_mean)
        return False
    if abs(measured_rms/target_rms - 1.0) > 1.0e-10:
        logger.warning('Mechanical acceleration RMS is wrong: %s %s',
                       measured_rms, target_rms)
        return False

    velocity_change = np.sqrt(sum(
        (final[name] - initial[name])**2
        for name in ('velx', 'vely', 'velz')))
    if np.max(velocity_change) < 1.0e-8:
        logger.warning('Mechanical forcing did not change the fluid velocity')
        return False
    if np.min(final['dens']) <= 0.0 or np.min(final['eint']) <= 0.0:
        logger.warning('Mechanical forcing produced a nonphysical primitive state')
        return False
    magnetic_max = max(np.max(np.abs(final[name]))
                       for name in ('bcc1', 'bcc2', 'bcc3'))
    if magnetic_max > 1.0e-14:
        logger.warning('Mechanical forcing directly changed the magnetic field')
        return False

    coupled = np.loadtxt('build/src/rsrmhd_roundtrip-errs.dat')
    if not np.all(np.isfinite(coupled)):
        logger.warning('Coupled forcing run produced non-finite diagnostics: %s',
                       coupled)
        return False
    counters = coupled[2:6]
    if np.any(counters != 0):
        logger.warning('Coupled forcing used EOS corrections or failed recovery: %s',
                       counters)
        return False
    if int(coupled[6]) != 8:
        logger.warning('Coupled forcing did not retain the eight-component MHD state')
        return False
    return True
