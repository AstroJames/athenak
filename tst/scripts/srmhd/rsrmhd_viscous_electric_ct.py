"""Direct FC-E/CC-E regression for coupled resistive-viscous SRMHD."""

import logging
import os

import numpy as np
import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])

RESOLUTIONS = (32, 64, 128)


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for layout, electric_ct in (('cell', False), ('face', True)):
        basename = 'rsrmhd_viscous_ect_uniform_' + layout
        filename = 'build/src/' + basename + '-errs.dat'
        if os.path.exists(filename):
            os.remove(filename)
        athena.run('tests/rsrmhd_viscous_relaxation.athinput', [
            'job/basename=' + basename,
            'mhd/electric_ct=' + str(electric_ct).lower(),
            'mesh/nx1=32',
            'meshblock/nx1=32',
            'problem/background_b1=0.7',
            'problem/background_b2=-0.3',
            'problem/background_b3=0.5',
            'problem/background_e1=0.2',
            'problem/background_e2=-0.15',
            'problem/background_e3=0.1',
            'problem/viscous_diagnostic_name=' + basename,
        ])
    for layout, electric_ct in (('cell', False), ('face', True)):
        for resolution in RESOLUTIONS:
            basename = 'rsrmhd_viscous_ect_{}_{}'.format(layout, resolution)
            for suffix in ('-errs.dat', '-state.dat'):
                filename = 'build/src/' + basename + suffix
                if os.path.exists(filename):
                    os.remove(filename)
            athena.run('tests/rsrmhd_charged_vortex.athinput', [
                'job/basename=' + basename,
                'mhd/electric_ct=' + str(electric_ct).lower(),
                'mhd/relativistic_viscosity=true',
                'mesh/nx1=' + repr(resolution),
                'mesh/nx2=' + repr(resolution),
                'meshblock/nx1=' + repr(resolution),
                'meshblock/nx2=' + repr(resolution),
                'time/nlim=-1',
                'time/tlim=0.5',
                'problem/viscous_state_name=' + basename,
                'output1/dt=-1.0',
            ])


def _relative_l2(difference, reference):
    return np.linalg.norm(difference.ravel())/np.linalg.norm(reference.ravel())


def analyze():
    logger.debug('Analyzing test ' + __name__)
    uniform = {}
    for layout in ('cell', 'face'):
        basename = 'build/src/rsrmhd_viscous_ect_uniform_' + layout
        uniform[layout] = np.loadtxt(basename + '-errs.dat')
        if uniform[layout].shape != (19,) or \
                not np.all(np.isfinite(uniform[layout])):
            logger.warning('%s diagnostics are invalid: %s', basename,
                           uniform[layout])
            return False
    if not np.allclose(uniform['cell'][4:10], uniform['face'][4:10],
                       rtol=0.0, atol=2.0e-9) or \
            not np.allclose(uniform['cell'][16:19], uniform['face'][16:19],
                            rtol=0.0, atol=2.0e-7):
        logger.warning('Uniform FC-E/CC-E coupled states disagree: cell=%s face=%s',
                       uniform['cell'], uniform['face'])
        return False

    electric_errors = []
    velocity_errors = []
    shear_errors = []
    component_errors = []
    for resolution in RESOLUTIONS:
        states = {}
        for layout in ('cell', 'face'):
            basename = 'build/src/rsrmhd_viscous_ect_{}_{}'.format(
                layout, resolution)
            state = np.loadtxt(basename + '-state.dat')
            errors = np.loadtxt(basename + '-errs.dat', ndmin=2)
            if state.shape != (resolution*resolution, 17) or \
                    not np.all(np.isfinite(state)) or errors.shape != (1, 11) or \
                    not np.all(np.isfinite(errors)):
                logger.warning('%s diagnostics are invalid', basename)
                return False
            if errors[0, 0] != resolution or errors[0, 9] != 0 or \
                    errors[0, 10] != 0.5:
                logger.warning('%s metadata/recovery is wrong: %s', basename, errors)
                return False
            order = np.lexsort((state[:, 0], state[:, 1], state[:, 2]))
            states[layout] = state[order]

        cell = states['cell']
        face = states['face']
        if not np.array_equal(cell[:, :3], face[:, :3]):
            logger.warning('Cell/face coordinates disagree at N=%d', resolution)
            return False
        difference = face - cell
        velocity_errors.append(_relative_l2(difference[:, 5:8], cell[:, 5:8]))
        electric_errors.append(_relative_l2(difference[:, 8:11], cell[:, 8:11]))
        shear_errors.append(_relative_l2(difference[:, 11:17], cell[:, 11:17]))
        component_errors.append(np.linalg.norm(difference[:, 11:17], axis=0)
                                / np.maximum(np.linalg.norm(cell[:, 11:17], axis=0),
                                             1.0e-300))

    velocity_errors = np.asarray(velocity_errors)
    electric_errors = np.asarray(electric_errors)
    shear_errors = np.asarray(shear_errors)
    component_errors = np.asarray(component_errors)
    velocity_rates = np.log2(velocity_errors[:-1]/velocity_errors[1:])
    electric_rates = np.log2(electric_errors[:-1]/electric_errors[1:])
    shear_rates = np.log2(shear_errors[:-1]/shear_errors[1:])
    if velocity_errors[-1] > 4.0e-3 or electric_errors[-1] > 7.0e-3 or \
            shear_errors[-1] > 8.0e-1:
        logger.warning('FC-E/CC-E final differences are too large: v=%s E=%s pi=%s',
                       velocity_errors, electric_errors, shear_errors)
        return False
    if velocity_rates[-1] < 1.5 or electric_rates[-1] < 1.5 or \
            shear_rates[-1] < 1.5:
        logger.warning('FC-E/CC-E convergence regressed: v=%s E=%s pi=%s',
                       velocity_rates, electric_rates, shear_rates)
        return False
    if not np.all(np.isfinite(component_errors)):
        logger.warning('One or more shear-component comparisons are non-finite: %s',
                       component_errors)
        return False
    return True
