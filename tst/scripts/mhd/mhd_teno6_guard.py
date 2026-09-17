# Regression test for the Hydro-only TENO6 safety guard.

import logging
import os
import subprocess

import scripts.utils.athena as athena  # noqa: F401

logger = logging.getLogger('athena' + __name__[7:])

_RECONSTRUCTIONS = ('teno6', 'teno6_opt')
_results = []


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    del _results[:]
    executable = os.path.join(os.getcwd(), 'build', 'src', 'athena')
    input_file = os.path.join(os.getcwd(), '..', 'inputs', 'tests',
                              'linear_wave_mhd.athinput')
    for reconstruction in _RECONSTRUCTIONS:
        command = [executable, '-i', input_file,
                   'mesh/nghost=3',
                   'mhd/reconstruct=' + reconstruction]
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        message = result.stdout
        rejected = (result.returncode != 0 and
                    'currently supported only for hydrodynamics' in message and
                    'disabled for all MHD/RMHD calculations' in message)
        _results.append(rejected)
        if not rejected:
            logger.warning('%s was not rejected with the Hydro-only support message',
                           reconstruction)


def analyze():
    logger.debug('Analyzing test ' + __name__)
    return len(_results) == len(_RECONSTRUCTIONS) and all(_results)
