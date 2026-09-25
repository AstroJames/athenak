"""Check zero-scalar isothermal MHD and ideal-gas controls with built_in_pgens.

Run with Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON to catch invalid IEN accesses
deterministically. Adding a passive scalar must not change the physical solution
or timestep. Reuse the existing linear-wave input and binary-output reader.
"""
import logging
import os
from pathlib import Path

import numpy as np
import scripts.utils.athena as athena
from scripts.mhd.curl_outputs import assemble

logger = logging.getLogger('athena' + __name__[7:])
REPO = Path(__file__).resolve().parents[3]
PREFIX = 'MHDIsothermal'


def run(**kwargs):
    text = (REPO / 'inputs/tests/linear_wave_mhd.athinput').read_text()
    text = text.split('<output1>')[0]
    text = text.replace('<mhd>', '<mhd>\niso_sound_speed = 1.0\nnscalars = 0')
    text += ('\n<output1>\nfile_type = bin\nvariable = mhd_w_bcc\n'
             'id = prim\ndt = 1.0\n'
             '\n<output2>\nfile_type = hst\ndt = 1.0\ndata_format = %24.16e\n')
    path = Path('build/src/MHDIsothermal.athinput').resolve()
    path.write_text(text)
    for eos in ('isothermal', 'ideal'):
        for nscalars in (0, 1):
            args = [f'job/basename={PREFIX}_{eos}_{nscalars}',
                    f'mhd/eos={eos}', f'mhd/nscalars={nscalars}',
                    'time/nlim=4', 'time/integrator=rk3', 'mesh/nghost=4',
                    'mhd/reconstruct=teno5', 'mhd/rsolver=hlld', 'mhd/fofc=true',
                    'problem/amp=0.01', 'problem/vflow=0.2']
            for axis in (1, 2, 3):
                args += [f'mesh/nx{axis}=16', f'meshblock/nx{axis}=8']
            athena.run(os.path.relpath(path, REPO / 'inputs'), args)
    path.unlink()


def analyze():
    root = Path('build/src')
    try:
        for eos in ('isothermal', 'ideal'):
            fields, histories = [], []
            for nscalars in (0, 1):
                name = f'{PREFIX}_{eos}_{nscalars}'
                paths = sorted((root / 'bin').glob(f'{name}.prim.*.bin'))
                assert len(paths) == 2, (name, 'initial and final fields required')
                final = assemble(paths[-1])
                assert all(np.all(np.isfinite(a)) for a in final.values()), name
                assert np.min(final['dens']) > 0.0, name
                assert sum(k.startswith('s_') for k in final) == nscalars, name
                assert ('eint' in final) == (eos == 'ideal'), name
                fields.append(final)
                history = np.loadtxt(root / f'{name}.mhd.hst', ndmin=2)
                assert len(history) >= 2 and np.all(np.isfinite(history)), name
                assert history[-1, 0] > history[0, 0], name
                assert np.all(history[:, 1] > 0.0), name
                histories.append(history)
            for key, value in fields[0].items():
                np.testing.assert_array_equal(value, fields[1][key], err_msg=eos)
            # Standard MHD history columns precede any passive-scalar columns.
            np.testing.assert_array_equal(histories[0],
                                           histories[1][:, :histories[0].shape[1]],
                                           err_msg=eos)
    except (AssertionError, OSError, ValueError, KeyError) as error:
        logger.warning('Zero-scalar MHD regression failed: %s', error, exc_info=True)
        return False
    for directory in (root, root / 'bin'):
        for path in directory.glob(PREFIX + '*'):
            if path.is_file():
                path.unlink()
    return True
