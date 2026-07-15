#!/usr/bin/env python3
"""Summarize a Zhdankin-style SRRMHD antenna calibration history."""

import argparse
import json
from pathlib import Path

import numpy as np

import athena_read


def summarize(history_path):
    history = athena_read.hst(str(history_path))
    time = np.asarray(history['time'])
    volume = np.asarray(history['volume'])
    box_length = float(volume[0]**(1.0/3.0))
    b0_squared = 2.0*history['emag'][0]/volume[0]
    reference_samples = np.flatnonzero(np.asarray(history['vA_ant']) > 0.0)
    if reference_samples.size == 0:
        raise ValueError('history has no initialized antenna reference state')
    reference_index = reference_samples[0]
    va0 = float(history['vA_ant'][reference_index])
    alfven_time = box_length/va0

    delta_b_squared = np.maximum(
        2.0*np.asarray(history['emag'])/volume - b0_squared, 0.0
    )
    delta_b_ratio = np.sqrt(delta_b_squared/b0_squared)
    velocity_rms = np.sqrt(np.asarray(history['v2'])/volume)
    alfven_speed = np.asarray(history['v_alfven'])/volume
    magnetization = np.asarray(history['sigma'])/volume
    energy_residual = (
        np.asarray(history['etot']) - history['etot'][0]
        - np.asarray(history['e_ant'])
    )

    normalization = 0.5*b0_squared*va0*box_length**2*time
    valid = time > 0.0
    injection_efficiency = np.full_like(time, np.nan)
    heating_efficiency = np.full_like(time, np.nan)
    injection_efficiency[valid] = history['e_ant'][valid]/normalization[valid]
    heating_efficiency[valid] = (
        history['eint'][valid] - history['eint'][0]
    )/normalization[valid]

    developed = time >= 4.0*alfven_time
    if not np.any(developed):
        developed = time >= 0.5*time[-1]

    return {
        'history': str(Path(history_path).resolve()),
        'samples': int(time.size),
        'final_time_over_alfven_time': float(time[-1]/alfven_time),
        'initial_alfven_speed': va0,
        'initial_magnetization': float(history['sigma_ant'][reference_index]),
        'developed_delta_b_rms_over_b0': float(np.mean(delta_b_ratio[developed])),
        'developed_v_rms_over_v_alfven': float(np.mean(
            velocity_rms[developed]/alfven_speed[developed])),
        'developed_injection_efficiency': float(np.nanmean(
            injection_efficiency[developed])),
        'developed_heating_efficiency': float(np.nanmean(
            heating_efficiency[developed])),
        'final_magnetization': float(magnetization[-1]),
        'max_relative_energy_audit_error': float(
            np.max(np.abs(energy_residual))/history['etot'][0]
        ),
        'max_discrete_current_divergence': float(
            np.max(np.abs(history['div_jant']))
        ),
        'max_face_cell_current_mismatch': float(
            np.max(np.abs(history['jant_fc_cc']))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('history', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.history)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.write_text(serialized + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
