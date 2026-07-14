#!/usr/bin/env python3
"""Summarize integral budgets from a driven SRRMHD turbulence history."""

import argparse
import glob
import json
import os
import re

import numpy as np


def history_columns(path):
    """Return Athena history data and a label-to-column map."""
    labels = {}
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            if not line.startswith('#'):
                break
            for number, label in re.findall(r'\[(\d+)\]=(\S+)', line):
                labels[label] = int(number) - 1
    if 'magnetizat' in labels:
        labels['sigma'] = labels['magnetizat']
    if 'alfven_spe' in labels:
        labels['v_alfven'] = labels['alfven_spe']
    return np.atleast_2d(np.loadtxt(path)), labels


def trapezoid(values, times):
    """Use the current NumPy trapezoid spelling with an older-version fallback."""
    integrate = getattr(np, 'trapezoid', np.trapz)
    return float(integrate(values, times))


def linear_slope(values, times):
    """Return the least-squares slope with time measured in turnover times."""
    return float(np.polyfit(times, values, 1)[0])


def summarize(args):
    history_path = os.path.join(args.data_dir, args.history)
    history, column = history_columns(history_path)

    def values(label):
        return history[:, column[label]]

    time = values('time')
    volume = values('volume')
    vrms = np.sqrt(values('v2')/volume)
    mach = np.sqrt(values('mach2')/volume)
    reynolds = vrms*args.drive_scale/args.viscosity
    magnetic_reynolds = vrms*args.drive_scale/args.resistivity
    target_vrms = args.drive_scale/args.eddy_time
    mean_magnetization = values('sigma')/volume
    mean_alfven_speed = values('v_alfven')/volume
    target_alfven_mach = target_vrms/mean_alfven_speed
    post = time >= args.eddy_time
    turnover_time = time/args.eddy_time
    second_half = turnover_time >= 0.5*turnover_time[-1]
    q_ohm = values('q_ohm')
    q_visc = values('q_visc')
    mean_pressure = values('pgas')/volume
    ohmic_integral = trapezoid(q_ohm, time)
    viscous_integral = trapezoid(q_visc, time)
    heating_integral = ohmic_integral + viscous_integral
    cooling_enabled = 'e_cool' in column
    cooling_power = (values('cool_power') if cooling_enabled
                     else np.zeros_like(time))
    cooled_energy = (values('e_cool') if cooling_enabled
                     else np.zeros_like(time))
    cooled_momentum = np.column_stack([
        values('p_cool1') if cooling_enabled else np.zeros_like(time),
        values('p_cool2') if cooling_enabled else np.zeros_like(time),
        values('p_cool3') if cooling_enabled else np.zeros_like(time),
    ])
    limited_cooling_energy = (values('e_cool_lim') if cooling_enabled
                              else np.zeros_like(time))

    energy_residual = (
        values('etot') - values('etot')[0] - values('e_inj')
        + cooled_energy
    )
    momentum = np.column_stack([
        values('mom1') - values('mom1')[0] - values('p_inj1'),
        values('mom2') - values('mom2')[0] - values('p_inj2'),
        values('mom3') - values('mom3')[0] - values('p_inj3'),
    ]) + cooled_momentum

    turnover = []
    nturn = int(round(time[-1]/args.eddy_time))
    for index in range(nturn):
        selected = ((time >= index*args.eddy_time)
                    & (time <= (index + 1)*args.eddy_time + 1.0e-12))
        turnover.append({
            'turnover': index + 1,
            'mach_mean': float(np.mean(mach[selected])),
            'vrms_mean': float(np.mean(vrms[selected])),
            're_mean': float(np.mean(reynolds[selected])),
            'rm_mean': float(np.mean(magnetic_reynolds[selected])),
        })

    diagnostic_files = sorted(glob.glob(
        os.path.join(args.data_dir, '*-forcing.dat')))
    final_diagnostic = np.loadtxt(diagnostic_files[-1])
    primitive_frames = glob.glob(os.path.join(
        args.data_dir, 'bin', '*.prim.*.bin'))
    force_frames = glob.glob(os.path.join(
        args.data_dir, 'bin', '*.force.*.bin'))

    summary = {
        'time': float(time[-1]),
        'turnover_times': float(time[-1]/args.eddy_time),
        'target_mach': args.target_mach,
        'target_vrms': float(target_vrms),
        'initial_mean_density': float(values('rho')[0]/volume[0]),
        'initial_mean_pressure': float(values('pgas')[0]/volume[0]),
        'initial_vrms': float(vrms[0]),
        'initial_mach': float(mach[0]),
        'initial_mean_magnetization': float(mean_magnetization[0]),
        'initial_mean_alfven_speed': float(mean_alfven_speed[0]),
        'initial_target_alfven_mach': float(target_alfven_mach[0]),
        'post_spinup_mach_mean': float(np.mean(mach[post])),
        'post_spinup_mach_std': float(np.std(mach[post])),
        'post_spinup_vrms_mean': float(np.mean(vrms[post])),
        'post_spinup_re_mean': float(np.mean(reynolds[post])),
        'post_spinup_rm_mean': float(np.mean(magnetic_reynolds[post])),
        'post_spinup_mean_magnetization': float(np.mean(
            mean_magnetization[post])),
        'post_spinup_mean_alfven_speed': float(np.mean(
            mean_alfven_speed[post])),
        'post_spinup_target_alfven_mach_mean': float(np.mean(
            target_alfven_mach[post])),
        'stationarity_window_turnovers': [
            float(turnover_time[second_half][0]),
            float(turnover_time[-1]),
        ],
        'second_half_mach_mean': float(np.mean(mach[second_half])),
        'second_half_mach_std': float(np.std(mach[second_half])),
        'second_half_mach_slope_per_turnover': linear_slope(
            mach[second_half], turnover_time[second_half]),
        'second_half_vrms_mean': float(np.mean(vrms[second_half])),
        'second_half_vrms_std': float(np.std(vrms[second_half])),
        'second_half_vrms_slope_per_turnover': linear_slope(
            vrms[second_half], turnover_time[second_half]),
        'second_half_mean_pressure': float(np.mean(mean_pressure[second_half])),
        'second_half_mean_pressure_std': float(np.std(mean_pressure[second_half])),
        'second_half_mean_pressure_slope_per_turnover': linear_slope(
            mean_pressure[second_half], turnover_time[second_half]),
        'second_half_mean_magnetization': float(np.mean(
            mean_magnetization[second_half])),
        'second_half_mean_alfven_speed': float(np.mean(
            mean_alfven_speed[second_half])),
        'second_half_target_alfven_mach_mean': float(np.mean(
            target_alfven_mach[second_half])),
        'second_half_ekin_slope_per_turnover': linear_slope(
            values('ekin')[second_half], turnover_time[second_half]),
        'second_half_eint_slope_per_turnover': linear_slope(
            values('eint')[second_half], turnover_time[second_half]),
        'second_half_injected_power_mean': float(np.mean(
            values('f_power')[second_half])),
        'second_half_injected_power_slope_per_turnover': linear_slope(
            values('f_power')[second_half], turnover_time[second_half]),
        'second_half_heating_estimator_mean': float(np.mean(
            (q_ohm + q_visc)[second_half])),
        'second_half_heating_estimator_slope_per_turnover': linear_slope(
            (q_ohm + q_visc)[second_half], turnover_time[second_half]),
        'cooling_enabled': cooling_enabled,
        'second_half_cooling_power_mean': float(np.mean(
            cooling_power[second_half])),
        'second_half_cooling_power_slope_per_turnover': linear_slope(
            cooling_power[second_half], turnover_time[second_half]),
        'second_half_drive_minus_cooling_power_mean': float(np.mean(
            values('f_power')[second_half] - cooling_power[second_half])),
        'second_half_drive_minus_cooling_power_std': float(np.std(
            values('f_power')[second_half] - cooling_power[second_half])),
        'second_half_cooling_minus_heating_mean': float(np.mean(
            cooling_power[second_half]
            - (q_ohm + q_visc)[second_half])),
        'injected_energy': float(values('e_inj')[-1]),
        'mean_injected_power': float(values('e_inj')[-1]/time[-1]),
        'cooled_energy': float(cooled_energy[-1]),
        'mean_cooling_power': float(cooled_energy[-1]/time[-1]),
        'limited_cooling_energy': float(limited_cooling_energy[-1]),
        'internal_energy_change': float(values('eint')[-1] - values('eint')[0]),
        'magnetic_energy_initial': float(values('emag')[0]),
        'magnetic_energy_final': float(values('emag')[-1]),
        'magnetic_energy_final_to_initial': float(
            values('emag')[-1]/values('emag')[0]),
        'entropy_proxy_change': float(
            values('entropy')[-1] - values('entropy')[0]),
        'second_half_entropy_proxy_slope_per_turnover': linear_slope(
            values('entropy')[second_half], turnover_time[second_half]),
        'integrated_ohmic_heating_estimator': ohmic_integral,
        'integrated_viscous_heating_estimator': viscous_integral,
        'ohmic_heating_fraction': ohmic_integral/heating_integral,
        'viscous_heating_fraction': viscous_integral/heating_integral,
        'energy_audit_max_abs': float(np.max(np.abs(energy_residual))),
        'momentum_audit_max_abs': float(np.max(np.abs(momentum))),
        'mass_relative_error_max': float(np.max(np.abs(
            values('mass')/values('mass')[0] - 1.0))),
        'divb2_max': float(np.max(values('divb2'))),
        'minimum_density': float(final_diagnostic[0]),
        'minimum_internal_energy': float(final_diagnostic[1]),
        'maximum_lorentz_factor': float(final_diagnostic[2]),
        'fofc_count': int(final_diagnostic[3]),
        'density_floor_count': int(final_diagnostic[4]),
        'energy_floor_count': int(final_diagnostic[5]),
        'velocity_ceiling_count': int(final_diagnostic[6]),
        'recovery_failure_count': int(final_diagnostic[7]),
        'primitive_movie_frames': len(primitive_frames),
        'force_movie_frames': len(force_frames),
        'turnover_statistics': turnover,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--history', default='driven_mach0p5_re200.user.hst')
    parser.add_argument('--eddy-time', type=float, required=True)
    parser.add_argument('--drive-scale', type=float, required=True)
    parser.add_argument('--viscosity', type=float, required=True)
    parser.add_argument('--resistivity', type=float, required=True)
    parser.add_argument('--target-mach', type=float, required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    summary = summarize(args)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as stream:
            stream.write(rendered + '\n')


if __name__ == '__main__':
    main()
