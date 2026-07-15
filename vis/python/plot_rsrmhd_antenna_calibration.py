#!/usr/bin/env python3
"""Plot time histories for one or more antenna calibration runs."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import athena_read


def load_series(path):
    history = athena_read.hst(str(path))
    time = np.asarray(history['time'])
    volume = np.asarray(history['volume'])
    box_length = float(volume[0]**(1.0/3.0))
    b0_squared = 2.0*history['emag'][0]/volume[0]
    reference = np.flatnonzero(np.asarray(history['vA_ant']) > 0.0)[0]
    va0 = float(history['vA_ant'][reference])
    alfven_time = box_length/va0
    nominal_power = 0.5*b0_squared*va0*box_length**2

    delta_b_squared = np.maximum(
        2.0*np.asarray(history['emag'])/volume - b0_squared, 0.0
    )
    velocity_rms = np.sqrt(np.asarray(history['v2'])/volume)
    alfven_speed = np.asarray(history['v_alfven'])/volume
    valid_time = np.where(time > 0.0, time, np.nan)
    return {
        'time': time/alfven_time,
        'delta_b': np.sqrt(delta_b_squared/b0_squared),
        'velocity': velocity_rms/alfven_speed,
        'injection': np.asarray(history['e_ant'])/(nominal_power*valid_time),
        'heating': (
            np.asarray(history['eint']) - history['eint'][0]
        )/(nominal_power*valid_time),
        'sigma': np.asarray(history['sigma'])/volume,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('histories', nargs='+', type=Path)
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.histories):
        parser.error('--labels must have one entry per history')

    mpl.rc_file('/Users/beattijr/.matplotlib/matplotlibrc')
    labels = args.labels or [path.stem for path in args.histories]
    series = [load_series(path) for path in args.histories]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharex=True,
                             layout='none')
    panels = (
        ('delta_b', r'$\delta B_{\rm rms}/B_0$', 1.0),
        ('velocity', r'$v_{\rm rms}/v_{\rm A}$', 0.7),
        ('efficiency', r'normalized energy', 1.7),
        ('sigma', r'$\sigma$', None),
    )
    handles = []
    for index, (label, values) in enumerate(zip(labels, series)):
        color = colors[index % len(colors)]
        for axis, (key, _, _) in zip(axes.flat, panels):
            if key == 'efficiency':
                line, = axis.plot(values['time'], values['injection'],
                                  color=color, label=label)
                axis.plot(values['time'], values['heating'], color=color,
                          linestyle='--')
            else:
                line, = axis.plot(values['time'], values[key], color=color,
                                  label=label)
        handles.append(line)

    for axis, (_, ylabel, target) in zip(axes.flat, panels):
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 6.0)
        axis.axvspan(4.0, 6.0, color='0.92', zorder=-10)
        if target is not None:
            axis.axhline(target, color='0.35', linestyle=':', linewidth=1.0)
    axes[1, 0].set_xlabel(r'$t/t_{\rm A0}$')
    axes[1, 1].set_xlabel(r'$t/t_{\rm A0}$')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=len(labels), frameon=False)
    style_handles = [
        Line2D([], [], color='0.25', linestyle='-', label='injected'),
        Line2D([], [], color='0.25', linestyle='--', label='internal'),
    ]
    fig.legend(handles=style_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
