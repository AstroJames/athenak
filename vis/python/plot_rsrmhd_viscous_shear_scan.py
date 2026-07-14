#!/usr/bin/env python3
"""Plot vorticity and causal shear stress for a 1D viscosity scan."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.rc_file('/Users/beattijr/.matplotlib/matplotlibrc')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


VISCOSITIES = (0.005, 0.01, 0.02, 0.04, 0.05)
TAGS = ('005', '010', '020', '040', '050')
AMPLITUDE = 0.5
TAU_PI = 0.2
TIME = 0.4
WAVE_NUMBER = 2.0 * np.pi
ENTHALPY_DENSITY = 8.0 / 3.0


def telegraph_amplitudes(viscosity):
    discriminant = 1.0 - 4.0 * TAU_PI * viscosity * WAVE_NUMBER**2
    decay = np.exp(-TIME / (2.0 * TAU_PI))
    if discriminant > 1.0e-14:
        rate = np.sqrt(discriminant) / (2.0 * TAU_PI)
        velocity = AMPLITUDE * decay * (
            np.cosh(rate * TIME)
            + np.sinh(rate * TIME) / (2.0 * TAU_PI * rate))
        derivative = -AMPLITUDE * decay * viscosity * WAVE_NUMBER**2 * (
            np.sinh(rate * TIME) / (TAU_PI * rate))
    elif discriminant < -1.0e-14:
        frequency = np.sqrt(-discriminant) / (2.0 * TAU_PI)
        velocity = AMPLITUDE * decay * (
            np.cos(frequency * TIME)
            + np.sin(frequency * TIME) / (2.0 * TAU_PI * frequency))
        derivative = -AMPLITUDE * decay * viscosity * WAVE_NUMBER**2 * (
            np.sin(frequency * TIME) / (TAU_PI * frequency))
    else:
        alpha = 1.0 / (2.0 * TAU_PI)
        velocity = AMPLITUDE * decay * (1.0 + alpha * TIME)
        derivative = -AMPLITUDE * decay * alpha**2 * TIME
    stress = ENTHALPY_DENSITY * derivative / WAVE_NUMBER
    return velocity, stress


def load_profiles(data_dir):
    profiles = []
    for viscosity, tag in zip(VISCOSITIES, TAGS):
        data = np.loadtxt(data_dir / f'shear_nu{tag}-profile.dat')
        data = data[np.argsort(data[:, 0])]
        x, u2, pi12 = data.T
        lorentz = np.sqrt(1.0 + u2**2)
        v2 = u2 / lorentz
        dx = x[1] - x[0]
        vorticity = (np.roll(v2, -1) - np.roll(v2, 1)) / (2.0 * dx)
        profiles.append((viscosity, x, vorticity, pi12))
    return profiles


def plot(profiles, output_prefix):
    colors = plt.get_cmap('viridis')(np.linspace(0.12, 0.90, len(profiles)))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharex=True)

    for (viscosity, x, vorticity, pi12), color in zip(profiles, colors):
        label = rf'$\nu_{{\rm sh}}={viscosity:g}$'
        axes[0].plot(x, -vorticity, color=color, label=label)
        axes[1].plot(x, pi12, color=color)
        velocity_amplitude, stress_amplitude = telegraph_amplitudes(viscosity)
        phase = WAVE_NUMBER * x
        analytic_negative_vorticity = (
            -WAVE_NUMBER * velocity_amplitude * np.cos(phase)
            / (1.0 + velocity_amplitude**2 * np.sin(phase)**2)**1.5)
        analytic_stress = stress_amplitude * np.cos(phase)
        axes[0].plot(x, analytic_negative_vorticity, color=color, linestyle=':')
        axes[1].plot(x, analytic_stress, color=color, linestyle=':')

    axes[0].set_ylabel(r'$-\omega^z=-\partial_x v^y$')
    axes[1].set_ylabel(r'$\pi^{xy}$')
    for axis in axes:
        axis.set_xlabel(r'$x/L$')
        axis.set_xlim(0.0, 1.0)
        axis.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=len(profiles),
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    style_handles = [
        Line2D([0], [0], color='0.25', label='AthenaK'),
        Line2D([0], [0], color='0.25', linestyle=':', label='analytical'),
    ]
    fig.legend(handles=style_handles, loc='upper center', ncols=2,
               bbox_to_anchor=(0.5, 0.91), frameon=False)
    axes[0].text(0.96, 0.94, '(a)', transform=axes[0].transAxes,
                 ha='right', va='top')
    axes[1].text(0.04, 0.94, '(b)', transform=axes[1].transAxes,
                 ha='left', va='top')
    axes[1].text(0.96, 0.06,
                 r'$t/t_c=0.4,\quad \tau_\pi/t_c=0.2$',
                 transform=axes[1].transAxes, ha='right', va='bottom')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.77))
    fig.savefig(output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output_prefix.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    plot(load_profiles(args.data_dir), args.output_prefix)


if __name__ == '__main__':
    main()
