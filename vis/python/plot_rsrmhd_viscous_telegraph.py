"""Run and plot the linear Israel--Stewart transverse telegraph test."""

import argparse
import csv
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.rc_file("/Users/beattijr/.matplotlib/matplotlibrc")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def telegraph_solution(time, nu, tau, wave_number, amplitude):
    """Return velocity and shear-stress Fourier amplitudes."""
    time = np.asarray(time)
    alpha = 0.5 / tau
    discriminant = 1.0 - 4.0 * tau * nu * wave_number**2
    decay = np.exp(-alpha * time)
    if discriminant > 1.0e-14:
        rate = np.sqrt(discriminant) / (2.0 * tau)
        velocity = amplitude * decay * (
            np.cosh(rate * time) + alpha * np.sinh(rate * time) / rate)
        derivative = (-amplitude * decay * nu * wave_number**2
                      * np.sinh(rate * time) / (tau * rate))
    elif discriminant < -1.0e-14:
        omega = np.sqrt(-discriminant) / (2.0 * tau)
        velocity = amplitude * decay * (
            np.cos(omega * time) + alpha * np.sin(omega * time) / omega)
        derivative = (-amplitude * decay * nu * wave_number**2
                      * np.sin(omega * time) / (tau * omega))
    else:
        velocity = amplitude * decay * (1.0 + alpha * time)
        derivative = -amplitude * decay * alpha**2 * time
    return velocity, derivative / wave_number


def run_sweep(athena, input_file, scans, times, output_dir):
    """Run AthenaK at the requested sample times and return Fourier amplitudes."""
    amplitude = 1.0e-4
    enthalpy_density = 1.0 + 5.0 / 3.0
    rows = []
    for scan_id, cases in enumerate(scans):
        for nu, tau in cases:
            rows.append((scan_id, nu, tau, 0.0, amplitude, amplitude, 0.0, 0.0))
            for time in times[1:]:
                with tempfile.TemporaryDirectory(
                        prefix="athenak-telegraph-", dir="/private/tmp") as run_dir:
                    command = [
                        str(athena), "-i", str(input_file),
                        f"time/tlim={time:.17g}",
                        f"mhd/shear_viscosity={nu:.17g}",
                        f"mhd/shear_relaxation_time={tau:.17g}",
                        "mesh/nx1=128", "meshblock/nx1=64",
                    ]
                    subprocess.run(command, cwd=run_dir, check=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.STDOUT)
                    values = np.loadtxt(
                        Path(run_dir) / "rsrmhd_viscous_telegraph-errs.dat")
                rows.append((scan_id, nu, tau, time, values[2], values[3],
                             values[4], values[5]))

    csv_path = output_dir / "viscous_telegraph_sweep.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "scan", "nu_sh", "tau_pi", "time", "velocity_athenak",
            "velocity_exact", "stress_athenak", "stress_exact",
        ])
        writer.writerows(rows)
    return np.asarray(rows), amplitude, enthalpy_density


def make_plot(rows, scans, amplitude, enthalpy_density, output_dir):
    """Plot analytic curves and AthenaK Fourier amplitudes."""
    wave_number = 2.0 * np.pi
    dense_time = np.linspace(0.0, 1.0, 500)
    figure, axes = plt.subplots(
        2, 2, figsize=(9.4, 7.3), sharex=True, sharey="col", layout="none",
    )
    figure.subplots_adjust(
        left=0.10, right=0.98, bottom=0.09, top=0.82,
        hspace=0.42, wspace=0.24,
    )
    panel_labels = (("(a)", "(b)"), ("(c)", "(d)"))
    for scan_id, cases in enumerate(scans):
        handles = []
        for nu, tau in cases:
            velocity, stress_over_w = telegraph_solution(
                dense_time, nu, tau, wave_number, amplitude)
            if scan_id == 0:
                label = (rf"${nu:.2f}$ "
                         rf"$(c_{{\rm T}}={np.sqrt(nu/tau):.3f})$")
            else:
                label = (rf"${tau:.1f}$ "
                         rf"$(c_{{\rm T}}={np.sqrt(nu/tau):.3f})$")
            line, = axes[scan_id, 0].plot(
                dense_time, velocity / amplitude)
            handles.append(Line2D(
                [0], [0], color=line.get_color(), marker="o",
                markerfacecolor="none", label=label,
            ))
            axes[scan_id, 1].plot(
                dense_time, stress_over_w / amplitude, color=line.get_color())
            subset = rows[
                np.isclose(rows[:, 0], scan_id)
                & np.isclose(rows[:, 1], nu)
                & np.isclose(rows[:, 2], tau)]
            axes[scan_id, 0].plot(
                subset[:, 3], subset[:, 4] / amplitude, linestyle="none",
                marker="o", markerfacecolor="none", color=line.get_color(),
                markersize=4.5)
            axes[scan_id, 1].plot(
                subset[:, 3], subset[:, 6] / (enthalpy_density * amplitude),
                linestyle="none", marker="o", markerfacecolor="none",
                color=line.get_color(), markersize=4.5)

        legend_title = (
            r"fixed $\tau_\pi=0.2$; varying $\nu_{\rm sh}$"
            if scan_id == 0 else
            r"fixed $\nu_{\rm sh}=0.02$; varying $\tau_\pi$"
        )
        axes[scan_id, 0].legend(
            handles=handles, title=legend_title, ncols=3,
            loc="lower center", bbox_to_anchor=(1.08, 1.04),
            frameon=False, fontsize=10.0, title_fontsize=10.5,
            handlelength=2.0, handletextpad=0.35, columnspacing=0.9,
            labelspacing=0.25, alignment="center",
        )
        for column in range(2):
            axis = axes[scan_id, column]
            axis.axhline(0.0, color="0.65", linewidth=0.7)
            axis.text(
                0.03, 0.08, panel_labels[scan_id][column],
                transform=axis.transAxes, ha="left", va="bottom",
                fontsize=13,
            )
            axis.grid(False)
            axis.set_xlim(0.0, 1.0)

    for row in range(2):
        axes[row, 0].set_ylabel(r"$\widehat{u^y}/A$")
        axes[row, 1].set_ylabel(r"$\widehat{\pi^{xy}}/(wA)$")
    axes[1, 0].set_xlabel(r"$t/t_c$")
    axes[1, 1].set_xlabel(r"$t/t_c$")
    figure.savefig(output_dir / "viscous_telegraph_sweep.pdf")
    figure.savefig(
        output_dir / "viscous_telegraph_sweep.png", dpi=250,
        transparent=False, facecolor=matplotlib.rcParams["figure.facecolor"],
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.athena = args.athena.resolve()
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scans = (
        ((0.01, 0.2), (0.03, 0.2), (0.05, 0.2)),
        ((0.02, 0.1), (0.02, 0.2), (0.02, 0.4)),
    )
    times = np.linspace(0.0, 1.0, 11)
    rows, amplitude, enthalpy_density = run_sweep(
        args.athena, args.input, scans, times, args.output_dir)
    make_plot(rows, scans, amplitude, enthalpy_density, args.output_dir)
    for scan_id, cases in enumerate(scans):
        for nu, tau in cases:
            subset = rows[
                np.isclose(rows[:, 0], scan_id)
                & np.isclose(rows[:, 1], nu)
                & np.isclose(rows[:, 2], tau)]
            velocity_error = np.max(np.abs(subset[:, 4] - subset[:, 5])) / amplitude
            stress_error = np.max(np.abs(subset[:, 6] - subset[:, 7])) / (
                enthalpy_density * amplitude)
            print(f"scan={scan_id} nu={nu:.2f} tau={tau:.1f} "
                  f"max_velocity_error={velocity_error:.6e} "
                  f"max_stress_error={stress_error:.6e}")


if __name__ == "__main__":
    main()
