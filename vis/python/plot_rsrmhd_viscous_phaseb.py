"""Plot the causal diffusion-limit and longitudinal viscous-sound regressions."""

import argparse
import csv
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.rc_file("/Users/beattijr/.matplotlib/matplotlibrc")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


def transverse_solution(time, nu, tau, wave_number, amplitude):
    alpha = 0.5 / tau
    discriminant = 1.0 - 4.0 * tau * nu * wave_number**2
    decay = np.exp(-alpha * time)
    if discriminant > 0.0:
        rate = np.sqrt(discriminant) / (2.0 * tau)
        return amplitude * decay * (
            np.cosh(rate * time) + alpha * np.sinh(rate * time) / rate)
    omega = np.sqrt(-discriminant) / (2.0 * tau)
    return amplitude * decay * (
        np.cos(omega * time) + alpha * np.sin(omega * time) / omega)


def longitudinal_solution(time, nu, tau, wave_number, amplitude):
    gamma = 5.0 / 3.0
    pressure = gamma - 1.0
    enthalpy = 1.0 + gamma
    matrix = np.array([
        [0.0, wave_number / enthalpy, wave_number / enthalpy],
        [-gamma * pressure * wave_number, 0.0, 0.0],
        [-(4.0 / 3.0) * enthalpy * nu * wave_number / tau, 0.0, -1.0 / tau],
    ])
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    coefficients = np.linalg.solve(eigenvectors, np.array([amplitude, 0.0, 0.0]))
    values = []
    for current_time in np.atleast_1d(time):
        state = eigenvectors @ (np.exp(eigenvalues * current_time) * coefficients)
        values.append(np.real_if_close(state).real)
    return np.asarray(values)


def run_athena(athena, input_file, overrides, result_name):
    with tempfile.TemporaryDirectory(
            prefix="athenak-phaseb-", dir="/private/tmp") as run_dir:
        command = [str(athena), "-i", str(input_file), *overrides]
        subprocess.run(command, cwd=run_dir, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return np.loadtxt(Path(run_dir) / result_name)


def collect_data(athena, telegraph_input, longitudinal_input):
    diffusion_times = np.linspace(0.0, 2.0, 11)
    diffusion_rows = [[0.0, 1.0e-4, 1.0 / 128.0]]
    for time in diffusion_times[1:]:
        result = run_athena(athena, telegraph_input, [
            f"time/tlim={time:.17g}",
            "mhd/shear_viscosity=0.0025",
            "mhd/shear_relaxation_time=0.01",
            "mesh/nx1=128", "meshblock/nx1=64",
        ], "rsrmhd_viscous_telegraph-errs.dat")
        diffusion_rows.append([time, result[2], result[6]])

    timestep_rows = []
    for resolution in (64, 128, 256, 512):
        result = run_athena(athena, telegraph_input, [
            "time/tlim=0.05",
            "mhd/shear_viscosity=0.0025",
            "mhd/shear_relaxation_time=0.01",
            f"mesh/nx1={resolution}", f"meshblock/nx1={resolution // 2}",
        ], "rsrmhd_viscous_telegraph-errs.dat")
        timestep_rows.append([1.0 / resolution, result[6]])

    longitudinal_times = np.linspace(0.0, 1.0, 11)
    longitudinal_rows = [[0.0, 1.0e-4, 0.0, 0.0]]
    for time in longitudinal_times[1:]:
        result = run_athena(athena, longitudinal_input, [
            f"time/tlim={time:.17g}",
            "mesh/nx1=128", "meshblock/nx1=64",
        ], "rsrmhd_viscous_longitudinal-amps.dat")
        longitudinal_rows.append([time, result[0], result[1], result[2]])
    return (np.asarray(diffusion_rows), np.asarray(timestep_rows),
            np.asarray(longitudinal_rows))


def write_csv(path, diffusion, timestep, longitudinal):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test", "x", "value1", "value2", "value3"])
        for row in diffusion:
            writer.writerow(["diffusion", *row, ""])
        for row in timestep:
            writer.writerow(["timestep", *row, "", ""])
        for row in longitudinal:
            writer.writerow(["longitudinal", *row])


def make_plot(diffusion, timestep, longitudinal, output_dir):
    amplitude = 1.0e-4
    enthalpy = 1.0 + 5.0 / 3.0
    wave_number = 2.0 * np.pi
    dense_diffusion_time = np.linspace(0.0, 2.0, 500)
    dense_longitudinal_time = np.linspace(0.0, 1.0, 500)
    causal = transverse_solution(
        dense_diffusion_time, 0.0025, 0.01, wave_number, amplitude)
    navier_stokes = amplitude * np.exp(
        -0.0025 * wave_number**2 * dense_diffusion_time)
    longitudinal_exact = longitudinal_solution(
        dense_longitudinal_time, 0.03, 0.2, wave_number, amplitude)

    figure = plt.figure(figsize=(9.4, 7.2), layout="none")
    grid = figure.add_gridspec(
        2, 2, left=0.10, right=0.98, bottom=0.09, top=0.97,
        hspace=0.32, wspace=0.30,
    )
    axes = np.asarray([
        [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
        [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
    ])
    axes[0, 0].plot(dense_diffusion_time, causal / amplitude,
                    label="Israel--Stewart")
    axes[0, 0].plot(dense_diffusion_time, navier_stokes / amplitude,
                    linestyle="--", label="Navier--Stokes limit")
    axes[0, 0].plot(diffusion[:, 0], diffusion[:, 1] / amplitude,
                    linestyle="none", marker="o", markerfacecolor="none",
                    label="AthenaK")
    axes[0, 0].set_xlabel(r"$t/t_c$")
    axes[0, 0].set_ylabel(r"$\widehat{u^y}/A$")
    axes[0, 0].set_xlim(-0.03, 2.10)
    axes[0, 0].set_ylim(0.81, 1.008)
    axes[0, 0].legend(
        frameon=False, loc="lower left", fontsize=10.5,
        handlelength=2.2, labelspacing=0.25,
    )
    axes[0, 0].text(
        0.97, 0.95, r"$\nu_{\rm sh}=0.0025$, $\tau_\pi=0.01$",
        transform=axes[0, 0].transAxes, ha="right", va="top",
        fontsize=10.5,
    )

    dx = timestep[:, 0]
    dtnew = timestep[:, 1]
    axes[0, 1].loglog(dx, dtnew, marker="o", markerfacecolor="none",
                      label="AthenaK")
    axes[0, 1].loglog(dx, dx, linestyle="--", label=r"$\propto\Delta x$")
    axes[0, 1].loglog(dx, dx**2 / np.max(dx), linestyle=":",
                      label=r"$\propto\Delta x^2$")
    axes[0, 1].set_xlabel(r"$\Delta x/L$")
    axes[0, 1].set_ylabel(r"$\Delta t_{\rm new}/t_c$")
    axes[0, 1].set_xticks(dx)
    axes[0, 1].set_xticklabels([
        r"$1/64$", r"$1/128$", r"$1/256$", r"$1/512$",
    ])
    axes[0, 1].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 1].legend(
        frameon=False, loc="lower right", fontsize=10.5,
        handlelength=2.2, labelspacing=0.25,
    )
    axes[0, 1].text(
        0.03, 0.83, "causal crossing-time bound",
        transform=axes[0, 1].transAxes, ha="left", va="top",
        fontsize=10.5,
    )

    axes[1, 0].plot(
        dense_longitudinal_time, longitudinal_exact[:, 0] / amplitude,
        label="exact")
    axes[1, 0].plot(
        longitudinal[:, 0], longitudinal[:, 1] / amplitude,
        linestyle="none", marker="o", markerfacecolor="none", label="AthenaK")
    axes[1, 0].set_xlabel(r"$t/t_c$")
    axes[1, 0].set_ylabel(r"$\widehat{u^x}/A$")
    axes[1, 0].set_xlim(-0.03, 1.05)
    axes[1, 0].legend(
        frameon=False, loc="upper right", fontsize=10.5,
        handlelength=2.2, labelspacing=0.25,
    )
    axes[1, 0].text(
        0.03, 0.08, r"$\nu_{\rm sh}=0.03$, $\tau_\pi=0.2$",
        transform=axes[1, 0].transAxes, ha="left", va="bottom",
        fontsize=10.5,
    )

    axes[1, 1].plot(
        dense_longitudinal_time, longitudinal_exact[:, 1] / (enthalpy * amplitude),
        label=r"$\widehat{\delta p}/(wA)$")
    axes[1, 1].plot(
        dense_longitudinal_time, longitudinal_exact[:, 2] / (enthalpy * amplitude),
        label=r"$\widehat{\pi^{xx}}/(wA)$")
    axes[1, 1].plot(
        longitudinal[:, 0], longitudinal[:, 2] / (enthalpy * amplitude),
        linestyle="none", marker="o", markerfacecolor="none")
    axes[1, 1].plot(
        longitudinal[:, 0], longitudinal[:, 3] / (enthalpy * amplitude),
        linestyle="none", marker="s", markerfacecolor="none")
    axes[1, 1].set_xlabel(r"$t/t_c$")
    axes[1, 1].set_ylabel("normalized amplitude")
    axes[1, 1].set_xlim(-0.03, 1.05)
    axes[1, 1].legend(
        frameon=False, loc="upper center", fontsize=10.5,
        handlelength=2.2, labelspacing=0.25,
    )

    panel_positions = ((0.03, 0.55), (0.03, 0.95),
                       (0.03, 0.55), (0.03, 0.95))
    for label, axis, position in zip(
            ("(a)", "(b)", "(c)", "(d)"), axes.flat, panel_positions):
        axis.text(
            *position, label, transform=axis.transAxes,
            ha="left", va="top", fontsize=13,
        )
        axis.grid(False)
    figure.savefig(output_dir / "viscous_phaseb_validation.pdf")
    figure.savefig(
        output_dir / "viscous_phaseb_validation.png", dpi=250,
        transparent=False, facecolor=matplotlib.rcParams["figure.facecolor"],
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--telegraph-input", type=Path, required=True)
    parser.add_argument("--longitudinal-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diffusion, timestep, longitudinal = collect_data(
        args.athena.resolve(), args.telegraph_input.resolve(),
        args.longitudinal_input.resolve())
    write_csv(args.output_dir / "viscous_phaseb_validation.csv",
              diffusion, timestep, longitudinal)
    make_plot(diffusion, timestep, longitudinal, args.output_dir)


if __name__ == "__main__":
    main()
