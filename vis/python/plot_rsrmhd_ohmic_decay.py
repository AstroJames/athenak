#!/usr/bin/env python3
"""Plot strong-guide-field Harris-sheet Ohmic decay diagnostics."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.rc_file("/Users/beattijr/.matplotlib/matplotlibrc")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import athena_read  # noqa: E402


def load_snapshots(directory, basename):
    snapshots = []
    pattern = f"{basename}.primitive.*.tab"
    for path in sorted((directory / "tab").glob(pattern)):
        data = athena_read.tab(str(path))
        order = np.argsort(data["x1v"])
        x = data["x1v"][order]
        by = data["bcc2"][order]
        current = np.gradient(by, x, edge_order=2)
        snapshots.append((float(data["time"]), x, current))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots matching tab/{pattern}")
    return snapshots


def diagnostics(snapshots):
    times = np.array([snapshot[0] for snapshot in snapshots])
    peaks = np.array([np.max(snapshot[2]) for snapshot in snapshots])
    moments = []
    for _, x, current in snapshots:
        weight = current**2
        moments.append(
            np.trapezoid(x**2 * weight, x) / np.trapezoid(weight, x))
    return times, peaks, np.array(moments)


def model(time, x, resistivity, sheet_width, field):
    shifted_time = time + sheet_width**2 / (np.pi * resistivity)
    peak = field / np.sqrt(np.pi * resistivity * shifted_time)
    current = peak * np.exp(-x**2 / (4.0 * resistivity * shifted_time))
    moment = resistivity * shifted_time
    return current, peak, moment


def log_slope(x, y, minimum_time):
    mask = x >= minimum_time
    return np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0]


def plot(snapshots, scan_cases, output_prefix, resistivity, sheet_width, field):
    times, _, _ = diagnostics(snapshots)
    positive = times > 0.0
    profile_indices = np.unique(np.linspace(
        0, len(snapshots) - 1, min(6, len(snapshots))).astype(int))
    profile_snapshots = [snapshots[index] for index in profile_indices]
    color_map = plt.get_cmap("viridis")
    color_norm = Normalize(times.min(), times.max())
    colors = [color_map(color_norm(snapshot[0]))
              for snapshot in profile_snapshots]
    figure = plt.figure(figsize=(12.0, 3.9), layout="none")
    grid = figure.add_gridspec(
        2, 3, height_ratios=(0.075, 1.0), hspace=0.08, wspace=0.32,
        left=0.07, right=0.985, bottom=0.18, top=0.91)
    axes = [figure.add_subplot(grid[1, column]) for column in range(3)]

    for (time, x, current), color in zip(profile_snapshots, colors):
        axes[0].plot(x, current, color=color)
        analytic, _, _ = model(time, x, resistivity, sheet_width, field)
        axes[0].plot(x, analytic, color=color, linestyle=":")
    axes[0].set_xlim(-0.45, 0.45)
    axes[0].set_xlabel(r"$x/L$")
    axes[0].set_ylabel(r"$J^z=\partial_x B^y$")
    color_axis = figure.add_subplot(grid[0, 0])
    colorbar = figure.colorbar(
        ScalarMappable(norm=color_norm, cmap=color_map), cax=color_axis,
        orientation="horizontal")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.set_label(r"$t/t_c$", labelpad=2)

    scan_colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.88, len(scan_cases)))
    slopes = []
    eta_labels = []
    for (eta, case_snapshots), color in zip(scan_cases, scan_colors):
        case_times, peaks, moments = diagnostics(case_snapshots)
        case_positive = case_times > 0.0
        fine_time = np.logspace(
            np.log10(case_times[case_positive].min()),
            np.log10(case_times.max()), 300)
        shifted_fine = fine_time + sheet_width**2 / (np.pi * eta)
        peak_model = field / np.sqrt(np.pi * eta * shifted_fine)
        rms_model = np.sqrt(eta * shifted_fine)
        axes[1].loglog(
            case_times[case_positive], peaks[case_positive], color=color,
            marker="o", markerfacecolor="none")
        axes[1].loglog(
            fine_time, peak_model, color=color, linestyle=":")
        axes[2].loglog(
            case_times[case_positive], np.sqrt(moments[case_positive]),
            color=color, marker="o", markerfacecolor="none")
        axes[2].loglog(
            fine_time, rms_model, color=color, linestyle=":")
        eta_labels.append((eta, color))
        fit_start = max(0.4, case_times[case_positive].min())
        slopes.append((
            eta,
            log_slope(case_times[case_positive], peaks[case_positive], fit_start),
            log_slope(
                case_times[case_positive], np.sqrt(moments[case_positive]),
                fit_start),
        ))
    axes[1].set_xlabel(r"$t/t_c$")
    axes[1].set_ylabel(r"$J^z_{\max}$")
    axes[2].set_xlabel(r"$t/t_c$")
    axes[2].set_ylabel(
        r"$x_{\rm rms}=\langle x^2\rangle_{(J^z)^2}^{1/2}$")
    maximum_time = max(diagnostics(case[1])[0].max() for case in scan_cases)
    for axis in axes[1:]:
        axis.set_xlim(right=2.3 * maximum_time)
    figure.canvas.draw()
    log_time_limits = np.log10(axes[1].get_xlim())
    label_time = 10.0**(
        log_time_limits[0] + 0.94 * np.diff(log_time_limits)[0])
    for eta, color in eta_labels:
        shifted_label_time = label_time + sheet_width**2 / (np.pi * eta)
        label_peak = field / np.sqrt(
            np.pi * eta * shifted_label_time)
        tangent_time = 0.9 * label_time
        shifted_tangent_time = (
            tangent_time + sheet_width**2 / (np.pi * eta))
        tangent_peak = field / np.sqrt(
            np.pi * eta * shifted_tangent_time)
        tangent = axes[1].transData.transform(np.array([
            [tangent_time, tangent_peak],
            [label_time, label_peak],
        ]))
        angle = np.degrees(np.arctan2(
            tangent[1, 1] - tangent[0, 1],
            tangent[1, 0] - tangent[0, 0]))
        axes[1].text(
            label_time, 1.12 * label_peak, rf"$\eta={eta:g}$", color=color,
            rotation=angle, rotation_mode="anchor", ha="right", va="bottom",
            clip_on=True)
    axes[0].text(0.95, 0.08, rf"$\eta={resistivity:g}$",
                 transform=axes[0].transAxes, ha="right", va="bottom")
    axes[2].text(0.95, 0.08, r"$N_x=2048$",
                 transform=axes[2].transAxes, ha="right", va="bottom")
    annotation_box = {
        "facecolor": matplotlib.rcParams["axes.facecolor"],
        "alpha": 0.75,
        "edgecolor": "none",
    }
    panel_positions = ((0.04, 0.94), (0.04, 0.06), (0.04, 0.94))
    for index, (axis, position) in enumerate(zip(axes, panel_positions)):
        axis.grid(False)
        axis.text(*position, f"({chr(ord('a') + index)})",
                  transform=axis.transAxes,
                  ha="left", va="bottom" if index == 1 else "top",
                  bbox=annotation_box)
    style = [
        Line2D([0], [0], color="0.25", label="AthenaK"),
        Line2D([0], [0], color="0.25", linestyle=":",
               label="diffusive model"),
    ]
    axes[1].legend(handles=style, frameon=False, loc="upper right",
                   fontsize="x-small", handlelength=1.0,
                   handletextpad=0.35, labelspacing=0.2,
                   borderaxespad=0.25)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=220,
                   bbox_inches="tight")
    plt.close(figure)
    return slopes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--basename", default="ohmic_decay")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--resistivity", type=float, default=0.01)
    parser.add_argument("--sheet-width", type=float, default=0.02)
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument(
        "--scan", action="append", default=[], metavar="ETA=PATH",
        help="add a uniform-resistivity run to the max/RMS panels")
    args = parser.parse_args()
    snapshots = load_snapshots(args.data_dir, args.basename)
    scan_cases = [(args.resistivity, snapshots)]
    for value in args.scan:
        eta_text, path_text = value.split("=", 1)
        scan_cases.append((
            float(eta_text), load_snapshots(Path(path_text), args.basename)))
    scan_cases.sort(key=lambda case: case[0])
    slopes = plot(snapshots, scan_cases, args.output_prefix, args.resistivity,
                  args.sheet_width, args.field)
    for eta, peak_slope, rms_slope in slopes:
        print(f"eta={eta:g} late-time slopes: peak={peak_slope:.6f}, "
              f"rms_width={rms_slope:.6f}")


if __name__ == "__main__":
    main()
