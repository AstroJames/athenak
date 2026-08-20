#!/usr/bin/env python3
"""Plot the six-scheme AthenaK Xu--Shu Rayleigh--Taylor comparison.

The run root must contain one directory per reconstruction key below.  Each
directory is expected to contain the final ``rt_xu_shu.out2.*.bin`` output.
"""

import argparse
import glob
import os
import sys

import matplotlib
import numpy as np


SCHEMES = (
    ("plm", r"\mathrm{PLM}"),
    ("wenoz", r"\mathrm{WENO\text{-}Z}"),
    ("teno5", r"\mathrm{TENO5}"),
    ("teno5_opt", r"\mathrm{TENO5\text{-}opt}"),
    ("teno6", r"\mathrm{TENO6}"),
    ("teno6_opt", r"\mathrm{TENO6\text{-}opt}"),
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="directory containing the six run folders")
    parser.add_argument("output", help="output PNG or PDF path")
    parser.add_argument(
        "--matplotlibrc",
        default="/Users/beattijr/.matplotlib/matplotlibrc",
        help="Matplotlib configuration file",
    )
    parser.add_argument("--contour-min", type=float, default=0.85)
    parser.add_argument("--contour-max", type=float, default=2.25)
    parser.add_argument("--num-contours", type=int, default=30)
    parser.add_argument("--expected-time", type=float, default=1.95)
    parser.add_argument("--expected-nx1", type=int, default=128)
    parser.add_argument("--expected-nx2", type=int, default=512)
    parser.add_argument(
        "--panel-dir",
        help="optional directory for individual panel PNGs used by the visualization",
    )
    return parser.parse_args()


def _latest_output(run_root, key):
    pattern = os.path.join(run_root, key, "bin", "rt_xu_shu.hydro_w.*.bin")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no output matching {pattern}")
    return candidates[-1]


def _assemble_density(filedata):
    nx1 = filedata["Nx1"]
    nx2 = filedata["Nx2"]
    density = np.full((nx2, nx1), np.nan)
    dx1 = (filedata["x1max"] - filedata["x1min"]) / nx1
    dx2 = (filedata["x2max"] - filedata["x2min"]) / nx2
    for block, geometry in zip(filedata["mb_data"]["dens"],
                               filedata["mb_geometry"]):
        i0 = int(round((geometry[0] - filedata["x1min"])/dx1))
        j0 = int(round((geometry[2] - filedata["x2min"])/dx2))
        block2d = block[0]
        nj, ni = block2d.shape
        density[j0:j0+nj, i0:i0+ni] = block2d

    if not np.isfinite(density).all():
        raise RuntimeError("assembled density contains missing or non-finite cells")

    x1 = filedata["x1min"] + (np.arange(nx1) + 0.5)*dx1
    x2 = filedata["x2min"] + (np.arange(nx2) + 0.5)*dx2
    return x1, x2, density


def _draw_panel(axis, x1, x2, density, label, panel_letter, levels):
    axis.contour(x1, x2, density, levels=levels, colors="black", linewidths=0.45)
    axis.set_xlim(0.0, 0.25)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        0.5,
        0.955,
        rf"$\mathbf{{({panel_letter})}}\ {label}$",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        clip_on=False,
        zorder=10,
    )


def main():
    args = _arguments()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import bin_convert

    matplotlib.use("Agg")
    matplotlib.rc_file(os.path.abspath(os.path.expanduser(args.matplotlibrc)))
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    import matplotlib.pyplot as plt

    datasets = []
    times = []
    resolutions = []
    for key, label in SCHEMES:
        filedata = bin_convert.read_binary(_latest_output(args.run_root, key))
        datasets.append((key, label, *_assemble_density(filedata)))
        times.append(filedata["time"])
        resolutions.append((filedata["Nx1"], filedata["Nx2"]))

    if not np.allclose(times, args.expected_time, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"expected all final outputs at t={args.expected_time}; found {times}"
        )
    expected_resolution = (args.expected_nx1, args.expected_nx2)
    if any(resolution != expected_resolution for resolution in resolutions):
        raise RuntimeError(
            f"expected resolution {expected_resolution}; found {resolutions}"
        )

    levels = np.linspace(args.contour_min, args.contour_max, args.num_contours)
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 14.0), constrained_layout=False)
    for axis, panel_letter, (_, label, x1, x2, density) in zip(
            axes.flat, "abcdef", datasets):
        _draw_panel(axis, x1, x2, density, label, panel_letter, levels)
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.01, top=0.99,
                        wspace=0.12, hspace=0.06)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if args.panel_dir:
        panel_dir = os.path.abspath(args.panel_dir)
        os.makedirs(panel_dir, exist_ok=True)
        for panel_letter, (key, label, x1, x2, density) in zip("abcdef", datasets):
            panel_fig, panel_axis = plt.subplots(figsize=(2.1, 8.0))
            _draw_panel(panel_axis, x1, x2, density, label, panel_letter, levels)
            panel_fig.subplots_adjust(left=0.01, right=0.99, bottom=0.005, top=0.995)
            panel_fig.savefig(os.path.join(panel_dir, f"{key}.png"), dpi=150,
                              bbox_inches="tight", facecolor="white")
            plt.close(panel_fig)

    print(output)


if __name__ == "__main__":
    main()
