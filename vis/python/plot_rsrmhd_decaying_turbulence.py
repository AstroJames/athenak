#!/usr/bin/env python3
"""Plot matched viscous, resistive, and visco-resistive turbulence runs."""

import argparse
import base64
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Normalize

plt = None


CASES = (
    ("viscous_hydro", "viscous hydro"),
    ("resistive", "resistive MHD"),
    ("viscoresistive", "visco-resistive MHD"),
)


def read_history(path):
    """Read the fixed-column user history written by the problem generator."""
    raw = np.loadtxt(path)
    names = (
        "time", "dt", "volume", "ekin", "emag", "eelec", "enstrophy",
        "current2", "shear2", "rho", "pgas", "divb2",
    )
    return {name: raw[:, index] for index, name in enumerate(names)}


def read_profile(directory, stem):
    """Merge rank-local final profiles into ordered two-dimensional arrays."""
    files = sorted(directory.glob(f"{stem}-rank*-profile.dat"))
    if not files:
        files = [directory / f"{stem}-profile.dat"]
    data = np.concatenate([np.loadtxt(path) for path in files])
    order = np.lexsort((data[:, 0], data[:, 1]))
    data = data[order]
    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    if data.shape[0] != x.size*y.size:
        raise ValueError(f"Profile for {stem} is not a complete Cartesian grid")
    names = (
        "x", "y", "vx", "vy", "bx", "by", "bz", "omega", "current",
        "pi_norm", "rho", "eint",
    )
    fields = {
        name: data[:, index].reshape(y.size, x.size)
        for index, name in enumerate(names)
    }
    fields["chi"] = fields["pi_norm"] / (
        fields["rho"] + (5.0/3.0)*fields["eint"]
    )
    return x, y, fields


def robust_symmetric_limit(fields, percentile=99.8):
    values = np.concatenate([np.abs(field).ravel() for field in fields])
    return max(float(np.percentile(values, percentile)), np.finfo(float).eps)


def robust_positive_limit(fields, percentile=99.8):
    values = np.concatenate([field.ravel() for field in fields])
    return max(float(np.percentile(values, percentile)), np.finfo(float).eps)


def configure_axis(axis, row, column):
    axis.set_aspect("equal")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    if row == len(CASES) - 1:
        axis.set_xlabel(r"$x/L$")
    else:
        axis.set_xticklabels([])
    if column == 0:
        axis.set_ylabel(r"$y/L$")
    else:
        axis.set_yticklabels([])


def plot_fields(profiles, output, final_crossing_time, cases=CASES):
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.15), constrained_layout=True)
    omega_limit = robust_symmetric_limit(
        [profiles[key][2]["omega"] for key, _ in cases]
    )
    current_limit = robust_symmetric_limit(
        [profiles[key][2]["current"] for key, _ in cases]
    )
    pi_limit = robust_positive_limit(
        [profiles[key][2]["chi"] for key, _ in cases]
    )
    norms = (
        Normalize(-omega_limit, omega_limit),
        Normalize(-current_limit, current_limit),
        Normalize(0.0, pi_limit),
    )
    cmaps = ("RdBu_r", "PuOr_r", "magma")
    field_names = ("omega", "current", "chi")
    column_labels = (
        r"$\omega^z L/v_{\rm rms,0}$",
        r"$J^z L/B_{\rm rms,0}$",
        r"$\sqrt{\pi^{\mu\nu}\pi_{\mu\nu}}/(e+p)$",
    )
    scales = (1.0/0.15, 1.0/0.15, 1.0)
    images = []

    for row, (key, case_label) in enumerate(cases):
        x, y, fields = profiles[key]
        for column, (field_name, cmap, norm, scale) in enumerate(
                zip(field_names, cmaps, norms, scales)):
            axis = axes[row, column]
            image = axis.pcolormesh(
                x, y, fields[field_name]*scale, shading="nearest", cmap=cmap,
                norm=Normalize(norm.vmin*scale, norm.vmax*scale), rasterized=True,
            )
            images.append(image)
            configure_axis(axis, row, column)
            if row == 0:
                axis.text(
                    0.03, 0.97, column_labels[column], transform=axis.transAxes,
                    ha="left", va="top",
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
                )
            if column == 0:
                axis.text(
                    0.03, 0.04, case_label, transform=axis.transAxes,
                    ha="left", va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
                )
            if key == "viscous_hydro" and column == 1:
                axis.text(0.5, 0.5, r"$B=E=J=0$", transform=axis.transAxes,
                          ha="center", va="center")
            if key == "resistive" and column == 2:
                axis.text(0.5, 0.5, r"$\pi^{\mu\nu}=0$", transform=axis.transAxes,
                          ha="center", va="center")

    fig.colorbar(images[0], ax=axes[:, 0], location="right", shrink=0.72, pad=0.02)
    fig.colorbar(images[4], ax=axes[:, 1], location="right", shrink=0.72, pad=0.02)
    fig.colorbar(images[8], ax=axes[:, 2], location="right", shrink=0.72, pad=0.02)
    axes[0, 2].text(
        0.97, 0.04, rf"$t/t_{{\rm eddy}}={final_crossing_time:.2f}$",
        transform=axes[0, 2].transAxes, ha="right", va="bottom",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    fig.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(fig)


def normalized(history, name):
    initial = history[name][0]
    if initial <= 0.0:
        return history[name]
    return history[name]/initial


def plot_histories(histories, output, eddy_time, cases=CASES):
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.6), sharex=True,
                             constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = ("-", "--", "-.")
    specs = (
        ("ekin", r"$E_{\rm kin}/E_{\rm kin,0}$", "linear"),
        ("emag", r"$E_B/E_{B,0}$", "linear"),
        ("eelec", r"$E_E/E_{E,0}$", "linear"),
        ("enstrophy", r"$\Omega/\Omega_0$", "linear"),
        ("current2", r"$\int J^2dV/(\int J^2dV)_0$", "linear"),
        ("shear2", r"$\int \pi^2dV/\max(\int \pi^2dV)$", "linear"),
    )
    handles = []
    labels = []
    shear_scale = max(np.max(histories[key]["shear2"]) for key, _ in cases)

    for axis, (name, annotation, yscale) in zip(axes.flat, specs):
        for index, (key, label) in enumerate(cases):
            history = histories[key]
            if np.all(history[name] == 0.0):
                continue
            if name == "shear2":
                values = history[name]/shear_scale
            else:
                values = normalized(history, name)
            line, = axis.plot(
                history["time"]/eddy_time, values, color=colors[index],
                linestyle=styles[index], label=label,
            )
            if name == "ekin":
                handles.append(line)
                labels.append(label)
        axis.set_yscale(yscale)
        if name == "shear2":
            axis.text(0.96, 0.92, annotation, transform=axis.transAxes,
                      ha="right", va="top")
        else:
            axis.text(0.04, 0.08, annotation, transform=axis.transAxes,
                      ha="left", va="bottom")
        axis.grid(alpha=0.18)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$t/t_{\rm eddy}$")
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               frameon=False)
    fig.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(fig)


def write_fragment(path, field_image, history_image,
                   field_alt="Final vorticity, current, and causal shear stress fields "
                             "for the three decaying-turbulence simulations",
                   history_alt="Normalized energy and gradient decay histories in "
                               "injection-scale eddy crossing times"):
    def encoded(image_path):
        return base64.b64encode(image_path.read_bytes()).decode("ascii")

    field_data = encoded(field_image)
    history_data = encoded(history_image)
    path.write_text(
        "<div id=\"rsrmhd-decaying-turbulence\" style=\"display:grid;gap:1rem\">\n"
        "  <img style=\"width:100%;height:auto\" "
        f"alt=\"{field_alt}\" "
        f"src=\"data:image/png;base64,{field_data}\">\n"
        "  <img style=\"width:100%;height:auto\" "
        f"alt=\"{history_alt}\" "
        f"src=\"data:image/png;base64,{history_data}\">\n"
        "</div>\n",
        encoding="utf-8",
    )


def main():
    global plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--velocity-rms", type=float, default=0.15)
    parser.add_argument("--peak-mode", type=float, default=2.5)
    parser.add_argument(
        "--matplotlibrc", type=Path,
        default=Path("~/.matplotlib/matplotlibrc").expanduser(),
    )
    args = parser.parse_args()
    mpl.rc_file(args.matplotlibrc)
    import matplotlib.pyplot as pyplot
    plt = pyplot
    args.output_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    profiles = {}
    for key, _ in CASES:
        directory = args.root / key
        histories[key] = read_history(directory / f"{key}.user.hst")
        profiles[key] = read_profile(directory, key)

    eddy_time = 1.0/(args.peak_mode*args.velocity_rms)
    final_crossing_time = histories["viscoresistive"]["time"][-1]/eddy_time
    field_output = args.output_dir / "decaying-turbulence-fields.png"
    history_output = args.output_dir / "decaying-turbulence-histories.png"
    plot_fields(profiles, field_output, final_crossing_time)
    plot_histories(histories, history_output, eddy_time)
    if args.html is not None:
        write_fragment(args.html, field_output, history_output)


if __name__ == "__main__":
    main()
