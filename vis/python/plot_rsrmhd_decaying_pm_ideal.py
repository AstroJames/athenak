#!/usr/bin/env python3
"""Plot ideal and Pm-scan decaying turbulence at a common resolution."""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

import plot_rsrmhd_decaying_turbulence as base


CASES = (
    ("ideal", r"ideal"),
    ("pm1", r"${\rm Pm}=1$"),
    ("pm10", r"${\rm Pm}=10$"),
    ("pm50", r"${\rm Pm}=50$"),
)


def rms(field):
    return float(np.sqrt(np.mean(np.square(field))))


def annotation_box():
    return {
        "facecolor": mpl.rcParams["axes.facecolor"],
        "alpha": 0.78,
        "edgecolor": "none",
    }


def configure_field_axis(axis, row, column):
    axis.set_aspect("equal")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_yticks((0.0, 0.5, 1.0))
    if row == len(CASES) - 1:
        axis.set_xlabel(r"$x/L$")
        if column == 0:
            axis.set_xticklabels(("0.0", "0.5", ""))
        elif column < 3:
            axis.set_xticklabels(("", "0.5", ""))
        else:
            axis.set_xticklabels(("", "0.5", "1.0"))
    else:
        axis.set_xticklabels([])
    if column == 0:
        axis.set_ylabel(r"$y/L$")
        if row == 0:
            axis.set_yticklabels(("", "0.5", "1.0"))
        elif row == len(CASES) - 1:
            axis.set_yticklabels(("0.0", "0.5", ""))
        else:
            axis.set_yticklabels(("", "0.5", ""))
    else:
        axis.set_yticklabels([])


def normalized_field_data(profiles):
    normalized = {}
    for key, _ in CASES:
        _, _, fields = profiles[key]
        omega_rms = rms(fields["omega"])
        current_rms = rms(fields["current"])
        normalized[key] = {
            "omega": fields["omega"]/omega_rms,
            "bmag": np.sqrt(
                fields["bx"]**2 + fields["by"]**2 + fields["bz"]**2
            )/0.15,
            "current": fields["current"]/current_rms,
            "chi": fields["chi"],
        }
    return normalized


def save_figure_pair(figure, output_stem, dpi):
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi,
                   bbox_inches="tight")


def plot_fields(profiles, output_stem, final_crossing_time):
    normalized = normalized_field_data(profiles)
    omega_limit = base.robust_symmetric_limit(
        [normalized[key]["omega"] for key, _ in CASES]
    )
    current_limit = base.robust_symmetric_limit(
        [normalized[key]["current"] for key, _ in CASES]
    )
    bmag_limit = base.robust_positive_limit(
        [normalized[key]["bmag"] for key, _ in CASES]
    )
    chi_limit = base.robust_positive_limit(
        [normalized[key]["chi"] for key, _ in CASES if key != "ideal"]
    )
    norms = (
        Normalize(-omega_limit, omega_limit),
        Normalize(0.0, bmag_limit),
        Normalize(0.0, chi_limit),
        Normalize(-current_limit, current_limit),
    )
    cmaps = ("RdBu_r", "viridis", "magma", "PuOr_r")
    labels = (
        r"$\omega^z/\sqrt{\langle\omega^i\omega_i\rangle}$",
        r"$\sqrt{B^iB_i/\langle B^iB_i\rangle_0}$",
        r"$\chi_\pi$",
        r"$J^z/\sqrt{\langle J^iJ_i\rangle}$",
    )
    figure_width = 7.5
    figure_height = 8.4
    panel_size = 1.58
    panel_gutter = 0.08
    panel_left = 0.70
    panel_bottom = 0.62
    fig = base.plt.figure(
        figsize=(figure_width, figure_height), constrained_layout=False,
    )
    axes = np.empty((len(CASES), 4), dtype=object)
    for row in range(len(CASES)):
        for column in range(4):
            left = panel_left + column*(panel_size + panel_gutter)
            bottom = panel_bottom + (len(CASES) - row - 1)*(
                panel_size + panel_gutter
            )
            axes[row, column] = fig.add_axes([
                left/figure_width, bottom/figure_height,
                panel_size/figure_width, panel_size/figure_height,
            ])

    for row, (key, case_label) in enumerate(CASES):
        x, y, _ = profiles[key]
        for column, (name, cmap, norm) in enumerate(
                zip(("omega", "bmag", "chi", "current"), cmaps, norms)):
            axis = axes[row, column]
            configure_field_axis(axis, row, column)
            if key == "ideal" and name == "chi":
                axis.text(
                    0.5, 0.5, r"$\pi^{\mu\nu}=0$", transform=axis.transAxes,
                    ha="center", va="center",
                )
            else:
                axis.pcolormesh(
                    x, y, normalized[key][name], shading="nearest", cmap=cmap,
                    norm=norm, rasterized=True,
                )
            if column == 0:
                axis.text(
                    0.03, 0.04, case_label, transform=axis.transAxes,
                    ha="left", va="bottom", bbox=annotation_box(),
                )
    fig.canvas.draw()
    for column, (cmap, norm, label) in enumerate(zip(cmaps, norms, labels)):
        panel_box = axes[0, column].get_position()
        colorbar_axis = fig.add_axes([
            panel_box.x0, panel_box.y1 + 0.028, panel_box.width, 0.012,
        ])
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.locator = MaxNLocator(nbins=3, prune="both")
        colorbar.update_ticks()
        colorbar.ax.tick_params(pad=1)
        colorbar.set_label(label, labelpad=5)
    axes[0, 2].text(
        0.97, 0.04,
        "$512^2$\n" + rf"$t/t_{{\rm eddy}}={final_crossing_time:.2f}$",
        transform=axes[0, 2].transAxes, ha="right", va="bottom",
        bbox=annotation_box(),
    )
    save_figure_pair(fig, output_stem, 220)
    base.plt.close(fig)


def normalized(history, name):
    return history[name]/history[name][0]


def plot_histories(histories, output_stem, eddy_time):
    fig, axes = base.plt.subplots(
        2, 3, figsize=(10.5, 5.8), sharex=True, constrained_layout=True,
    )
    colors = base.plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = ("-", "--", "-.", ":")
    specs = (
        ("ekin", r"$E_{\rm kin}/E_{\rm kin,0}$", "linear"),
        ("emag", r"$E_B/E_{B,0}$", "log"),
        ("eelec", r"$E_E/E_{E,0}$", "log"),
        ("enstrophy", r"$\langle\omega^i\omega_i\rangle/"
                      r"\langle\omega^i\omega_i\rangle_0$", "log"),
        ("current2", r"$\langle J^iJ_i\rangle/\langle J^iJ_i\rangle_0$", "log"),
        ("shear2", r"$\int \pi^2dV/\max(\int \pi^2dV)$", "linear"),
    )
    shear_scale = max(
        np.max(histories[key]["shear2"])
        for key, _ in CASES if key != "ideal"
    )
    handles = []
    legend_labels = []

    for axis, (name, label, scale) in zip(axes.flat, specs):
        for index, (key, case_label) in enumerate(CASES):
            history = histories[key]
            if name == "shear2" and key == "ideal":
                continue
            values = (
                history[name]/shear_scale
                if name == "shear2" else normalized(history, name)
            )
            line, = axis.plot(
                history["time"]/eddy_time, values,
                color=colors[index], linestyle=styles[index],
            )
            if name == "ekin":
                handles.append(line)
                legend_labels.append(case_label)
        axis.set_yscale(scale)
        if name == "shear2":
            axis.text(
                0.96, 0.92, label, transform=axis.transAxes,
                ha="right", va="top", bbox=annotation_box(),
            )
        else:
            axis.text(
                0.04, 0.08, label, transform=axis.transAxes,
                ha="left", va="bottom", bbox=annotation_box(),
            )
        axis.grid(alpha=0.18)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$t/t_{\rm eddy}$")
    fig.legend(
        handles, legend_labels, loc="outside upper center", ncol=4,
        frameon=False,
    )
    save_figure_pair(fig, output_stem, 220)
    base.plt.close(fig)


def main():
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
    base.plt = pyplot
    args.output_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    profiles = {}
    for key, _ in CASES:
        directory = args.root/key
        histories[key] = base.read_history(directory/f"{key}.user.hst")
        profiles[key] = base.read_profile(directory, key)

    eddy_time = 1.0/(args.peak_mode*args.velocity_rms)
    final_crossing_time = histories["ideal"]["time"][-1]/eddy_time
    field_output = args.output_dir/"decaying-turbulence-pm-512-fields"
    history_output = args.output_dir/"decaying-turbulence-pm-512-histories"
    plot_fields(profiles, field_output, final_crossing_time)
    plot_histories(histories, history_output, eddy_time)
    if args.html is not None:
        base.write_fragment(
            args.html, field_output.with_suffix(".png"),
            history_output.with_suffix(".png"),
            field_alt=("Vorticity, magnetic-field amplitude, causal shear stress, "
                       "and current for ideal and three Pm turbulence simulations"),
            history_alt=("Logarithmic and linear decay histories for ideal and three "
                         "Pm turbulence simulations at 512 squared resolution"),
        )


if __name__ == "__main__":
    main()
