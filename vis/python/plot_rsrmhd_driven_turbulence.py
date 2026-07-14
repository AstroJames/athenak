#!/usr/bin/env python3
"""Plot integral histories for mechanically driven visco-resistive SRMHD."""

import argparse
import base64
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np

plt = None


def read_history(path):
    """Read an Athena history file using its labeled column header."""
    labels = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            for number, label in re.findall(r"\[(\d+)\]=(\S+)", line):
                labels[label] = int(number) - 1
    data = np.atleast_2d(np.loadtxt(path))
    if "magnetizat" in labels:
        labels["sigma"] = labels["magnetizat"]
    if "alfven_spe" in labels:
        labels["v_alfven"] = labels["alfven_spe"]
    return {label: data[:, index] for label, index in labels.items()}


def cumulative_trapezoid(values, time):
    """Return the cumulative trapezoidal integral with zero initial value."""
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5*(values[1:] + values[:-1])*np.diff(time)
    )
    return result


def positive(values):
    """Mask non-positive values for logarithmic plotting."""
    return np.where(values > 0.0, values, np.nan)


def normalized_by_max(values):
    """Normalize a nonnegative history by its maximum."""
    maximum = np.max(values)
    return values/maximum if maximum > 0.0 else values


def panel_label(axis, label):
    axis.text(
        0.025, 0.965, label, transform=axis.transAxes,
        ha="left", va="top", fontweight="bold",
    )


def legend_above(axis, ncol):
    """Place a panel legend in reserved space above the data rectangle."""
    return axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.015),
        frameon=False, ncol=ncol, columnspacing=0.8,
        handlelength=2.2, borderaxespad=0.0,
    )


def configure_axes(axes, x_max):
    for axis in axes.flat:
        axis.axvspan(0.0, 1.0, color="0.92", zorder=-20)
        axis.set_xlim(0.0, x_max)
        axis.set_xticks(np.arange(0.0, x_max + 0.1, 2.0))
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.tick_params(direction="in", top=True, right=True)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$t/t_0$")


def plot_histories(history, output_base, eddy_time, drive_scale,
                   viscosity, resistivity):
    """Create PDF, PNG, and CSV versions of the publication figure."""
    time = history["time"]
    x = time/eddy_time
    volume = history["volume"]
    mach = np.sqrt(history["mach2"]/volume)
    vrms = np.sqrt(history["v2"]/volume)
    reynolds = vrms*drive_scale/viscosity
    magnetic_reynolds = vrms*drive_scale/resistivity
    target_vrms = drive_scale/eddy_time
    mean_magnetization = history["sigma"]/volume
    mean_alfven_speed = history["v_alfven"]/volume
    target_alfven_mach = target_vrms/mean_alfven_speed

    delta_internal = history["eint"] - history["eint"][0]
    delta_kinetic = history["ekin"] - history["ekin"][0]
    electromagnetic = history["emag"] + history["eelec"]
    delta_electromagnetic = electromagnetic - electromagnetic[0]
    delta_entropy = history["entropy"] - history["entropy"][0]
    cumulative_ohmic = cumulative_trapezoid(history["q_ohm"], time)
    cumulative_viscous = cumulative_trapezoid(history["q_visc"], time)
    cooling_enabled = "e_cool" in history
    cooling_power = history.get("cool_power", np.zeros_like(time))
    cooled_energy = history.get("e_cool", np.zeros_like(time))
    cooled_momentum1 = history.get("p_cool1", np.zeros_like(time))
    cooled_momentum2 = history.get("p_cool2", np.zeros_like(time))
    cooled_momentum3 = history.get("p_cool3", np.zeros_like(time))
    limited_cooling = history.get("e_cool_lim", np.zeros_like(time))

    energy_residual = (
        history["etot"] - history["etot"][0] - history["e_inj"]
        + cooled_energy
    )
    momentum_residual = np.sqrt(
        (history["mom1"] - history["mom1"][0] - history["p_inj1"]
         + cooled_momentum1)**2
        + (history["mom2"] - history["mom2"][0] - history["p_inj2"]
           + cooled_momentum2)**2
        + (history["mom3"] - history["mom3"][0] - history["p_inj3"]
           + cooled_momentum3)**2
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_max = float(np.ceil(x[-1]))
    fig, axes = plt.subplots(
        2, 3, figsize=(7.2, 5.2), sharex=True, layout="constrained",
    )
    configure_axes(axes, x_max)

    axis = axes[0, 0]
    axis.plot(x, mach, color=colors[0], linewidth=1.5,
              label=r"$\mathcal{M}_{\rm turb}$")
    axis.axhline(0.5, color="0.2", linestyle=":", linewidth=1.0,
                 label=r"target")
    mach_min = min(float(np.min(mach)), 0.5)
    mach_max = max(float(np.max(mach)), 0.5)
    mach_pad = max(0.005, 0.06*(mach_max - mach_min))
    axis.set_ylim(max(0.0, mach_min - mach_pad), mach_max + mach_pad)
    axis.set_ylabel(r"$\mathcal{M}_{\rm turb}$")
    legend_above(axis, ncol=2)
    post = x >= 0.5*x[-1]
    axis.text(
        0.98, 0.035,
        (r"$\langle\mathcal{M}_{\rm turb}\rangle_{t>5t_0}=%.3f$" "\n"
         r"$\langle{\rm Re}\rangle=\langle{\rm Rm}\rangle=%.1f$" "\n"
         r"$\langle\mathcal{M}_{A,{\rm target}}\rangle=%.3f$" "\n"
         r"$\langle\sigma\rangle=%.3f$")
        % (np.mean(mach[post]), np.mean(reynolds[post]),
           np.mean(target_alfven_mach[post]), np.mean(mean_magnetization[post])),
        transform=axis.transAxes, ha="right", va="bottom", fontsize=6.7,
        bbox={"facecolor": "white", "alpha": 0.78,
              "edgecolor": "none", "pad": 0.8},
    )
    panel_label(axis, r"\textbf{(a)}")

    axis = axes[0, 1]
    axis.axhline(0.0, color="0.65", linewidth=0.6)
    axis.plot(x, history["e_inj"], color="0.15", linewidth=1.5,
              label=r"$E_{\rm inj}$")
    if cooling_enabled:
        axis.plot(x, cooled_energy, color=colors[2], linewidth=1.4,
                  label=r"$E_{\rm cool}$")
    axis.plot(x, delta_internal, color=colors[3], linewidth=1.4,
              linestyle="--", label=r"$\Delta E_{\rm int}$")
    axis.plot(x, delta_kinetic, color=colors[0], linewidth=1.4,
              linestyle="-.", label=r"$\Delta E_{\rm kin}$")
    axis.plot(x, delta_electromagnetic, color=colors[4], linewidth=1.4,
              linestyle=":", label=r"$\Delta E_{\rm EM}$")
    axis.set_ylabel(r"volume-integrated energy")
    legend_above(axis, ncol=2)
    panel_label(axis, r"\textbf{(b)}")

    axis = axes[0, 2]
    axis.plot(x, positive(history["q_visc"]), color=colors[3],
              linewidth=1.4, label=r"$\mathcal{Q}_{\rm visc}$")
    axis.plot(x, positive(history["q_ohm"]), color=colors[1],
              linewidth=1.4, linestyle="--",
              label=r"$\mathcal{Q}_{\rm Ohm}$")
    if cooling_enabled:
        axis.plot(x, positive(cooling_power), color=colors[2],
                  linewidth=1.4, linestyle=":",
                  label=r"$P_{\rm cool}$")
    axis.set_yscale("log")
    positive_heating = np.concatenate([
        history["q_visc"][history["q_visc"] > 0.0],
        history["q_ohm"][history["q_ohm"] > 0.0],
        cooling_power[cooling_power > 0.0],
    ])
    heating_max = float(np.max(positive_heating))
    heating_min = max(float(np.min(positive_heating)), heating_max*1.0e-7)
    axis.set_ylim(0.8*heating_min, 1.5*heating_max)
    axis.set_ylabel(r"volume-integrated rate")
    legend_above(axis, ncol=3)
    panel_label(axis, r"\textbf{(c)}")

    axis = axes[1, 0]
    axis.plot(x, delta_internal, color=colors[0], linewidth=1.4,
              label=r"$\Delta E_{\rm int}$")
    axis.plot(x, cumulative_viscous, color=colors[3], linewidth=1.4,
              linestyle="--",
              label=r"$\int\mathcal{Q}_{\rm visc}\,\mathrm{d}t$")
    axis.plot(x, cumulative_ohmic, color=colors[1], linewidth=1.4,
              linestyle="-.",
              label=r"$\int\mathcal{Q}_{\rm Ohm}\,\mathrm{d}t$")
    axis.plot(x, delta_entropy, color=colors[2], linewidth=1.4,
              linestyle=":", label=r"$\Delta S_{\rm proxy}$")
    if cooling_enabled:
        axis.plot(x, cooled_energy, color="0.15", linewidth=1.3,
                  label=r"$E_{\rm cool}$")
    axis.set_ylabel(r"cumulative integral or change")
    legend_above(axis, ncol=2)
    panel_label(axis, r"\textbf{(d)}")

    axis = axes[1, 1]
    structural = (
        ("enstrophy", colors[0], "-",
         r"$\int\omega^i\omega_i\,\mathrm{d}V$"),
        ("current2", colors[1], "--",
         r"$\int J^iJ_i\,\mathrm{d}V$"),
        ("shear2", colors[3], "-.",
         r"$\int\pi^{\mu\nu}\pi_{\mu\nu}\,\mathrm{d}V$"),
        ("divv2", colors[2], ":",
         r"$\int\theta^2\,\mathrm{d}V$"),
    )
    for name, color, style, label in structural:
        axis.plot(
            x, normalized_by_max(history[name]), color=color,
            linewidth=1.35, linestyle=style, label=label,
        )
    axis.set_ylim(-0.03, 1.06)
    axis.set_ylabel(r"$\mathcal{I}/\max(\mathcal{I})$")
    legend_above(axis, ncol=2)
    panel_label(axis, r"\textbf{(e)}")

    axis = axes[1, 2]
    audit_floor = 1.0e-17
    axis.plot(
        x, np.maximum(np.abs(energy_residual), audit_floor),
        color=colors[0], linewidth=1.4,
        label=(r"$|\Delta E_{\rm tot}-E_{\rm inj}+E_{\rm cool}|$"
               if cooling_enabled else
               r"$|\Delta E_{\rm tot}-E_{\rm inj}|$"),
    )
    axis.plot(
        x, np.maximum(momentum_residual, audit_floor),
        color=colors[1], linewidth=1.4, linestyle="--",
        label=(r"$\|\Delta\bm{P}-\bm{P}_{\rm inj}"
               r"+\bm{P}_{\rm cool}\|_2$"
               if cooling_enabled else
               r"$\|\Delta\bm{P}-\bm{P}_{\rm inj}\|_2$"),
    )
    axis.set_yscale("log")
    audit_max = max(float(np.max(np.abs(energy_residual))),
                    float(np.max(momentum_residual)), 10.0*audit_floor)
    axis.set_ylim(0.7*audit_floor, 2.0*audit_max)
    axis.set_ylabel(r"conservation residual")
    legend_above(axis, ncol=1)
    panel_label(axis, r"\textbf{(f)}")

    figure_metadata = {
        "Title": "Integral histories of mechanically driven "
                 "visco-resistive SRMHD turbulence",
        "Author": "James R. Beattie et al.",
        "Subject": "Ten injection-scale eddy turnover times",
    }
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(
        pdf_path, bbox_inches="tight", metadata=figure_metadata,
    )
    fig.savefig(
        png_path, dpi=300, bbox_inches="tight",
        metadata={"Title": figure_metadata["Title"]},
    )
    plt.close(fig)

    columns = np.column_stack([
        x, mach, reynolds, magnetic_reynolds, history["e_inj"],
        delta_internal, delta_kinetic, delta_electromagnetic,
        history["q_ohm"], history["q_visc"], cumulative_ohmic,
        cumulative_viscous, delta_entropy, energy_residual,
        momentum_residual, mean_alfven_speed, target_alfven_mach,
        mean_magnetization, cooling_power, cooled_energy, limited_cooling,
    ])
    header = (
        "t_over_t0 mach_turb re rm e_inj delta_eint delta_ekin "
        "delta_electromagnetic q_ohm q_visc int_q_ohm_dt "
        "int_q_visc_dt delta_entropy energy_residual momentum_residual "
        "mean_alfven_speed target_alfven_mach mean_magnetization "
        "cool_power e_cool e_cool_limited"
    )
    np.savetxt(output_base.with_suffix(".csv"), columns, header=header)
    return pdf_path, png_path


def write_fragment(path, image_path):
    """Write an inline preview fragment for the Codex visualization surface."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    path.write_text(
        '<div id="driven-turbulence-publication-figure">\n'
        '  <img style="width:100%;height:auto" '
        'alt="Six-panel publication figure showing driven relativistic '
        'turbulence integral histories over ten eddy turnover times" '
        f'src="data:image/png;base64,{encoded}">\n'
        '</div>\n',
        encoding="utf-8",
    )


def main():
    global plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--eddy-time", type=float, required=True)
    parser.add_argument("--drive-scale", type=float, required=True)
    parser.add_argument("--viscosity", type=float, required=True)
    parser.add_argument("--resistivity", type=float, required=True)
    parser.add_argument("--html", type=Path)
    parser.add_argument(
        "--matplotlibrc", type=Path,
        default=Path("~/.matplotlib/matplotlibrc").expanduser(),
    )
    args = parser.parse_args()

    mpl.rc_file(args.matplotlibrc)
    mpl.rcParams.update({
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.35,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
    })
    import matplotlib.pyplot as pyplot
    plt = pyplot

    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    history = read_history(args.history)
    _, png_path = plot_histories(
        history, args.output_base, args.eddy_time, args.drive_scale,
        args.viscosity, args.resistivity,
    )
    if args.html is not None:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        write_fragment(args.html, png_path)


if __name__ == "__main__":
    main()
