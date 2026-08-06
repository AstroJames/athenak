#!/usr/bin/env python3
"""Plot Mach number and internal energy for the driven cooling-time scan."""

import argparse
import json
from pathlib import Path
import re

import matplotlib as mpl
import numpy as np

plt = None


CASE_ORDER = ("no_cooling", "tcool_0p1", "tcool_1", "tcool_10")
CASE_LABELS = {
    "no_cooling": r"no cooling",
    "tcool_0p1": r"$t_{\rm cool}/t_0=0.1$",
    "tcool_1": r"$t_{\rm cool}/t_0=1$",
    "tcool_10": r"$t_{\rm cool}/t_0=10$",
}


def read_history(path):
    """Read an Athena history file using its labeled header."""
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
    return {label: data[:, index] for label, index in labels.items()}


def load_cases(root):
    """Load histories and metadata for all completed scan members."""
    cases = {}
    for case_name in CASE_ORDER:
        case_dir = root/case_name
        record = json.loads((case_dir/"run.json").read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            raise RuntimeError(f"{case_name} is not completed")
        histories = list(case_dir.glob("*.user.hst"))
        if len(histories) != 1:
            raise RuntimeError(
                f"Expected one history in {case_dir}, found {len(histories)}"
            )
        cases[case_name] = (read_history(histories[0]), record)
    return cases


def analyze_cases(cases):
    """Return a compact machine-readable comparison of the four histories."""
    summary = {}
    for case_name, (history, record) in cases.items():
        volume = history["volume"]
        x = history["time"]/record["eddy_time"]
        mach = np.sqrt(history["mach2"]/volume)
        internal = history["eint"]/volume
        internal_ratio = internal/internal[0]
        vrms = np.sqrt(history["v2"]/volume)
        reynolds = vrms*record["drive_scale"]/record["viscosity"]
        sigma = history["sigma"]/volume
        second_half = x >= 0.5*x[-1]
        cooled = history.get("e_cool", np.zeros_like(x))
        ohmic_heating = history["q_ohm"]
        viscous_heating = history["q_visc"]
        energy_residual = (
            history["etot"] - history["etot"][0]
            - history["e_inj"] + cooled
        )
        summary[case_name] = {
            "cooling_time_over_eddy_time": record["cooling_time_over_eddy_time"],
            "turnovers": float(x[-1]),
            "initial_mach": float(mach[0]),
            "maximum_mach": float(np.max(mach)),
            "second_half_mach_mean": float(np.mean(mach[second_half])),
            "second_half_mach_std": float(np.std(mach[second_half])),
            "second_half_reynolds_mean": float(np.mean(reynolds[second_half])),
            "initial_internal_energy": float(internal[0]),
            "final_internal_energy": float(internal[-1]),
            "final_internal_energy_ratio": float(internal[-1]/internal[0]),
            "second_half_internal_energy_ratio_mean": float(np.mean(
                internal_ratio[second_half],
            )),
            "second_half_internal_energy_ratio_std": float(np.std(
                internal_ratio[second_half],
            )),
            "second_half_internal_energy_slope_per_turnover": float(np.polyfit(
                x[second_half], internal[second_half], 1,
            )[0]),
            "initial_magnetization": float(sigma[0]),
            "second_half_magnetization_mean": float(np.mean(sigma[second_half])),
            "second_half_ohmic_heating_mean": float(np.mean(
                ohmic_heating[second_half],
            )),
            "second_half_viscous_heating_mean": float(np.mean(
                viscous_heating[second_half],
            )),
            "injected_energy": float(history["e_inj"][-1]),
            "cooled_energy": float(cooled[-1]),
            "energy_audit_max_abs": float(np.max(np.abs(energy_residual))),
            "cooling_limiter_max": float(np.max(
                history.get("cool_limit", np.zeros_like(x)),
            )),
        }
    return summary


def make_plot(cases, output_base):
    """Create the two-row publication figure and its source-data table."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = ("-", "--", "-.", ":")
    figure, axes = plt.subplots(
        2, 1, figsize=(7.1, 6.8), sharex=True, layout="none",
    )
    figure.subplots_adjust(
        left=0.12, right=0.985, bottom=0.10, top=0.82, hspace=0.10,
    )
    case_colors = {}
    csv_columns = []
    csv_header = []
    for index, case_name in enumerate(CASE_ORDER):
        history, record = cases[case_name]
        volume = history["volume"]
        x = history["time"]/record["eddy_time"]
        mach = np.sqrt(history["mach2"]/volume)
        internal = history["eint"]/volume
        internal_ratio = internal/internal[0]
        color = "0.18" if case_name == "no_cooling" else colors[index - 1]
        case_colors[case_name] = color
        axes[0].plot(
            x, mach, color=color, linestyle=styles[index],
            label=CASE_LABELS[case_name],
        )
        axes[1].plot(
            x, internal_ratio, color=color, linestyle=styles[index],
        )
        csv_columns.extend((
            x, mach, internal_ratio, history["q_visc"], history["q_ohm"],
        ))
        csv_header.extend((
            f"{case_name}_t_over_t0", f"{case_name}_mach",
            f"{case_name}_eint_over_initial",
            f"{case_name}_q_visc", f"{case_name}_q_ohm",
        ))

    axes[0].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
        frameon=False, columnspacing=1.4, handlelength=2.7,
    )
    axes[0].set_ylabel(r"$\mathcal{M}_{\rm turb}$")
    axes[1].set_ylabel(r"$\langle e\rangle/\langle e\rangle_0$")
    axes[1].set_xlabel(r"$t/t_0$")
    axes[0].tick_params(labelbottom=False)
    for axis, label in zip(axes, (r"\textbf{(a)}", r"\textbf{(b)}")):
        axis.set_xlim(0.0, 10.0)
        axis.set_xticks(np.arange(0.0, 10.1, 2.0))
        axis.tick_params(direction="in", top=True, right=True)
        axis.text(
            0.02, 0.95, label, transform=axis.transAxes,
            ha="left", va="top",
        )
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.9)
    axes[0].text(
        0.98, 0.05,
        (r"$\beta_0=1$, $\sigma_0=0.1$, ${\rm Re}={\rm Rm}=100$"
         "\n" r"$t_{\rm corr}=t_0$, $\gamma=4/3$"),
        transform=axes[0].transAxes, ha="right", va="bottom", fontsize="small",
    )

    rate_scale = 1.0e3
    inset = axes[0].inset_axes([0.14, 0.08, 0.40, 0.36])
    for case_name in CASE_ORDER:
        history, record = cases[case_name]
        x = history["time"]/record["eddy_time"]
        inset.plot(
            x, rate_scale*history["q_visc"], color=case_colors[case_name],
            linestyle="-", linewidth=0.9,
        )
        inset.plot(
            x, rate_scale*history["q_ohm"], color=case_colors[case_name],
            linestyle="--", linewidth=0.9,
        )
    viscous_handle, = inset.plot(
        [], [], color="0.2", linestyle="-", linewidth=0.9,
        label=r"$Q_{\rm visc}$",
    )
    ohmic_handle, = inset.plot(
        [], [], color="0.2", linestyle="--", linewidth=0.9,
        label=r"$Q_{\rm Ohmic}$",
    )
    inset.set_xlim(0.0, 10.0)
    inset.set_ylim(0.0, 5.8)
    inset.set_xticks((0.0, 5.0, 10.0))
    inset.set_yticks((0.0, 2.0, 4.0))
    inset.set_ylabel(r"$Q\;(10^{-3})$", fontsize="x-small", labelpad=1.5)
    inset.tick_params(
        direction="in", top=True, right=True, labelsize="x-small",
    )
    inset.legend(
        handles=(viscous_handle, ohmic_handle), loc="upper center",
        ncol=2, frameon=False,
        fontsize="x-small", handlelength=1.6, handletextpad=0.4,
        columnspacing=0.8,
    )

    figure.savefig(output_base.with_suffix(".pdf"))
    figure.savefig(
        output_base.with_suffix(".png"), dpi=300,
        transparent=False, facecolor=mpl.rcParams["figure.facecolor"],
    )
    plt.close(figure)
    np.savetxt(
        output_base.with_suffix(".csv"), np.column_stack(csv_columns),
        header=" ".join(csv_header),
    )


def main():
    global plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument(
        "--matplotlibrc", type=Path,
        default=Path("~/.matplotlib/matplotlibrc").expanduser(),
    )
    args = parser.parse_args()
    mpl.rc_file(args.matplotlibrc)
    import matplotlib.pyplot as pyplot
    plt = pyplot
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.root)
    summary = analyze_cases(cases)
    (args.root/"summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    make_plot(cases, args.output_base)


if __name__ == "__main__":
    main()
