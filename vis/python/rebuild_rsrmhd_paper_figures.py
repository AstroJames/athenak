#!/usr/bin/env python3
"""Rebuild every manuscript figure from a permanent paper campaign."""

import argparse
import csv
import math
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import athena_read  # noqa: E402
import bin_convert  # noqa: E402
import plot_rsrmhd_decaying_pm_ideal as decaying  # noqa: E402
import plot_rsrmhd_decaying_turbulence as decaying_base  # noqa: E402
import plot_rsrmhd_driven_turbulence as driven  # noqa: E402
import plot_rsrmhd_ohmic_decay as ohmic  # noqa: E402
import plot_rsrmhd_viscous_phaseb as phaseb  # noqa: E402
import plot_rsrmhd_viscous_shear_scan as shear  # noqa: E402
import plot_rsrmhd_viscous_telegraph as telegraph  # noqa: E402


ETAS = (0.001, 0.003, 0.01, 0.03)


def eta_tag(eta):
    return f"eta{eta:g}".replace(".", "p")


def save_pair(figure, stem, dpi=220):
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight",
        transparent=False, facecolor=mpl.rcParams["figure.facecolor"],
    )
    mpl.pyplot.close(figure)


def load_current(path):
    data = athena_read.tab(str(path))
    order = np.argsort(data["x1v"])
    return data["x1v"][order], data["jz"][order]


def plot_current_sheet(root, figures):
    colors = mpl.pyplot.get_cmap("viridis")(
        np.linspace(0.08, 0.9, len(ETAS))
    )
    figure, axes = mpl.pyplot.subplots(1, 2, figsize=(12.0, 4.1))
    for eta, color in zip(ETAS, colors):
        tag = eta_tag(eta)
        for layout, linestyle in (("cell", "-"), ("face", "--")):
            case = root/"01_current_sheet"/f"{layout}_{tag}"
            profile = np.loadtxt(case/f"sheet_{layout}_{tag}-profile.dat")
            order = np.argsort(profile[:, 0])
            x = profile[order, 0]
            by = profile[order, 1]
            current_path = sorted(
                (case/"tab").glob(f"sheet_{layout}_{tag}.current.*.tab")
            )[-1]
            current_x, current = load_current(current_path)
            axes[0].plot(x, by, color=color, linestyle=linestyle)
            axes[1].plot(current_x, current, color=color,
                         linestyle=linestyle)
        analytic_x = np.linspace(-1.5, 1.5, 2000)
        analytic_by = np.array([
            math.erf(value/(2.0*np.sqrt(eta*10.0)))
            for value in analytic_x
        ])
        analytic_current = np.exp(-analytic_x**2/(4.0*eta*10.0))/np.sqrt(
            np.pi*eta*10.0
        )
        axes[0].plot(analytic_x, analytic_by, color=color, linestyle=":")
        axes[1].plot(
            analytic_x, analytic_current, color=color, linestyle=":"
        )

    axes[0].set_ylabel(r"$B^y$")
    axes[1].set_ylabel(r"$J^z=\partial_x B^y$")
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.set_xlabel(r"$x/L$")
        axis.set_xlim(-1.5, 1.5)
        axis.grid(False)
        axis.text(0.97 if label == "(a)" else 0.03, 0.95, label,
                  transform=axis.transAxes,
                  ha="right" if label == "(a)" else "left", va="top")
    layout_handles = (
        Line2D([0], [0], color="0.25", label=r"cell-centred $E^i$"),
        Line2D([0], [0], color="0.25", linestyle="--",
               label=r"face-centred $E^i$"),
        Line2D([0], [0], color="0.25", linestyle=":", label="analytic"),
    )
    axes[0].legend(handles=layout_handles, loc="upper left", frameon=False)
    eta_handles = [
        Line2D([0], [0], color=color, label=rf"$\eta={eta:g}$")
        for eta, color in zip(ETAS, colors)
    ]
    axes[0].legend(handles=eta_handles, loc="lower right", frameon=False,
                   ncols=2)
    axes[1].legend(handles=layout_handles, loc="upper right", frameon=False)
    axes[0].text(0.03, 0.94, r"$N_x=1024,\quad t/t_c=10$",
                 transform=axes[0].transAxes, ha="left", va="top")
    figure.tight_layout()
    save_pair(figure, figures/"rsrmhd_eta_sweep_1024")


def read_tab_columns(path):
    return np.loadtxt(path, comments="#")


def field_from_tab(path):
    data = read_tab_columns(path)
    x = data[:, -3]
    y = data[:, -2]
    values = data[:, -1]
    order = np.lexsort((x, y))
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    return x_unique, y_unique, values[order].reshape(y_unique.size, x_unique.size)


def field_from_binary(path):
    data = bin_convert.read_binary(path)
    nx = data["Nx1"]
    ny = data["Nx2"]
    nx_mb = data["nx1_out_mb"]
    ny_mb = data["nx2_out_mb"]
    field = np.empty((ny, nx))
    variable = next(iter(data["mb_data"]))
    for block, logical in enumerate(data["mb_logical"]):
        block_x, block_y, block_z, level = logical
        if block_z != 0 or level != 0:
            raise ValueError("Vortex reader requires a uniform 2D mesh")
        x_slice = slice(block_x*nx_mb, (block_x + 1)*nx_mb)
        y_slice = slice(block_y*ny_mb, (block_y + 1)*ny_mb)
        field[y_slice, x_slice] = data["mb_data"][variable][block][0]
    dx = (data["x1max"] - data["x1min"])/nx
    dy = (data["x2max"] - data["x2min"])/ny
    x = np.linspace(data["x1min"] + dx/2.0, data["x1max"] - dx/2.0, nx)
    y = np.linspace(data["x2min"] + dy/2.0, data["x2max"] - dy/2.0, ny)
    return x, y, field


def profile_from_tab(path):
    data = read_tab_columns(path)
    order = np.argsort(data[:, -2])
    return data[order, -2], data[order, -1]


def vortex_error_row(path):
    values = np.loadtxt(path, comments="#")
    if values.ndim > 1:
        values = values[-1]
    return values


def plot_charged_vortex(root, figures):
    vortex_root = root/"03_charged_vortex"
    layouts = (
        ("cell", "cell-centred", "cell", "C0", "o", "--"),
        ("face", "face-centred", "face", "C1", "s", "-."),
    )
    resolutions = np.asarray((32, 64, 128, 256, 512))

    figure = mpl.pyplot.figure(figsize=(11.0, 4.3), layout="none")
    grid = figure.add_gridspec(
        1, 2, left=0.075, right=0.99, bottom=0.17, top=0.84, wspace=0.32,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    analytic_x = np.linspace(-5.0, 5.0, 2000)
    analytic_q = 0.7/(analytic_x**2 + 1.0)**2
    axes[0].plot(analytic_x, analytic_q, color="0.15", linewidth=1.8,
                 label=r"equilibrium, $t/t_c=0$")
    for stem, label, _, color, _, linestyle in layouts:
        profile_path = sorted(
            (vortex_root/f"{stem}_512"/"tab").glob(
                f"vortex_{stem}.charge_profile.*.tab"
            )
        )[-1]
        x, charge = profile_from_tab(profile_path)
        axes[0].plot(x, charge, color=color, linestyle=linestyle,
                     label=rf"{label}, $t/t_c=5$")

    convergence = {}
    for stem, _, short_label, color, marker, _ in layouts:
        rows = np.asarray([
            vortex_error_row(
                vortex_root/f"{stem}_{resolution}"/f"vortex_{stem}-errs.dat"
            )
            for resolution in resolutions
        ])
        l1_pressure = rows[:, 4]
        linf_pressure = rows[:, 5]
        convergence[stem] = {
            "l1": np.polyfit(np.log(resolutions), np.log(l1_pressure), 1)[0],
            "linf": np.polyfit(np.log(resolutions), np.log(linf_pressure), 1)[0],
        }
        axes[1].loglog(
            resolutions, l1_pressure, color=color, marker=marker,
            linestyle="-", label=rf"{short_label}, $L_1$",
        )
        axes[1].loglog(
            resolutions, linf_pressure, color=color, marker=marker,
            markerfacecolor="none", linestyle="--",
            label=rf"{short_label}, $L_\infty$",
        )

    reference = 0.45*min(
        vortex_error_row(
            vortex_root/"cell_32"/"vortex_cell-errs.dat"
        )[4],
        vortex_error_row(
            vortex_root/"face_32"/"vortex_face-errs.dat"
        )[4],
    )*(resolutions/resolutions[0])**-2
    axes[1].loglog(resolutions, reference, color="0.2", linestyle=":",
                   label=r"$\propto N_x^{-2}$")

    axes[0].set_xlim(-5.0, 5.0)
    axes[0].set_ylim(-0.04, 0.74)
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$q(x,y\simeq0)$")
    axes[0].legend(frameon=False, loc="upper right", fontsize=11,
                   handlelength=2.2, labelspacing=0.35)
    axes[0].text(0.03, 0.96, r"$512^2,\quad\eta=10^{-3}$",
                 transform=axes[0].transAxes, ha="left", va="top",
                 fontsize=11)
    axes[1].set_xlabel(r"$N_x=N_y$")
    axes[1].set_ylabel(r"$\|p(t)-p(0)\|_{r<5}$")
    axes[1].set_xticks(resolutions)
    axes[1].set_xticklabels([str(value) for value in resolutions])
    axes[1].legend(frameon=False, loc="lower center", ncols=3,
                   bbox_to_anchor=(0.5, 1.02),
                   fontsize=11, columnspacing=0.8, handletextpad=0.4,
                   handlelength=2.2, labelspacing=0.35)
    for label, axis in zip(("(a)", "(b)"), axes):
        axis.grid(False)
        axis.text(0.03, 0.05, label, transform=axis.transAxes,
                  ha="left", va="bottom")
    save_pair(figure, figures/"rsrmhd_charged_vortex_staggering_compare")
    with (figures/"rsrmhd_charged_vortex_convergence.json").open(
            "w", encoding="utf-8") as stream:
        import json
        json.dump(convergence, stream, indent=2)
        stream.write("\n")


def plot_ohmic(root, figures):
    scan_cases = []
    fiducial = None
    for eta in ETAS:
        tag = eta_tag(eta)
        snapshots = ohmic.load_snapshots(
            root/"02_ohmic_harris"/tag, f"ohmic_{tag}"
        )
        scan_cases.append((eta, snapshots))
        if eta == 0.01:
            fiducial = snapshots
    slopes = ohmic.plot(
        fiducial, scan_cases, figures/"rsrmhd_ohmic_harris_decay",
        0.01, 0.02, 1.0,
    )
    np.savetxt(
        figures/"rsrmhd_ohmic_harris_decay_slopes.txt", slopes,
        header="eta peak_slope rms_width_slope",
    )


def telegraph_rows(root):
    scans = (
        ((0.01, 0.2), (0.03, 0.2), (0.05, 0.2)),
        ((0.02, 0.1), (0.02, 0.2), (0.02, 0.4)),
    )
    rows = []
    for scan_id, cases in enumerate(scans):
        scan_name = "fixed_tau" if scan_id == 0 else "fixed_nu"
        for nu, tau in cases:
            rows.append((scan_id, nu, tau, 0.0, 1.0e-4, 1.0e-4, 0.0, 0.0))
            parameter = f"nu{nu:g}_tau{tau:g}".replace(".", "p")
            for sample in range(1, 11):
                path = (root/"05_viscous_telegraph"/scan_name/parameter/
                        f"t{sample:02d}"/"rsrmhd_viscous_telegraph-errs.dat")
                values = np.loadtxt(path)
                rows.append((scan_id, nu, tau, sample/10.0,
                             values[2], values[3], values[4], values[5]))
    return np.asarray(rows), scans


def plot_telegraph(root, figures):
    rows, scans = telegraph_rows(root)
    with (figures/"viscous_telegraph_sweep.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scan", "nu_sh", "tau_pi", "time",
                         "velocity_athenak", "velocity_exact",
                         "stress_athenak", "stress_exact"))
        writer.writerows(rows)
    telegraph.make_plot(rows, scans, 1.0e-4, 8.0/3.0, figures)


def plot_phaseb(root, figures):
    diffusion = [[0.0, 1.0e-4, 1.0/128.0]]
    for sample in range(1, 11):
        values = np.loadtxt(
            root/"06_viscous_phaseb"/"diffusion"/f"t{sample:02d}"/
            "rsrmhd_viscous_telegraph-errs.dat"
        )
        diffusion.append((sample/5.0, values[2], values[6]))
    timestep = []
    for resolution in (64, 128, 256, 512):
        values = np.loadtxt(
            root/"06_viscous_phaseb"/"timestep"/f"n{resolution}"/
            "rsrmhd_viscous_telegraph-errs.dat"
        )
        timestep.append((1.0/resolution, values[6]))
    longitudinal = [[0.0, 1.0e-4, 0.0, 0.0]]
    for sample in range(1, 11):
        values = np.loadtxt(
            root/"06_viscous_phaseb"/"longitudinal"/f"t{sample:02d}"/
            "rsrmhd_viscous_longitudinal-amps.dat"
        )
        longitudinal.append((sample/10.0, values[0], values[1], values[2]))
    arrays = tuple(np.asarray(values) for values in
                   (diffusion, timestep, longitudinal))
    phaseb.write_csv(figures/"viscous_phaseb_validation.csv", *arrays)
    phaseb.make_plot(*arrays, figures)


def plot_shear(root, figures):
    profiles = []
    for viscosity, tag in zip(shear.VISCOSITIES, shear.TAGS):
        path = root/"07_viscous_shear_layer"/f"nu{tag}"
        data = np.loadtxt(path/f"shear_nu{tag}-profile.dat")
        data = data[np.argsort(data[:, 0])]
        x, u2, pi12 = data.T
        velocity = u2/np.sqrt(1.0 + u2**2)
        dx = x[1] - x[0]
        vorticity = (np.roll(velocity, -1)-np.roll(velocity, 1))/(2.0*dx)
        profiles.append((viscosity, x, vorticity, pi12))
    shear.plot(profiles, figures/"viscous_shear_scan")


def plot_khi(root, figures, repo):
    inviscid = sorted(
        (root/"09_viscous_khi"/"inviscid_512").glob(
            "kh_inviscid-rank*-profile.dat"
        )
    )
    viscous = sorted(
        (root/"09_viscous_khi"/"viscous_512").glob(
            "kh_viscous-rank*-profile.dat"
        )
    )
    command = [
        sys.executable, str(repo/"vis/python/plot_rsrmhd_viscous_kh.py"),
        "--inviscid", *map(str, inviscid), "--viscous", *map(str, viscous),
        "--output", str(figures/"viscous_kh_fields.png"), "--time", "4",
        "--viscosity", "0.0001",
    ]
    subprocess.run(command, check=True)


def plot_decaying(root, figures):
    histories = {}
    profiles = {}
    data_root = root/"10_decaying_turbulence"
    for key, _ in decaying.CASES:
        histories[key] = decaying_base.read_history(
            data_root/key/f"{key}.user.hst"
        )
        profiles[key] = decaying_base.read_profile(data_root/key, key)
    eddy_time = 1.0/(2.5*0.15)
    final_time = histories["ideal"]["time"][-1]/eddy_time
    decaying.plot_fields(
        profiles, figures/"decaying_turbulence_pm_fields", final_time
    )
    decaying.plot_histories(
        histories, figures/"decaying_turbulence_pm_histories", eddy_time
    )


def plot_driven(root, figures):
    history_path = (
        root/"11_driven_turbulence"/"n64"/
        "driven3d64_mach0p5_re50.user.hst"
    )
    history = driven.read_history(history_path)
    driven.plot_histories(
        history, figures/"driven_turbulence_integrals",
        1.449137674618944, 0.5, 0.003450327796711771,
        0.003450327796711771,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--matplotlibrc", type=Path,
        default=Path("~/.matplotlib/matplotlibrc").expanduser(),
    )
    parser.add_argument(
        "--only", action="append",
        choices=("current_sheet", "ohmic_harris", "charged_vortex",
                 "telegraph", "phaseb", "shear_layer", "khi",
                 "decaying", "driven"),
        help="rebuild only the selected figure family",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    figures = root/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    mpl.rc_file(args.matplotlibrc)
    import matplotlib.pyplot as pyplot
    mpl.pyplot = pyplot
    decaying_base.plt = pyplot
    driven.plt = pyplot

    actions = {
        "current_sheet": lambda: plot_current_sheet(root, figures),
        "ohmic_harris": lambda: plot_ohmic(root, figures),
        "charged_vortex": lambda: plot_charged_vortex(root, figures),
        "telegraph": lambda: plot_telegraph(root, figures),
        "phaseb": lambda: plot_phaseb(root, figures),
        "shear_layer": lambda: plot_shear(root, figures),
        "khi": lambda: plot_khi(root, figures, repo),
        "decaying": lambda: plot_decaying(root, figures),
        "driven": lambda: plot_driven(root, figures),
    }
    selected = args.only if args.only else actions
    for name in selected:
        actions[name]()
    shutil.copy2(__file__, root/"scripts"/Path(__file__).name)


if __name__ == "__main__":
    main()
