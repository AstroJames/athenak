#!/usr/bin/env python3
"""Plot final shell-integrated spectra for the 512^2 decaying-Pm scan."""

import argparse
import base64
from pathlib import Path

import matplotlib as mpl
import numpy as np

import bin_convert
import plot_rsrmhd_decaying_pm_ideal as comparison
import plot_rsrmhd_decaying_turbulence as base


def shell_integrated_spectrum(components):
    """Return 2D isotropic modal shells with Parseval normalization."""
    ny, nx = components[0].shape
    if any(component.shape != (ny, nx) for component in components):
        raise ValueError("All spectral components must share one Cartesian grid")

    mode_x = np.fft.fftfreq(nx)*nx
    mode_y = np.fft.fftfreq(ny)*ny
    shell = np.floor(np.hypot(mode_y[:, None], mode_x[None, :])).astype(int)
    modal_energy = np.zeros((ny, nx))
    normalization = float(nx*ny)
    for component in components:
        transform = np.fft.fft2(component)/normalization
        modal_energy += 0.5*np.square(np.abs(transform))

    maximum_shell = min(nx, ny)//2
    spectrum = np.bincount(
        shell.ravel(), weights=modal_energy.ravel(),
        minlength=maximum_shell + 1,
    )
    modes = np.arange(1, maximum_shell + 1)
    return modes, spectrum[1:maximum_shell + 1]


def spectra_from_profile(profile):
    _, _, fields = profile
    kinetic = shell_integrated_spectrum((fields["vx"], fields["vy"]))
    magnetic = shell_integrated_spectrum(
        (fields["bx"], fields["by"], fields["bz"])
    )
    return kinetic, magnetic


def fields_from_binary(path):
    """Merge a uniform 2D Athena binary dump and return three-velocity and B."""
    data = bin_convert.read_binary(path)
    nx = data["Nx1"]
    ny = data["Nx2"]
    nx_mb = data["nx1_out_mb"]
    ny_mb = data["nx2_out_mb"]
    names = ("velx", "vely", "velz", "bcc1", "bcc2", "bcc3")
    fields = {name: np.empty((ny, nx)) for name in names}

    for block, logical in enumerate(data["mb_logical"]):
        block_x, block_y, block_z, level = logical
        if block_z != 0 or level != 0:
            raise ValueError("Spectrum reader requires a uniform two-dimensional mesh")
        x_slice = slice(block_x*nx_mb, (block_x + 1)*nx_mb)
        y_slice = slice(block_y*ny_mb, (block_y + 1)*ny_mb)
        for name in names:
            fields[name][y_slice, x_slice] = data["mb_data"][name][block][0]

    lorentz_factor = np.sqrt(
        1.0 + fields["velx"]**2 + fields["vely"]**2 + fields["velz"]**2
    )
    fields["vx"] = fields["velx"]/lorentz_factor
    fields["vy"] = fields["vely"]/lorentz_factor
    fields["vz"] = fields["velz"]/lorentz_factor
    return data["time"], fields


def spectra_from_binary(path):
    time, fields = fields_from_binary(path)
    kinetic = shell_integrated_spectrum(
        (fields["vx"], fields["vy"], fields["vz"])
    )
    magnetic = shell_integrated_spectrum(
        (fields["bcc1"], fields["bcc2"], fields["bcc3"])
    )
    return time, (kinetic, magnetic)


def plot_spectra(spectra, output, crossing_time=None):
    fig, axes = base.plt.subplots(
        1, 2, figsize=(10.5, 4.4), sharex=True, constrained_layout=True,
    )
    colors = base.plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = ("-", "--", "-.", ":")
    handles = []
    labels = []

    for index, (key, case_label) in enumerate(comparison.CASES):
        for axis, spectral_index in zip(axes, (0, 1)):
            modes, power = spectra[key][spectral_index]
            line, = axis.loglog(
                modes, power, color=colors[index], linestyle=styles[index],
            )
            if spectral_index == 0:
                handles.append(line)
                labels.append(case_label)

    axes[0].set_ylabel(r"$E_{\rm kin}(k)$")
    axes[1].set_ylabel(r"$E_B(k)$")
    for axis in axes:
        axis.set_xlabel(r"$kL/(2\pi)$")
        axis.set_xlim(1.0, 256.0)
        axis.grid(alpha=0.18)
    if crossing_time is not None:
        axes[1].text(
            0.05, 0.08, rf"$t/t_{{\rm eddy}}={crossing_time:.4g}$",
            transform=axes[1].transAxes, ha="left", va="bottom",
            bbox=comparison.annotation_box(),
        )
    fig.legend(
        handles, labels, loc="outside upper center", ncol=4, frameon=False,
    )
    fig.savefig(output, dpi=145, bbox_inches="tight")
    base.plt.close(fig)


def write_fragment(path, image):
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    path.write_text(
        "<div id=\"rsrmhd-decaying-pm-spectra\">\n"
        "  <img style=\"width:100%;height:auto\" "
        "alt=\"Final shell-integrated kinetic and magnetic energy spectra for "
        "ideal and three magnetic-Prandtl-number decaying turbulence runs\" "
        f"src=\"data:image/png;base64,{encoded}\">\n"
        "</div>\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--snapshot", type=int)
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

    spectra = {}
    times = []
    for key, _ in comparison.CASES:
        if args.snapshot is None:
            profile = base.read_profile(args.root/key, key)
            spectra[key] = spectra_from_profile(profile)
        else:
            binary = args.root/key/"bin"/f"{key}.prim.{args.snapshot:05d}.bin"
            time, spectra[key] = spectra_from_binary(binary)
            times.append(time)

    crossing_time = None
    if times:
        if np.ptp(times) > 2.0e-3:
            raise ValueError("Compared binary snapshots do not share one time")
        eddy_time = 1.0/(args.peak_mode*args.velocity_rms)
        crossing_time = np.median(times)/eddy_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_spectra(spectra, args.output, crossing_time)
    if args.html is not None:
        write_fragment(args.html, args.output)


if __name__ == "__main__":
    main()
