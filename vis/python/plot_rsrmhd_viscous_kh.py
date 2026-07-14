#!/usr/bin/env python3
"""Plot two-dimensional relativistic Kelvin--Helmholtz benchmark fields."""

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.rc_file("/Users/beattijr/.matplotlib/matplotlibrc")
import matplotlib.pyplot as plt
import numpy as np


def load_field(paths):
    data = np.concatenate([np.loadtxt(path) for path in paths], axis=0)
    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    order = np.lexsort((data[:, 0], data[:, 1]))
    data = data[order]
    fields = {
        "vx": data[:, 2].reshape(y.size, x.size),
        "vy": data[:, 3].reshape(y.size, x.size),
        "rho": data[:, 4].reshape(y.size, x.size),
        "pi12": data[:, 8].reshape(y.size, x.size),
        "omega": data[:, 9].reshape(y.size, x.size),
    }
    return x, y, fields


def symmetric_limit(arrays):
    return max(float(np.max(np.abs(array))) for array in arrays)


def image_extent(x, y):
    """Return cell-edge limits for cell-centered, uniformly spaced data."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return (x[0] - 0.5*dx, x[-1] + 0.5*dx,
            y[0] - 0.5*dy, y[-1] + 0.5*dy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inviscid", required=True, nargs='+')
    parser.add_argument("--viscous", required=True, nargs='+')
    parser.add_argument("--output", required=True)
    parser.add_argument("--time", type=float, default=2.0)
    parser.add_argument("--viscosity", type=float, default=0.0001)
    args = parser.parse_args()

    datasets = [load_field(args.inviscid), load_field(args.viscous)]
    omega_limit = symmetric_limit([item[2]["omega"] for item in datasets])
    vy_limit = symmetric_limit([item[2]["vy"] for item in datasets])
    pi_limit = symmetric_limit([item[2]["pi12"] for item in datasets])

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.7), constrained_layout=True,
                             sharex=True, sharey=True)
    row_labels = [r"$\nu_{\rm sh}=0$",
                  rf"$\nu_{{\rm sh}}={args.viscosity:g}$"]
    specs = [
        ("omega", r"$\omega^z$", omega_limit),
        ("vy", r"$v^y$", vy_limit),
        ("pi12", r"$\pi^{xy}$", pi_limit),
    ]
    for row, (x, y, fields) in enumerate(datasets):
        extent = image_extent(x, y)
        for column, (field, label, limit) in enumerate(specs):
            ax = axes[row, column]
            if limit == 0.0:
                image = ax.imshow(fields[field], origin="lower", extent=extent,
                                  interpolation="nearest", cmap="RdBu_r",
                                  vmin=-1.0, vmax=1.0)
            else:
                image = ax.imshow(fields[field], origin="lower", extent=extent,
                                  interpolation="nearest", cmap="RdBu_r",
                                  vmin=-limit, vmax=limit)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.text(0.03, 0.95, row_labels[row], transform=ax.transAxes,
                    ha="left", va="top",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
            ax.text(0.97, 0.95, label, transform=ax.transAxes, ha="right", va="top",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
            fig.colorbar(image, ax=ax, fraction=0.047, pad=0.02)
            if row == 1:
                ax.set_xlabel(r"$x/L$")
            if column == 0:
                ax.set_ylabel(r"$y/L$")
    axes[0, 0].text(0.03, 0.04, rf"$t/t_c={args.time:g}$",
                    transform=axes[0, 0].transAxes, ha="left", va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
