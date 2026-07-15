#!/usr/bin/env python3
"""Plot a final 2D slice from an antenna-driven SRRMHD binary output."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import bin_convert


def merge_uniform(data):
    shape = (data['Nx3'], data['Nx2'], data['Nx1'])
    block_shape = (
        data['nx3_out_mb'], data['nx2_out_mb'], data['nx1_out_mb']
    )
    merged = {
        name: np.empty(shape, dtype=float) for name in data['mb_data']
    }
    for block, logical in enumerate(data['mb_logical']):
        block_x, block_y, block_z, level = logical
        if level != 0:
            raise ValueError('Slice plot requires a uniform mesh')
        xs = slice(block_x*block_shape[2], (block_x + 1)*block_shape[2])
        ys = slice(block_y*block_shape[1], (block_y + 1)*block_shape[1])
        zs = slice(block_z*block_shape[0], (block_z + 1)*block_shape[0])
        for name, values in data['mb_data'].items():
            merged[name][zs, ys, xs] = values[block]
    return merged


def derivative(values, axis, spacing):
    return (
        np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
    )/(2.0*spacing)


def robust_limits(values, signed=False):
    finite = values[np.isfinite(values)]
    if signed:
        limit = np.percentile(np.abs(finite), 99.0)
        return -limit, limit
    return tuple(np.percentile(finite, (1.0, 99.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('primitive', type=Path)
    parser.add_argument('antenna', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--alfven-speed', type=float, default=0.576869744537761)
    parser.add_argument('--gamma', type=float, default=4.0/3.0)
    parser.add_argument('--slice-index', type=int)
    args = parser.parse_args()

    primitive_data = bin_convert.read_binary(str(args.primitive))
    antenna_data = bin_convert.read_binary(str(args.antenna))
    if not np.isclose(primitive_data['time'], antenna_data['time']):
        raise ValueError('Primitive and antenna snapshots have different times')
    primitive = merge_uniform(primitive_data)
    antenna = merge_uniform(antenna_data)

    nx = primitive_data['Nx1']
    ny = primitive_data['Nx2']
    nz = primitive_data['Nx3']
    lx = primitive_data['x1max'] - primitive_data['x1min']
    ly = primitive_data['x2max'] - primitive_data['x2min']
    lz = primitive_data['x3max'] - primitive_data['x3min']
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    kslice = nz//2 if args.slice_index is None else args.slice_index
    if not 0 <= kslice < nz:
        raise ValueError('slice index lies outside the mesh')

    rho = primitive['dens']
    internal = primitive['eint']
    pressure = (args.gamma - 1.0)*internal
    u1, u2, u3 = primitive['velx'], primitive['vely'], primitive['velz']
    lorentz = np.sqrt(1.0 + u1*u1 + u2*u2 + u3*u3)
    v1, v2, v3 = u1/lorentz, u2/lorentz, u3/lorentz
    b1, b2, b3 = primitive['bcc1'], primitive['bcc2'], primitive['bcc3']
    b0 = abs(np.mean(b3))

    current3 = derivative(b2, 2, dx) - derivative(b1, 1, dy)
    vorticity3 = derivative(v2, 2, dx) - derivative(v1, 1, dy)
    speed = np.sqrt(v1*v1 + v2*v2 + v3*v3)
    bdotv = b1*v1 + b2*v2 + b3*v3
    bcom2 = (b1*b1 + b2*b2 + b3*b3)/(lorentz*lorentz) + bdotv*bdotv
    enthalpy = rho + args.gamma*internal
    sigma = bcom2/enthalpy
    jant = np.sqrt(
        antenna['jant1']**2 + antenna['jant2']**2 + antenna['jant3']**2
    )

    fields = (
        (rho/np.mean(rho), r'$\rho/\langle\rho\rangle$', 'viridis', False),
        (pressure/np.mean(pressure), r'$P/\langle P\rangle$', 'magma', False),
        (speed/args.alfven_speed, r'$|\mathbf{v}|/v_{\rm A0}$',
         'cividis', False),
        (sigma, r'$b_{\rm ideal}^2/w$', 'plasma', False),
        ((b3 - b0)/b0, r'$(B^z-B_0)/B_0$', 'RdBu_r', True),
        (current3*lx/b0,
         r'$(\boldsymbol{\nabla}\!\times\!\mathbf{B})^zL/B_0$',
         'RdBu_r', True),
        (vorticity3*lx/args.alfven_speed,
         r'$\omega^zL/v_{\rm A0}$', 'RdBu_r', True),
        (jant*lx/b0, r'$|\mathbf{J}_{\rm ant}|L/B_0$', 'inferno', False),
    )

    mpl.rc_file('/Users/beattijr/.matplotlib/matplotlibrc')
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8),
                             layout='constrained')
    extent = (
        primitive_data['x1min']/lx, primitive_data['x1max']/lx,
        primitive_data['x2min']/ly, primitive_data['x2max']/ly,
    )
    for index, (axis, (values, label, cmap, signed)) in enumerate(
            zip(axes.flat, fields)):
        plane = values[kslice]
        vmin, vmax = robust_limits(plane, signed=signed)
        image = axis.imshow(
            plane, origin='lower', interpolation='nearest', extent=extent,
            cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal'
        )
        axis.set_title(label)
        if index >= 4:
            axis.set_xlabel(r'$x/L$')
        else:
            axis.set_xticklabels([])
        if index % 4 == 0:
            axis.set_ylabel(r'$y/L$')
        else:
            axis.set_yticklabels([])
        colorbar = fig.colorbar(image, ax=axis, orientation='horizontal',
                               pad=0.02, fraction=0.05)
        colorbar.ax.tick_params(labelsize=7)

    z_position = primitive_data['x3min'] + (kslice + 0.5)*dz
    alfven_time = lx/args.alfven_speed
    fig.suptitle(
        rf'$t/t_{{\rm A0}}={primitive_data["time"]/alfven_time:.2f}$, '
        rf'$z/L={z_position/lz:.3f}$; native ${nx}^3$ cells'
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
