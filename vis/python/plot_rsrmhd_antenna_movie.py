#!/usr/bin/env python3
"""Render fixed-scale mid-plane frames and a movie of antenna-driven SRRMHD."""

import argparse
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import bin_convert
from plot_rsrmhd_antenna_slices import derivative, merge_uniform


def _snapshot_number(path):
    try:
        return int(path.name.split('.')[-2])
    except (IndexError, ValueError) as error:
        raise ValueError(f'cannot parse snapshot number from {path}') from error


def _snapshot_map(directory, output_id):
    snapshots = {}
    for path in directory.glob(f'*.{output_id}.*.bin'):
        number = _snapshot_number(path)
        if number in snapshots:
            raise ValueError(f'duplicate {output_id} snapshot number {number}')
        snapshots[number] = path
    return snapshots


def pair_snapshots(directory):
    """Return primitive/current snapshots paired and ordered by output number."""
    directory = Path(directory)
    primitive = _snapshot_map(directory, 'prim')
    antenna = _snapshot_map(directory, 'antenna')
    if not primitive or primitive.keys() != antenna.keys():
        raise ValueError('primitive and antenna snapshots require matching numbers')
    return [(primitive[number], antenna[number]) for number in sorted(primitive)]


def fixed_limits(arrays, signed, percentile=99.0, include_zero=True):
    """Return one robust color range for a sequence of two-dimensional fields."""
    finite = np.concatenate([
        np.asarray(array)[np.isfinite(array)].ravel() for array in arrays
    ])
    if finite.size == 0:
        raise ValueError('cannot determine color limits without finite values')
    if signed:
        limit = float(np.percentile(np.abs(finite), percentile))
        if limit <= 0.0:
            limit = 1.0
        return -limit, limit
    upper = float(np.percentile(finite, percentile))
    lower = 0.0 if include_zero else float(
        np.percentile(finite, 100.0 - percentile)
    )
    if upper <= lower:
        upper = lower + max(abs(lower), 1.0)*1.0e-12
    return lower, upper


def ffmpeg_command(frame_pattern, movie_path, fps):
    """Return an H.264 command with dimensions compatible with YUV420."""
    return [
        'ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps),
        '-i', str(frame_pattern), '-vf',
        'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(movie_path),
    ]


def _load_frame(primitive_path, antenna_path, alfven_speed, gamma, slice_index):
    primitive_data = bin_convert.read_binary(str(primitive_path))
    antenna_data = bin_convert.read_binary(str(antenna_path))
    if not np.isclose(primitive_data['time'], antenna_data['time']):
        raise ValueError('primitive and antenna snapshots have different times')
    primitive = merge_uniform(primitive_data)
    antenna = merge_uniform(antenna_data)

    nx = primitive_data['Nx1']
    ny = primitive_data['Nx2']
    nz = primitive_data['Nx3']
    lx = primitive_data['x1max'] - primitive_data['x1min']
    ly = primitive_data['x2max'] - primitive_data['x2min']
    lz = primitive_data['x3max'] - primitive_data['x3min']
    dx = lx/nx
    dy = ly/ny
    kslice = nz//2 if slice_index is None else slice_index
    if not 0 <= kslice < nz:
        raise ValueError('slice index lies outside the mesh')

    rho = primitive['dens']
    internal = primitive['eint']
    u1, u2, u3 = primitive['velx'], primitive['vely'], primitive['velz']
    lorentz = np.sqrt(1.0 + u1*u1 + u2*u2 + u3*u3)
    v1, v2, v3 = u1/lorentz, u2/lorentz, u3/lorentz
    b1, b2, b3 = primitive['bcc1'], primitive['bcc2'], primitive['bcc3']
    b0 = abs(np.mean(b3))
    speed = np.sqrt(v1*v1 + v2*v2 + v3*v3)
    vorticity3 = derivative(v2, 2, dx) - derivative(v1, 1, dy)
    bdotv = b1*v1 + b2*v2 + b3*v3
    bcom2 = (b1*b1 + b2*b2 + b3*b3)/(lorentz*lorentz) + bdotv*bdotv
    sigma = bcom2/(rho + gamma*internal)
    j1 = antenna['jant1']*lx/b0
    j2 = antenna['jant2']*lx/b0
    j3 = antenna['jant3']*lx/b0
    jmag = np.sqrt(j1*j1 + j2*j2 + j3*j3)

    fields = (
        (j1[kslice], r'$J_{\rm ant}^xL/B_0$', 'RdBu_r', True,
         'current_components'),
        (j2[kslice], r'$J_{\rm ant}^yL/B_0$', 'RdBu_r', True,
         'current_components'),
        (j3[kslice], r'$J_{\rm ant}^zL/B_0$', 'RdBu_r', True,
         'current_components'),
        (jmag[kslice], r'$|\mathbf{J}_{\rm ant}|L/B_0$', 'inferno', False,
         'current_magnitude'),
        (((b3 - b0)/b0)[kslice], r'$(B^z-B_0)/B_0$', 'RdBu_r', True,
         'magnetic_response'),
        ((speed/alfven_speed)[kslice], r'$|\mathbf{v}|/v_{\rm A0}$',
         'cividis', False, 'speed'),
        ((vorticity3*lx/alfven_speed)[kslice],
         r'$\omega^zL/v_{\rm A0}$', 'RdBu_r', True, 'vorticity'),
        (sigma[kslice], r'$b_{\rm ideal}^2/w$', 'plasma', False,
         'magnetization'),
    )
    extent = (
        primitive_data['x1min']/lx, primitive_data['x1max']/lx,
        primitive_data['x2min']/ly, primitive_data['x2max']/ly,
    )
    return {
        'fields': fields,
        'time': primitive_data['time'],
        'extent': extent,
        'slice_position': (
            primitive_data['x3min'] + (kslice + 0.5)*lz/nz
        )/lz,
        'resolution': nx,
        'alfven_time': lx/alfven_speed,
    }


def _color_limits(frames, percentile):
    groups = {}
    properties = {}
    for frame in frames:
        for values, _, _, signed, group in frame['fields']:
            groups.setdefault(group, []).append(values)
            properties[group] = signed
    return {
        group: fixed_limits(values, properties[group], percentile=percentile)
        for group, values in groups.items()
    }


def _render_frame(frame, limits, output, dpi):
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8),
                             layout='constrained')
    for index, (axis, field) in enumerate(zip(axes.flat, frame['fields'])):
        values, label, cmap, _, group = field
        image = axis.imshow(
            values, origin='lower', interpolation='nearest',
            extent=frame['extent'], cmap=cmap,
            vmin=limits[group][0], vmax=limits[group][1], aspect='equal'
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
        colorbar = fig.colorbar(
            image, ax=axis, orientation='horizontal', pad=0.02, fraction=0.05
        )
        colorbar.ax.tick_params(labelsize=7)
    time_over_alfven = frame['time']/frame['alfven_time']
    fig.suptitle(
        rf'$t/t_{{\rm A0}}={time_over_alfven:.2f}$, '
        rf'$z/L={frame["slice_position"]:.3f}$; '
        rf'native ${frame["resolution"]}^3$ cells'
    )
    fig.savefig(output, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--alfven-speed', type=float, default=0.576869744537761)
    parser.add_argument('--gamma', type=float, default=4.0/3.0)
    parser.add_argument('--slice-index', type=int)
    parser.add_argument('--percentile', type=float, default=99.0)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    if not 50.0 <= args.percentile <= 100.0:
        parser.error('--percentile must lie between 50 and 100')
    if args.fps <= 0 or args.dpi <= 0:
        parser.error('--fps and --dpi must be positive')

    pairs = pair_snapshots(args.input_directory)
    frames = [
        _load_frame(primitive, antenna, args.alfven_speed, args.gamma,
                    args.slice_index)
        for primitive, antenna in pairs
    ]
    limits = _color_limits(frames, args.percentile)

    mpl.rc_file('/Users/beattijr/.matplotlib/matplotlibrc')
    frame_directory = args.output_directory/'frames'
    frame_directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame_path = frame_directory/f'frame_{index:05d}.png'
        _render_frame(frame, limits, frame_path, args.dpi)
        if index == len(frames) - 1:
            _render_frame(
                frame, limits, args.output_directory/'antenna_final.png',
                args.dpi
            )
            _render_frame(
                frame, limits, args.output_directory/'antenna_final.pdf',
                args.dpi
            )

    movie_path = args.output_directory/'antenna_driving.mp4'
    subprocess.run(ffmpeg_command(
        frame_directory/'frame_%05d.png', movie_path, args.fps
    ), check=True)
    print(f'Rendered {len(frames)} frames to {movie_path}')


if __name__ == '__main__':
    main()
