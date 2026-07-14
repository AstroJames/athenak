#!/usr/bin/env python3
"""Plot the Pm scan for visco-resistive decaying turbulence."""

import argparse
from pathlib import Path

import matplotlib as mpl

import plot_rsrmhd_decaying_turbulence as base


CASES = (
    ("pm0p1", r"$Pm=0.1$"),
    ("pm1", r"$Pm=1$"),
    ("pm10", r"$Pm=10$"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm0p1", type=Path, required=True)
    parser.add_argument("--pm1", type=Path, required=True)
    parser.add_argument("--pm10", type=Path, required=True)
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

    directories = {
        "pm0p1": args.pm0p1,
        "pm1": args.pm1,
        "pm10": args.pm10,
    }
    histories = {}
    profiles = {}
    for key, _ in CASES:
        directory = directories[key]
        stem = "viscoresistive" if key == "pm1" else key
        histories[key] = base.read_history(directory / f"{stem}.user.hst")
        profiles[key] = base.read_profile(directory, stem)

    eddy_time = 1.0/(args.peak_mode*args.velocity_rms)
    final_crossing_time = histories["pm1"]["time"][-1]/eddy_time
    field_output = args.output_dir / "decaying-turbulence-pm-fields.png"
    history_output = args.output_dir / "decaying-turbulence-pm-histories.png"
    base.plot_fields(
        profiles, field_output, final_crossing_time, cases=CASES,
    )
    base.plot_histories(histories, history_output, eddy_time, cases=CASES)
    if args.html is not None:
        base.write_fragment(args.html, field_output, history_output)


if __name__ == "__main__":
    main()
