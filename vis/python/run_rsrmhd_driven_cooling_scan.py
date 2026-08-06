#!/usr/bin/env python3
"""Run the matched OU four-acceleration cooling-time production scan."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


T_EDDY = 15.0**0.5
DRIVE_SCALE = 0.5
TARGET_MACH = 0.5
ADIABATIC_INDEX = 4.0/3.0
INITIAL_DENSITY = 1.0
INITIAL_PRESSURE = 1.0/16.0
INITIAL_ADIABAT = INITIAL_PRESSURE

CASES = {
    "no_cooling": {"cooling_model": "none", "cooling_ratio": None},
    "tcool_0p1": {"cooling_model": "entropy", "cooling_ratio": 0.1},
    "tcool_1": {"cooling_model": "entropy", "cooling_ratio": 1.0},
    "tcool_10": {"cooling_model": "entropy", "cooling_ratio": 10.0},
}


def sha256(path):
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo, *arguments):
    """Return a short git query, or an empty string outside a repository."""
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=False,
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def rewrite_input(template, replacements):
    """Replace existing Athena input parameters by block and key."""
    block = None
    used = set()
    output = []
    for raw_line in template.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("<") and stripped.endswith(">"):
            block = stripped[1:-1]
        if block is not None and "=" in raw_line and not stripped.startswith("#"):
            key = raw_line.split("=", 1)[0].strip()
            location = (block, key)
            if location in replacements:
                raw_line = f"{key} = {replacements[location]}"
                used.add(location)
        output.append(raw_line)
    missing = set(replacements) - used
    if missing:
        rendered = ", ".join(f"<{block}>/{key}" for block, key in sorted(missing))
        raise RuntimeError(f"Template is missing replacement parameters: {rendered}")
    return "\n".join(output) + "\n"


def resolved_input(template, case_name, case, args):
    """Construct the exact input text for one scan member."""
    target_vrms = DRIVE_SCALE/T_EDDY
    viscosity = target_vrms*DRIVE_SCALE/args.reynolds
    basename = f"driven_cooling_{case_name}"
    cooling_ratio = case["cooling_ratio"]
    cooling_time = T_EDDY if cooling_ratio is None else cooling_ratio*T_EDDY
    replacements = {
        ("job", "basename"): basename,
        ("mesh", "nx1"): args.resolution,
        ("mesh", "nx2"): args.resolution,
        ("mesh", "nx3"): args.resolution,
        ("meshblock", "nx1"): args.meshblock,
        ("meshblock", "nx2"): args.meshblock,
        ("meshblock", "nx3"): args.meshblock,
        ("time", "tlim"): args.turnovers*T_EDDY,
        ("mhd", "resistivity"): viscosity,
        ("mhd", "shear_viscosity"): viscosity,
        ("mhd", "relativistic_cooling"): case["cooling_model"],
        ("mhd", "cooling_time"): cooling_time,
        ("mhd", "cooling_adiabat"): INITIAL_ADIABAT,
        ("problem", "pressure"): INITIAL_PRESSURE,
        ("problem", "profile_name"): basename,
        ("turb_driving", "tcorr"): T_EDDY,
        ("turb_driving", "accel_rms"): args.accel_rms,
        ("output1", "dt"): T_EDDY/args.samples_per_turnover,
    }
    return rewrite_input(template, replacements), basename, viscosity, cooling_time


def run_case(case_name, case, template, args):
    """Generate, run, and record one cooling-time calculation."""
    case_dir = args.output_root/case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    input_text, basename, viscosity, cooling_time = resolved_input(
        template, case_name, case, args,
    )
    input_path = case_dir/"input.athinput"
    input_path.write_text(input_text, encoding="utf-8")
    history_path = case_dir/f"{basename}.user.hst"
    record_path = case_dir/"run.json"
    if record_path.exists() and history_path.exists() and not args.force:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "completed":
            print(json.dumps({"case": case_name, "status": "skipped"}), flush=True)
            return

    command = ["mpirun", "-n", str(args.ranks), str(args.binary.resolve()),
               "-i", str(input_path)]
    started = time.time()
    record = {
        "group": "driven_cooling_scan",
        "case": case_name,
        "status": "running",
        "command": command,
        "mpi_ranks": args.ranks,
        "timeout_seconds": args.timeout,
        "resolution": args.resolution,
        "meshblock": args.meshblock,
        "turnovers": args.turnovers,
        "eddy_time": T_EDDY,
        "forcing_correlation_time": T_EDDY,
        "accel_rms": args.accel_rms,
        "target_mach_reference": TARGET_MACH,
        "drive_scale": DRIVE_SCALE,
        "nominal_reynolds": args.reynolds,
        "nominal_magnetic_reynolds": args.reynolds,
        "viscosity": viscosity,
        "resistivity": viscosity,
        "initial_beta": 1.0,
        "initial_magnetization": 0.1,
        "initial_density": INITIAL_DENSITY,
        "initial_pressure": INITIAL_PRESSURE,
        "adiabatic_index": ADIABATIC_INDEX,
        "integrator": "imex3",
        "cfl_number": 0.4,
        "electric_field_layout": "cell_centered",
        "reconstruction": "wenoz",
        "fofc": True,
        "cooling_model": case["cooling_model"],
        "cooling_time_over_eddy_time": case["cooling_ratio"],
        "cooling_time": cooling_time,
        "binary": str(args.binary.resolve()),
        "binary_sha256": sha256(args.binary),
        "source_input": str(args.input.resolve()),
        "source_input_sha256": sha256(args.input),
        "git_branch": git_value(args.repo, "branch", "--show-current"),
        "git_commit": git_value(args.repo, "rev-parse", "HEAD"),
        "started_unix": started,
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log_path = case_dir/"run.log"
    return_code = None
    status = "failed"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=args.timeout)
            if return_code == 0 and history_path.exists():
                status = "completed"
        except subprocess.TimeoutExpired:
            status = "timed_out"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return_code = process.returncode
    finished = time.time()
    record.update({
        "status": status,
        "return_code": return_code,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "history": str(history_path) if history_path.exists() else None,
    })
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "case": case_name, "status": status,
        "elapsed_seconds": finished - started,
    }), flush=True)
    if status != "completed":
        raise RuntimeError(f"{case_name} ended with status {status}; see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo", type=Path, default=repo_default)
    parser.add_argument(
        "--input", type=Path,
        default=(repo_default/"inputs/paper_tests"
                 /"rsrmhd_driven_cooling_scan_3d64.athinput"),
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3590.0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--meshblock", type=int, default=32)
    parser.add_argument("--turnovers", type=float, default=10.0)
    parser.add_argument("--reynolds", type=float, default=100.0)
    parser.add_argument("--accel-rms", type=float, default=0.09)
    parser.add_argument("--samples-per-turnover", type=float, default=20.0)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.resolution % args.meshblock != 0:
        parser.error("resolution must be divisible by meshblock")
    if args.reynolds <= 0.0 or args.accel_rms <= 0.0:
        parser.error("reynolds and accel-rms must be positive")
    if not args.binary.is_file():
        parser.error(f"binary does not exist: {args.binary}")
    template = args.input.read_text(encoding="utf-8")
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input, args.output_root/"source_input.athinput")
    for case_name in args.cases:
        run_case(case_name, CASES[case_name], template, args)


if __name__ == "__main__":
    main()
