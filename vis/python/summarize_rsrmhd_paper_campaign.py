#!/usr/bin/env python3
"""Audit a completed visco-resistive SRMHD paper campaign."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def is_production(record):
    """Return whether a run record belongs to the final production matrix."""
    group = record["group"]
    case = record["case"]
    if group == "charged_vortex":
        return case in {
            f"{layout}_{resolution}"
            for layout in ("cell", "face")
            for resolution in (32, 64, 128, 256, 512)
        }
    if group == "decomposition":
        return case.startswith("dual_")
    return True


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-cap", type=float, default=3600.0)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_dir = root/"00_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    production = []
    preliminary = []
    for path in sorted(root.rglob("run.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["record_path"] = str(path.relative_to(root))
        (production if is_production(record) else preliminary).append(record)

    columns = (
        "group", "case", "status", "mpi_ranks", "elapsed_seconds",
        "timeout_seconds", "return_code", "started_utc", "finished_utc",
        "source_input", "record_path", "command",
    )
    with (manifest_dir/"campaign_manifest.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in production:
            row = {column: record.get(column, "") for column in columns}
            row["command"] = " ".join(record.get("command", ()))
            writer.writerow(row)

    figures = sorted((root/"figures").glob("*.pdf"))
    with (manifest_dir/"figure_checksums.sha256").open(
            "w", encoding="utf-8") as stream:
        for path in figures:
            stream.write(f"{sha256(path)}  {path.relative_to(root)}\n")

    statuses = {}
    for record in production:
        status = record.get("status", "missing")
        statuses[status] = statuses.get(status, 0) + 1
    elapsed = [
        float(record["elapsed_seconds"])
        for record in production if "elapsed_seconds" in record
    ]
    summary = {
        "production_runs": len(production),
        "production_statuses": statuses,
        "preliminary_runs_excluded": len(preliminary),
        "preliminary_record_paths": [
            record["record_path"] for record in preliminary
        ],
        "runtime_cap_seconds": args.runtime_cap,
        "maximum_elapsed_seconds": max(elapsed, default=None),
        "figure_pdfs": len(figures),
        "figure_files": [str(path.relative_to(root)) for path in figures],
    }
    (manifest_dir/"campaign_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    scripts = root/"scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, scripts/Path(__file__).name)

    incomplete = [
        record for record in production if record.get("status") != "completed"
    ]
    over_cap = [
        record for record in production
        if float(record.get("elapsed_seconds", 0.0)) >= args.runtime_cap
    ]
    if incomplete or over_cap:
        raise SystemExit(
            f"campaign audit failed: {len(incomplete)} incomplete, "
            f"{len(over_cap)} at or above the runtime cap"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
