#!/usr/bin/env python3
"""Run the complete visco-resistive SRMHD paper campaign with provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


GROUP_DIRS = {
    "current_sheet": "01_current_sheet",
    "ohmic_harris": "02_ohmic_harris",
    "charged_vortex": "03_charged_vortex",
    "decomposition": "04_cyclic_and_decomposition",
    "telegraph": "05_viscous_telegraph",
    "phaseb": "06_viscous_phaseb",
    "shear_layer": "07_viscous_shear_layer",
    "boost_rotation": "08_boosted_and_rotated_shear",
    "khi": "09_viscous_khi",
    "decaying": "10_decaying_turbulence",
    "driven": "11_driven_turbulence",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_value(repo, *arguments):
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


class Campaign:
    def __init__(self, root, repo, athena, timeout, force):
        self.root = root
        self.repo = repo
        self.athena = athena
        self.timeout = timeout
        self.force = force
        self.commit = git_value(repo, "rev-parse", "HEAD")
        self.branch = git_value(repo, "branch", "--show-current")
        self.write_source_manifest()

    def write_source_manifest(self):
        manifest_dir = self.root/"00_manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        status = subprocess.run(
            ["git", "status", "--short"], cwd=self.repo, check=True,
            text=True, stdout=subprocess.PIPE,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary"], cwd=self.repo, check=True,
            stdout=subprocess.PIPE,
        ).stdout
        (manifest_dir/"git-status.txt").write_text(status, encoding="utf-8")
        (manifest_dir/"working-tree.patch").write_bytes(diff)
        digest = hashlib.sha256(self.athena.read_bytes()).hexdigest()
        source = {
            "created_utc": utc_now(),
            "repository": str(self.repo),
            "git_branch": self.branch,
            "git_commit": self.commit,
            "athena_executable": str(self.athena),
            "athena_sha256": digest,
        }
        (manifest_dir/"source.json").write_text(
            json.dumps(source, indent=2) + "\n", encoding="utf-8",
        )

    def run_case(self, group, name, input_path, overrides=(), ranks=1):
        case_dir = self.root/GROUP_DIRS[group]/name
        case_dir.mkdir(parents=True, exist_ok=True)
        record_path = case_dir/"run.json"
        if record_path.exists() and not self.force:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") == "completed":
                print(f"SKIP {group}/{name}", flush=True)
                return

        copied_input = case_dir/"input.athinput"
        shutil.copy2(input_path, copied_input)
        command = [str(self.athena), "-i", str(copied_input), *overrides]
        if ranks > 1:
            command = ["mpirun", "-n", str(ranks), *command]
        (case_dir/"command.txt").write_text(
            " ".join(command) + "\n", encoding="utf-8",
        )
        record = {
            "group": group,
            "case": name,
            "source_input": str(input_path),
            "command": command,
            "mpi_ranks": ranks,
            "timeout_seconds": self.timeout,
            "git_branch": self.branch,
            "git_commit": self.commit,
            "started_utc": utc_now(),
            "status": "running",
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n",
                               encoding="utf-8")
        start = time.monotonic()
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        print(f"RUN  {group}/{name} ({ranks} rank{'s' if ranks != 1 else ''})",
              flush=True)
        with (case_dir/"stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=case_dir, env=environment, text=True,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=self.timeout)
                status = "completed" if return_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return_code = 124
                status = "timed_out"
        record.update({
            "finished_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start,
            "return_code": return_code,
            "status": status,
        })
        record_path.write_text(json.dumps(record, indent=2) + "\n",
                               encoding="utf-8")
        if status != "completed":
            raise RuntimeError(f"{group}/{name} ended with status {status}")

    def current_sheet(self):
        input_path = self.repo/"inputs/tests/rsrmhd_current_sheet.athinput"
        for eta in (0.001, 0.003, 0.01, 0.03):
            tag = f"eta{eta:g}".replace(".", "p")
            for layout, electric_ct in (("cell", "false"), ("face", "true")):
                name = f"{layout}_{tag}"
                block_size = 1024 if layout == "face" else 256
                self.run_case("current_sheet", name, input_path, (
                    f"job/basename=sheet_{name}", "mesh/nx1=1024",
                    f"meshblock/nx1={block_size}",
                    f"mhd/electric_ct={electric_ct}",
                    f"mhd/resistivity={eta:.17g}",
                    f"problem/initial_eta={eta:.17g}",
                    "output1/dt=9.0", "output3/dt=9.0",
                ))

    def ohmic_harris(self):
        input_path = self.repo/"inputs/paper_tests/rsrmhd_ohmic_decay.athinput"
        for eta in (0.001, 0.003, 0.01, 0.03):
            tag = f"eta{eta:g}".replace(".", "p")
            self.run_case("ohmic_harris", tag, input_path, (
                f"job/basename=ohmic_{tag}",
                f"mhd/resistivity={eta:.17g}",
            ))

    def charged_vortex(self):
        input_path = (
            self.repo/"inputs/paper_tests/rsrmhd_charged_vortex_512.athinput"
        )
        for resolution in (32, 64, 128, 256, 512):
            block = min(resolution, 64)
            blocks = (resolution // block)**2
            ranks = min(8, blocks)
            for layout, electric_ct in (("cell", "false"), ("face", "true")):
                case_name = f"{layout}_{resolution}"
                self.run_case("charged_vortex", case_name, input_path, (
                    f"job/basename=vortex_{layout}",
                    f"mhd/electric_ct={electric_ct}",
                    f"mesh/nx1={resolution}", f"mesh/nx2={resolution}",
                    f"meshblock/nx1={block}", f"meshblock/nx2={block}",
                ), ranks=ranks)

    def decomposition(self):
        input_path = self.repo/"inputs/tests/rsrmhd_charged_vortex_3d.athinput"
        for plane in ("xy", "yz", "zx"):
            cyclic_name = f"dual_cyclic32_{plane}"
            self.run_case("decomposition", cyclic_name, input_path, (
                f"job/basename={cyclic_name}", f"problem/plane={plane}",
                "mhd/electric_ct=true",
                "mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=32",
                "meshblock/nx1=32", "meshblock/nx2=32", "meshblock/nx3=32",
            ))
            for layout, block, ranks in (
                    ("one_block", 16, 1), ("eight_blocks", 8, 1),
                    ("eight_blocks_mpi2", 8, 2)):
                name = f"dual_{layout}_{plane}"
                self.run_case("decomposition", name, input_path, (
                    f"job/basename={name}", f"problem/plane={plane}",
                    "mhd/electric_ct=true",
                    "mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=16",
                    f"meshblock/nx1={block}", f"meshblock/nx2={block}",
                    f"meshblock/nx3={block}",
                ), ranks=ranks)

    def telegraph(self):
        input_path = self.repo/"inputs/tests/rsrmhd_viscous_telegraph.athinput"
        scans = (
            ("fixed_tau", ((0.01, 0.2), (0.03, 0.2), (0.05, 0.2))),
            ("fixed_nu", ((0.02, 0.1), (0.02, 0.2), (0.02, 0.4))),
        )
        for scan, cases in scans:
            for nu, tau in cases:
                parameter = f"nu{nu:g}_tau{tau:g}".replace(".", "p")
                for sample in range(1, 11):
                    current_time = sample/10.0
                    name = f"{scan}/{parameter}/t{sample:02d}"
                    self.run_case("telegraph", name, input_path, (
                        "mesh/nx1=128", "meshblock/nx1=64",
                        f"time/tlim={current_time:.17g}",
                        f"mhd/shear_viscosity={nu:.17g}",
                        f"mhd/shear_relaxation_time={tau:.17g}",
                    ))

    def phaseb(self):
        telegraph = self.repo/"inputs/tests/rsrmhd_viscous_telegraph.athinput"
        longitudinal = self.repo/"inputs/tests/rsrmhd_viscous_longitudinal.athinput"
        for sample in range(1, 11):
            current_time = sample/5.0
            self.run_case("phaseb", f"diffusion/t{sample:02d}", telegraph, (
                f"time/tlim={current_time:.17g}",
                "mhd/shear_viscosity=0.0025",
                "mhd/shear_relaxation_time=0.01", "mesh/nx1=128",
                "meshblock/nx1=64",
            ))
        for resolution in (64, 128, 256, 512):
            self.run_case("phaseb", f"timestep/n{resolution}", telegraph, (
                "time/tlim=0.05", "mhd/shear_viscosity=0.0025",
                "mhd/shear_relaxation_time=0.01",
                f"mesh/nx1={resolution}",
                f"meshblock/nx1={resolution // 2}",
            ))
        for sample in range(1, 11):
            current_time = sample/10.0
            self.run_case("phaseb", f"longitudinal/t{sample:02d}",
                          longitudinal, (
                              f"time/tlim={current_time:.17g}",
                              "mesh/nx1=128", "meshblock/nx1=64",
                          ))

    def shear_layer(self):
        for tag in ("005", "010", "020", "040", "050"):
            input_path = (
                self.repo/"inputs/paper_tests"/
                f"rsrmhd_viscous_shear_nu{tag}.athinput"
            )
            self.run_case("shear_layer", f"nu{tag}", input_path, (
                "mesh/nx1=1024", "meshblock/nx1=256",
            ))

    def boost_rotation(self):
        boosted = self.repo/"inputs/tests/rsrmhd_viscous_boosted.athinput"
        self.run_case("boost_rotation", "boosted_x1", boosted)
        multid = self.repo/"inputs/tests/rsrmhd_viscous_multid.athinput"
        self.run_case("boost_rotation", "rotated_x2", multid, (
            "job/basename=rotated_x2",
            "problem/viscous_diagnostic_name=rotated_x2",
        ))
        self.run_case("boost_rotation", "rotated_x3", multid, (
            "job/basename=rotated_x3", "mesh/nx1=4", "mesh/nx2=4",
            "mesh/nx3=256", "meshblock/nx1=4", "meshblock/nx2=4",
            "meshblock/nx3=128", "problem/wave_direction=3",
            "problem/viscous_diagnostic_name=rotated_x3",
        ))

    def khi(self):
        input_path = self.repo/"inputs/tests/rsrmhd_viscous_kh.athinput"
        for name, viscosity in (("inviscid_512", 0.0), ("viscous_512", 0.0001)):
            stem = name.removesuffix("_512")
            self.run_case("khi", name, input_path, (
                f"job/basename=kh_{stem}", f"mhd/shear_viscosity={viscosity}",
                "mesh/nx1=512", "mesh/nx2=512", "meshblock/nx1=64",
                "meshblock/nx2=64", "time/tlim=4.0",
                f"problem/viscous_diagnostic_name=kh_{stem}",
                f"problem/viscous_profile_name=kh_{stem}",
            ), ranks=8)

    def decaying(self):
        for name in ("ideal", "pm1", "pm10", "pm50"):
            input_path = (
                self.repo/"inputs/paper_tests"/
                f"rsrmhd_decaying_turbulence_{name}.athinput"
            )
            self.run_case("decaying", name, input_path, ranks=8)

    def driven(self):
        cases = (
            ("n32", "rsrmhd_driven_turbulence_3d_mach0p5_re50.athinput"),
            ("n64", "rsrmhd_driven_turbulence_3d64_mach0p5_re50.athinput"),
        )
        for name, filename in cases:
            self.run_case(
                "driven", name, self.repo/"inputs/paper_tests"/filename,
                ranks=8,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=(*GROUP_DIRS, "all"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3590.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    campaign = Campaign(
        args.root.resolve(), args.repo.resolve(), args.athena.resolve(),
        args.timeout, args.force,
    )
    groups = GROUP_DIRS if args.group == "all" else (args.group,)
    for group in groups:
        getattr(campaign, group)()


if __name__ == "__main__":
    main()
