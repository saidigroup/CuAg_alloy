#!/usr/bin/env python3
"""
Collect Cu-Ag Supporting Information diagnostics from TPC and viscosity runs.

The collector is intentionally conservative:

TPC
---
A directory is treated as a TPC run when it contains nph_timeseries.dat,
nph_solid_profile.dat, or Melt_nph.* files. The collector copies the numerical
diagnostics, run provenance, and only the first and last Melt_nph snapshots by
default. Use --all-tpc-snapshots to copy every Melt_nph.* frame.

Viscosity
---------
The collector searches recursively for Green-Kubo/viscosity postprocessing
outputs using filename keywords. It copies stress-autocorrelation data,
cumulative/running integrals, plateau/summary files, diagnostic plots, and
nearby provenance files. It does not copy generic trajectory dumps.

The output contains:
    collection_manifest.csv
    tpc_run_manifest.csv
    viscosity_directory_manifest.csv
    missing_or_skipped.csv
    README_collection.txt

Example
-------
python3 collect_cuag_si_diagnostics.py \
  --tpc-root PBE=/path/to/PBE/TM \
  --tpc-root PBEsol=/path/to/PBEsol/TM \
  --viscosity-root PBE=/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au/Validation/Viscosity \
  --viscosity-root PBEsol=/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au_sol/Validation/Viscosity \
  --outdir CuAg_SI_diagnostics \
  --archive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


TPC_CORE_FILES = (
    "nph_timeseries.dat",
    "nph_solid_profile.dat",
    "nph_msd_liquid.dat",
    "composition.txt",
    "element_counts.dat",
    "nph_report.md",
    "melt.in",
    "log.lammps",
    "out_melt",
)

TPC_OPTIONAL_PLOTS = (
    "thermo_block_means.png",
    "f_solid_vs_time.png",
    "msd_liquid_fit.png",
)

TPC_STATE_FILES = (
    "data.lmp",
    "two_phase",
    "cool1",
)

VISCOSITY_KEYWORDS = (
    "viscos",
    "green_kubo",
    "green-kubo",
    "greenkubo",
    "sacf",
    "stress_acf",
    "stress-acf",
    "stress_autocorr",
    "stress-autocorr",
    "autocorrelation",
    "autocorr",
    "cumulative",
    "running_integral",
    "running-integral",
    "plateau",
)

VISCOSITY_ALLOWED_SUFFIXES = {
    ".csv", ".dat", ".txt", ".json", ".npz", ".npy",
    ".png", ".pdf", ".md", ".yaml", ".yml",
}

VISCOSITY_PROVENANCE_NAMES = (
    "log.lammps",
    "thermo.in",
    "viscosity.in",
    "in.viscosity",
    "in.thermo",
    "submit",
    "submit.sh",
    "submit_viscosity",
    "potential_path.txt",
)


@dataclass(frozen=True)
class RootSpec:
    label: str
    path: Path


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unlabeled"


def parse_root_spec(raw: str) -> RootSpec:
    if "=" in raw:
        label, path_text = raw.split("=", 1)
    else:
        path_text = raw
        label = Path(path_text).name or "root"
    path = Path(path_text).expanduser().resolve()
    return RootSpec(sanitize_label(label), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Cu-Ag TPC and viscosity diagnostics for the SI."
    )
    parser.add_argument(
        "--tpc-root",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="TPC search root. May be supplied more than once.",
    )
    parser.add_argument(
        "--viscosity-root",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Viscosity search root. May be supplied more than once.",
    )
    parser.add_argument(
        "--outdir",
        default="CuAg_SI_diagnostics",
        help="Destination directory.",
    )
    parser.add_argument(
        "--all-tpc-snapshots",
        action="store_true",
        help="Copy every Melt_nph.* snapshot instead of only the first and last.",
    )
    parser.add_argument(
        "--include-tpc-state-files",
        action="store_true",
        help="Also copy data.lmp, two_phase, and cool1 from each TPC run.",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=250.0,
        help="Skip individual files larger than this limit (default: 250 MB).",
    )
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Calculate SHA-256 hashes for copied files.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create OUTDIR.tar.gz after collection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and write manifests without copying files.",
    )
    args = parser.parse_args()

    if not args.tpc_root and not args.viscosity_root:
        parser.error("Provide at least one --tpc-root or --viscosity-root.")
    if args.max_file_mb <= 0:
        parser.error("--max-file-mb must be positive.")
    return args


def sha256sum(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def relative_or_name(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def snapshot_step(path: Path) -> Optional[int]:
    match = re.fullmatch(r"Melt_nph\.(\d+)", path.name)
    return int(match.group(1)) if match else None


def select_tpc_snapshots(run_dir: Path, all_snapshots: bool) -> list[Path]:
    snapshots = []
    for path in run_dir.glob("Melt_nph.*"):
        if path.is_file() and snapshot_step(path) is not None:
            snapshots.append(path)
    snapshots.sort(key=lambda p: snapshot_step(p) or 0)

    if all_snapshots or len(snapshots) <= 2:
        return snapshots
    return [snapshots[0], snapshots[-1]]


def read_text_safely(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(errors="replace")
    except OSError:
        return ""


def extract_melt_metadata(run_dir: Path) -> dict[str, str]:
    result = {
        "composition_text": "",
        "deepmd_model": "",
        "target_Tm_expression": "",
        "target_pressure_expression": "",
    }

    composition = run_dir / "composition.txt"
    if composition.is_file():
        result["composition_text"] = " ".join(
            line.strip() for line in read_text_safely(composition).splitlines()
            if line.strip()
        )

    melt_file = run_dir / "melt.in"
    text = read_text_safely(melt_file)
    if text:
        model_match = re.search(
            r"^\s*pair_style\s+deepmd\s+(\S+)",
            text,
            flags=re.MULTILINE,
        )
        tm_match = re.search(
            r"^\s*variable\s+Tm\s+equal\s+([^\s#]+)",
            text,
            flags=re.MULTILINE,
        )
        press_match = re.search(
            r"^\s*variable\s+Press\s+equal\s+([^\s#]+)",
            text,
            flags=re.MULTILINE,
        )
        if model_match:
            result["deepmd_model"] = model_match.group(1)
        if tm_match:
            result["target_Tm_expression"] = tm_match.group(1)
        if press_match:
            result["target_pressure_expression"] = press_match.group(1)

    return result


def discover_tpc_runs(root: Path) -> list[Path]:
    markers = set()
    marker_names = ("nph_timeseries.dat", "nph_solid_profile.dat")
    for name in marker_names:
        for path in root.rglob(name):
            if path.is_file():
                markers.add(path.parent.resolve())

    for path in root.rglob("Melt_nph.*"):
        if path.is_file() and snapshot_step(path) is not None:
            markers.add(path.parent.resolve())

    return sorted(markers)


def viscosity_category(path: Path) -> str:
    name = path.name.lower()
    if any(token in name for token in ("sacf", "autocorr", "autocorrelation", "stress_acf", "stress-acf")):
        return "stress_autocorrelation"
    if any(token in name for token in ("cumulative", "running_integral", "running-integral")):
        return "cumulative_integral"
    if "plateau" in name:
        return "plateau"
    if path.suffix.lower() in {".png", ".pdf"}:
        return "diagnostic_plot"
    if "viscos" in name or "green_kubo" in name or "green-kubo" in name or "greenkubo" in name:
        return "viscosity_result_or_state"
    return "other_candidate"


def is_viscosity_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in VISCOSITY_ALLOWED_SUFFIXES:
        return False
    name = path.name.lower()
    return any(keyword in name for keyword in VISCOSITY_KEYWORDS)


def discover_viscosity_candidates(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if is_viscosity_candidate(path)
    )


class Collector:
    def __init__(
        self,
        outdir: Path,
        max_file_bytes: int,
        calculate_hash: bool,
        dry_run: bool,
    ) -> None:
        self.outdir = outdir
        self.max_file_bytes = max_file_bytes
        self.calculate_hash = calculate_hash
        self.dry_run = dry_run
        self.manifest_rows: list[dict[str, object]] = []
        self.issue_rows: list[dict[str, object]] = []

    def issue(
        self,
        workflow: str,
        label: str,
        source: Path,
        reason: str,
        detail: str = "",
    ) -> None:
        self.issue_rows.append(
            {
                "workflow": workflow,
                "label": label,
                "source": str(source),
                "reason": reason,
                "detail": detail,
            }
        )

    def copy(
        self,
        source: Path,
        destination: Path,
        workflow: str,
        label: str,
        category: str,
        run_source: Path,
    ) -> bool:
        try:
            size = source.stat().st_size
        except OSError as exc:
            self.issue(workflow, label, source, "stat_failed", str(exc))
            return False

        if size > self.max_file_bytes:
            self.issue(
                workflow,
                label,
                source,
                "over_size_limit",
                f"{size / 1024**2:.2f} MB",
            )
            return False

        copied = False
        if not self.dry_run:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied = True
            except OSError as exc:
                self.issue(workflow, label, source, "copy_failed", str(exc))
                return False

        hash_value = ""
        if self.calculate_hash:
            try:
                hash_value = sha256sum(source)
            except OSError as exc:
                self.issue(workflow, label, source, "hash_failed", str(exc))

        self.manifest_rows.append(
            {
                "workflow": workflow,
                "label": label,
                "category": category,
                "run_source": str(run_source),
                "source": str(source),
                "destination": str(destination),
                "size_bytes": size,
                "sha256": hash_value,
                "copied": copied,
            }
        )
        return True


def collect_tpc(
    spec: RootSpec,
    collector: Collector,
    all_snapshots: bool,
    include_state_files: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not spec.path.is_dir():
        collector.issue("TPC", spec.label, spec.path, "root_not_found")
        return rows

    runs = discover_tpc_runs(spec.path)
    if not runs:
        collector.issue("TPC", spec.label, spec.path, "no_runs_found")
        return rows

    for run_dir in runs:
        rel = relative_or_name(run_dir, spec.path)
        destination_run = collector.outdir / "TPC" / spec.label / rel
        copied_names: set[str] = set()

        for filename in TPC_CORE_FILES + TPC_OPTIONAL_PLOTS:
            source = run_dir / filename
            if source.is_file():
                if collector.copy(
                    source,
                    destination_run / filename,
                    "TPC",
                    spec.label,
                    "core_diagnostic" if filename in TPC_CORE_FILES else "diagnostic_plot",
                    run_dir,
                ):
                    copied_names.add(filename)

        if include_state_files:
            for filename in TPC_STATE_FILES:
                source = run_dir / filename
                if source.is_file():
                    if collector.copy(
                        source,
                        destination_run / filename,
                        "TPC",
                        spec.label,
                        "state_file",
                        run_dir,
                    ):
                        copied_names.add(filename)

        snapshots = select_tpc_snapshots(run_dir, all_snapshots)
        for source in snapshots:
            if collector.copy(
                source,
                destination_run / source.name,
                "TPC",
                spec.label,
                "nph_snapshot",
                run_dir,
            ):
                copied_names.add(source.name)

        metadata = extract_melt_metadata(run_dir)
        snapshot_steps = [
            snapshot_step(path) for path in snapshots if snapshot_step(path) is not None
        ]

        required = {
            "nph_timeseries.dat": (run_dir / "nph_timeseries.dat").is_file(),
            "nph_solid_profile.dat": (run_dir / "nph_solid_profile.dat").is_file(),
            "initial_snapshot": bool(snapshot_steps),
            "final_snapshot": len(snapshot_steps) >= 2,
        }
        missing = [name for name, present in required.items() if not present]
        for name in missing:
            collector.issue("TPC", spec.label, run_dir, "missing_required", name)

        rows.append(
            {
                "functional_or_label": spec.label,
                "source_root": str(spec.path),
                "run_source": str(run_dir),
                "relative_run_path": str(rel),
                "composition_text": metadata["composition_text"],
                "deepmd_model": metadata["deepmd_model"],
                "target_Tm_expression": metadata["target_Tm_expression"],
                "target_pressure_expression": metadata["target_pressure_expression"],
                "initial_snapshot_step": min(snapshot_steps) if snapshot_steps else "",
                "final_snapshot_step": max(snapshot_steps) if snapshot_steps else "",
                "has_nph_timeseries": required["nph_timeseries.dat"],
                "has_nph_solid_profile": required["nph_solid_profile.dat"],
                "has_initial_snapshot": required["initial_snapshot"],
                "has_final_snapshot": required["final_snapshot"],
                "collection_complete_for_interface_screening": not missing,
                "copied_file_count": len(copied_names),
                "missing_items": "; ".join(missing),
            }
        )

    return rows


def collect_viscosity(
    spec: RootSpec,
    collector: Collector,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not spec.path.is_dir():
        collector.issue("viscosity", spec.label, spec.path, "root_not_found")
        return rows

    candidates = discover_viscosity_candidates(spec.path)
    if not candidates:
        collector.issue(
            "viscosity",
            spec.label,
            spec.path,
            "no_candidate_diagnostics_found",
        )
        return rows

    by_directory: dict[Path, list[Path]] = {}
    for path in candidates:
        by_directory.setdefault(path.parent.resolve(), []).append(path)

    for source_dir, files in sorted(by_directory.items()):
        rel_dir = relative_or_name(source_dir, spec.path)
        destination_dir = collector.outdir / "Viscosity" / spec.label / rel_dir
        categories: set[str] = set()
        copied_names: set[str] = set()

        for source in sorted(files):
            category = viscosity_category(source)
            categories.add(category)
            if collector.copy(
                source,
                destination_dir / source.name,
                "viscosity",
                spec.label,
                category,
                source_dir,
            ):
                copied_names.add(source.name)

        # Copy nearby provenance files only when the directory already contains
        # a viscosity diagnostic candidate.
        for filename in VISCOSITY_PROVENANCE_NAMES:
            source = source_dir / filename
            if source.is_file() and source.name not in copied_names:
                if collector.copy(
                    source,
                    destination_dir / source.name,
                    "viscosity",
                    spec.label,
                    "provenance",
                    source_dir,
                ):
                    copied_names.add(source.name)

        has_acf = "stress_autocorrelation" in categories
        has_integral = "cumulative_integral" in categories
        has_result = "viscosity_result_or_state" in categories
        has_plateau = "plateau" in categories

        # Some workflows save ACF and cumulative integral in one general
        # viscosity state/result file. Mark such directories for manual review
        # rather than declaring them incomplete automatically.
        needs_manual_content_check = has_result and not (has_acf and has_integral)
        if not has_acf:
            collector.issue(
                "viscosity",
                spec.label,
                source_dir,
                "stress_acf_not_identified_by_filename",
                "May be embedded in a viscosity state/result file.",
            )
        if not has_integral:
            collector.issue(
                "viscosity",
                spec.label,
                source_dir,
                "cumulative_integral_not_identified_by_filename",
                "May be embedded in a viscosity state/result file.",
            )

        rows.append(
            {
                "functional_or_label": spec.label,
                "source_root": str(spec.path),
                "source_directory": str(source_dir),
                "relative_directory": str(rel_dir),
                "candidate_file_count": len(files),
                "copied_file_count": len(copied_names),
                "has_stress_autocorrelation_file": has_acf,
                "has_cumulative_integral_file": has_integral,
                "has_plateau_file": has_plateau,
                "has_general_viscosity_result_or_state": has_result,
                "needs_manual_content_check": needs_manual_content_check,
                "identified_categories": "; ".join(sorted(categories)),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path,
    args: argparse.Namespace,
    tpc_rows: list[dict[str, object]],
    viscosity_rows: list[dict[str, object]],
    collector: Collector,
) -> None:
    complete_tpc = sum(
        bool(row["collection_complete_for_interface_screening"])
        for row in tpc_rows
    )
    text = f"""Cu-Ag SI diagnostics collection
================================

TPC runs discovered: {len(tpc_rows)}
TPC runs complete for interface screening: {complete_tpc}

Viscosity diagnostic directories discovered: {len(viscosity_rows)}

Files recorded in collection_manifest.csv: {len(collector.manifest_rows)}
Missing/skipped records: {len(collector.issue_rows)}

TPC collection policy
---------------------
The default collection includes:
- nph_timeseries.dat
- nph_solid_profile.dat
- nph_msd_liquid.dat
- composition and run provenance
- diagnostic plots when present
- the first and last Melt_nph.* snapshots

The collector intentionally ignores any automated KEEP RUNNING or controller
verdict. Physical acceptance must be determined from retained interfaces and
the time dependence of the spatially resolved solid-slab width.

Viscosity collection policy
---------------------------
The collector searches filenames for Green-Kubo, stress-autocorrelation,
cumulative-integral, plateau, and viscosity keywords. Review
viscosity_directory_manifest.csv. A directory marked as needing manual content
check may store both the stress autocorrelation and cumulative integral inside
one NPZ/JSON/general viscosity state file.

No generic trajectory dumps are copied.

Command-line settings
---------------------
all_tpc_snapshots = {args.all_tpc_snapshots}
include_tpc_state_files = {args.include_tpc_state_files}
max_file_mb = {args.max_file_mb}
sha256 = {args.sha256}
dry_run = {args.dry_run}
"""
    path.write_text(text)


def create_archive(outdir: Path) -> Path:
    archive_path = outdir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(outdir, arcname=outdir.name)
    return archive_path


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()

    if outdir.exists() and not args.dry_run:
        print(
            f"ERROR: output directory already exists: {outdir}\n"
            "Choose a new --outdir or remove the existing directory.",
            file=sys.stderr,
        )
        return 2

    outdir.mkdir(parents=True, exist_ok=True)
    collector = Collector(
        outdir=outdir,
        max_file_bytes=int(args.max_file_mb * 1024**2),
        calculate_hash=args.sha256,
        dry_run=args.dry_run,
    )

    tpc_rows: list[dict[str, object]] = []
    viscosity_rows: list[dict[str, object]] = []

    for raw in args.tpc_root:
        spec = parse_root_spec(raw)
        tpc_rows.extend(
            collect_tpc(
                spec,
                collector,
                all_snapshots=args.all_tpc_snapshots,
                include_state_files=args.include_tpc_state_files,
            )
        )

    for raw in args.viscosity_root:
        spec = parse_root_spec(raw)
        viscosity_rows.extend(collect_viscosity(spec, collector))

    write_csv(outdir / "collection_manifest.csv", collector.manifest_rows)
    write_csv(outdir / "tpc_run_manifest.csv", tpc_rows)
    write_csv(
        outdir / "viscosity_directory_manifest.csv",
        viscosity_rows,
    )
    write_csv(outdir / "missing_or_skipped.csv", collector.issue_rows)
    write_readme(
        outdir / "README_collection.txt",
        args,
        tpc_rows,
        viscosity_rows,
        collector,
    )

    print(f"Output directory: {outdir}")
    print(f"TPC runs discovered: {len(tpc_rows)}")
    print(f"Viscosity diagnostic directories: {len(viscosity_rows)}")
    print(f"Manifested files: {len(collector.manifest_rows)}")
    print(f"Missing/skipped records: {len(collector.issue_rows)}")

    if args.archive and not args.dry_run:
        archive_path = create_archive(outdir)
        print(f"Archive: {archive_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
