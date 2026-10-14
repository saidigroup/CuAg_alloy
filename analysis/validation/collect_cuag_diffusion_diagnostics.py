#!/usr/bin/env python3
"""
Collect Cu-Ag diffusion diagnostics for Supporting Information.

The script accepts one or more labeled diffusion roots, finds individual
diffusion run directories, copies raw MSD data and provenance, and builds
manifests suitable for selecting representative PBE/PBEsol diagnostics.

Typical use
-----------
python3 collect_cuag_diffusion_diagnostics.py \
  --root PBE=/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au/Validation/Diffusion \
  --root PBEsol=/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au_sol/Validation/Diffusion \
  --processed-dir diffusion_reprocessed_v7_temperature_shading \
  --outdir CuAg_diffusion_SI_diagnostics \
  --archive

Default collection
------------------
For each identified run directory, the script copies:
  * raw MSD files and species-resolved MSD files;
  * diffusion fit/summary files;
  * LAMMPS input and log files;
  * model-path/provenance text;
  * final diagnostic plots when present.

It deliberately avoids copying generic trajectory dumps and large restart
files. Use --max-file-mb to change the individual-file limit.

Outputs
-------
  collection_manifest.csv
  run_manifest.csv
  missing_or_skipped.csv
  representative_case_manifest.csv
  README_collection.txt
  <outdir>.tar.gz  (with --archive)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


RAW_MSD_PATTERNS = (
    "*msd*.dat", "*MSD*.dat",
    "*msd*.txt", "*MSD*.txt",
    "*msd*.csv", "*MSD*.csv",
    "*msd*.npy", "*MSD*.npy",
    "*msd*.npz", "*MSD*.npz",
)

FIT_PATTERNS = (
    "*diffusion*.csv", "*diffusion*.txt", "*diffusion*.json",
    "*fit*.csv", "*fit*.txt", "*fit*.json",
    "*Einstein*.dat", "*einstein*.dat",
    "*bootstrap*.csv", "*bootstrap*.json",
)

PLOT_PATTERNS = (
    "*msd*.png", "*MSD*.png",
    "*diffusion*.png", "*fit*.png",
    "*msd*.pdf", "*diffusion*.pdf",
)

PROVENANCE_NAMES = (
    "log.lammps", "out", "out_diffusion", "out_msd",
    "diffusion.in", "in.diffusion", "thermo.in", "in.thermo",
    "submit", "submit.sh", "submit_diffusion",
    "potential_path.txt", "model_path.txt",
    "data.lmp", "final.data",
)

RUN_MARKER_PATTERNS = RAW_MSD_PATTERNS + (
    "diffusion_summary.json",
    "diffusion_results.csv",
    "diffusion_fit.csv",
)

PROCESSED_NAMES = (
    "diffusion_raw.csv",
    "diffusion_raw_PBE.csv",
    "diffusion_aggregate.csv",
    "diffusion_aggregate_PBE.csv",
    "endpoint_outliers_excluded.csv",
)

CASE_FIT_RE = re.compile(
    r"^T(?P<T>\d+)_Cu(?P<cu>\d+)_(?P<species>Cu|Ag|all)_(?P<run>[^.]+)\.(?P<ext>png|pdf)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RootSpec:
    label: str
    path: Path


def sanitize(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return result or "unlabeled"


def parse_root(raw: str) -> RootSpec:
    if "=" in raw:
        label, path = raw.split("=", 1)
    else:
        path = raw
        label = Path(path).name or "root"
    return RootSpec(sanitize(label), Path(path).expanduser().resolve())


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Cu-Ag diffusion diagnostics for the SI."
    )
    parser.add_argument(
        "--root", action="append", required=True, metavar="LABEL=PATH",
        help="Diffusion root. Supply once for each functional."
    )
    parser.add_argument(
        "--processed-dir",
        help="Optional processed-output directory containing case_fits and diffusion CSVs."
    )
    parser.add_argument(
        "--outdir", default="CuAg_diffusion_SI_diagnostics",
        help="Destination directory."
    )
    parser.add_argument(
        "--representative-temperatures", default="1423,1573",
        help="Comma-separated temperatures used to flag representative cases."
    )
    parser.add_argument(
        "--representative-compositions", default="0,50,80,100",
        help="Comma-separated Cu at.%% values used to flag representative cases."
    )
    parser.add_argument(
        "--max-file-mb", type=float, default=100.0,
        help="Maximum size of an individual copied file."
    )
    parser.add_argument(
        "--sha256", action="store_true",
        help="Calculate SHA-256 for copied files."
    )
    parser.add_argument(
        "--archive", action="store_true",
        help="Create OUTDIR.tar.gz after collection."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write manifests without copying files."
    )
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def parse_path_metadata(path: Path) -> dict[str, object]:
    text = str(path)
    temperature = None
    composition = None
    replicate = ""

    matches = list(re.finditer(r"(?:^|[/_-])T(\d{3,4})(?:$|[/_-])", text))
    if matches:
        temperature = int(matches[-1].group(1))

    matches = list(re.finditer(r"(?:Cu[_-]?|Cu_Ag[_-]?)(\d{1,3})(?:i)?(?:$|[/_-])", text))
    if matches:
        composition = int(matches[-1].group(1))

    for part in reversed(path.parts):
        if re.fullmatch(r"rep\d+", part, re.IGNORECASE):
            replicate = part
            break
        if part in {"A", "B", "C", "main"}:
            replicate = part
            break

    return {
        "temperature_K": temperature if temperature is not None else "",
        "Cu_at_pct": composition if composition is not None else "",
        "replicate": replicate,
    }


def extract_versions_and_model(run_dir: Path) -> dict[str, str]:
    result = {
        "lammps_version": "",
        "deepmd_source_version": "",
        "deepmd_environment": "",
        "model_path": "",
    }
    candidates = [
        run_dir / "out_diffusion",
        run_dir / "out",
        run_dir / "log.lammps",
        run_dir / "out_msd",
    ]
    text = ""
    for path in candidates:
        if path.is_file() and path.stat().st_size < 20_000_000:
            try:
                text += "\n" + path.read_text(errors="replace")
            except OSError:
                pass

    m = re.search(r"^LAMMPS \(([^)]+)\)", text, flags=re.MULTILINE)
    if m:
        result["lammps_version"] = m.group(1).strip()

    m = re.search(r"^\s*source:\s*(v[0-9][^\s]*)", text, flags=re.MULTILINE)
    if m:
        result["deepmd_source_version"] = m.group(1)

    m = re.search(r"installed to:\s*(\S+)", text)
    if m:
        result["deepmd_environment"] = m.group(1)

    model_patterns = (
        r"using\s+1 model\(s\):\s*(\S+\.pb)",
        r"pair_style\s+deepmd\s+(\S+\.pb)",
    )
    for pattern in model_patterns:
        m = re.search(pattern, text)
        if m:
            result["model_path"] = m.group(1)
            break

    if not result["model_path"]:
        for name in ("diffusion.in", "in.diffusion", "thermo.in", "in.thermo"):
            path = run_dir / name
            if path.is_file():
                try:
                    source = path.read_text(errors="replace")
                except OSError:
                    continue
                m = re.search(r"pair_style\s+deepmd\s+(\S+\.pb)", source)
                if m:
                    result["model_path"] = m.group(1)
                    break

    return result


def discover_run_dirs(root: Path) -> list[Path]:
    runs: set[Path] = set()
    for pattern in RUN_MARKER_PATTERNS:
        for path in root.rglob(pattern):
            if path.is_file():
                runs.add(path.parent.resolve())
    return sorted(runs)


def collect_matching_files(run_dir: Path) -> list[tuple[Path, str]]:
    selected: dict[Path, str] = {}

    for pattern in RAW_MSD_PATTERNS:
        for path in run_dir.glob(pattern):
            if path.is_file():
                selected[path] = "raw_msd"

    for pattern in FIT_PATTERNS:
        for path in run_dir.glob(pattern):
            if path.is_file():
                selected.setdefault(path, "fit_or_summary")

    for pattern in PLOT_PATTERNS:
        for path in run_dir.glob(pattern):
            if path.is_file():
                selected.setdefault(path, "diagnostic_plot")

    for name in PROVENANCE_NAMES:
        path = run_dir / name
        if path.is_file():
            selected.setdefault(path, "provenance")

    return sorted(selected.items())


class Collector:
    def __init__(
        self, outdir: Path, max_bytes: int, hashes: bool, dry_run: bool
    ) -> None:
        self.outdir = outdir
        self.max_bytes = max_bytes
        self.hashes = hashes
        self.dry_run = dry_run
        self.files: list[dict[str, object]] = []
        self.issues: list[dict[str, object]] = []

    def issue(self, label: str, source: Path, reason: str, detail: str = "") -> None:
        self.issues.append({
            "label": label,
            "source": str(source),
            "reason": reason,
            "detail": detail,
        })

    def copy(
        self, source: Path, destination: Path, label: str,
        category: str, run_dir: Path
    ) -> bool:
        try:
            size = source.stat().st_size
        except OSError as exc:
            self.issue(label, source, "stat_failed", str(exc))
            return False

        if size > self.max_bytes:
            self.issue(
                label, source, "over_size_limit",
                f"{size / 1024**2:.2f} MB"
            )
            return False

        copied = False
        if not self.dry_run:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied = True
            except OSError as exc:
                self.issue(label, source, "copy_failed", str(exc))
                return False

        digest = ""
        if self.hashes:
            try:
                digest = sha256sum(source)
            except OSError as exc:
                self.issue(label, source, "hash_failed", str(exc))

        self.files.append({
            "functional_or_label": label,
            "category": category,
            "run_source": str(run_dir),
            "source": str(source),
            "destination": str(destination),
            "size_bytes": size,
            "sha256": digest,
            "copied": copied,
        })
        return True


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_processed(
    processed_dir: Path, outdir: Path, collector: Collector,
    representative_t: set[int], representative_cu: set[int]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not processed_dir.is_dir():
        collector.issue("processed", processed_dir, "processed_dir_not_found")
        return rows

    destination = outdir / "Processed"
    for name in PROCESSED_NAMES:
        source = processed_dir / name
        if source.is_file():
            collector.copy(
                source, destination / name, "processed",
                "processed_summary", processed_dir
            )

    case_dir = processed_dir / "case_fits"
    if not case_dir.is_dir():
        collector.issue("processed", case_dir, "case_fits_not_found")
        return rows

    for source in sorted(case_dir.iterdir()):
        if not source.is_file():
            continue
        match = CASE_FIT_RE.match(source.name)
        if not match:
            continue
        temperature = int(match.group("T"))
        composition = int(match.group("cu"))
        representative = (
            temperature in representative_t and
            composition in representative_cu
        )
        category = "representative_case_fit" if representative else "case_fit"
        if collector.copy(
            source, destination / "case_fits" / source.name,
            "processed", category, case_dir
        ):
            rows.append({
                "functional_or_label": "processed",
                "temperature_K": temperature,
                "Cu_at_pct": composition,
                "species": match.group("species"),
                "run_label": match.group("run"),
                "source": str(source),
                "representative": representative,
            })
    return rows


def create_archive(outdir: Path) -> Path:
    archive_path = Path(str(outdir) + ".tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(outdir, arcname=outdir.name)
    return archive_path


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    if outdir.exists() and not args.dry_run:
        raise SystemExit(
            f"Output directory already exists: {outdir}\n"
            "Choose a different --outdir or remove it."
        )
    outdir.mkdir(parents=True, exist_ok=True)

    collector = Collector(
        outdir,
        max_bytes=int(args.max_file_mb * 1024**2),
        hashes=args.sha256,
        dry_run=args.dry_run,
    )
    representative_t = set(parse_int_list(args.representative_temperatures))
    representative_cu = set(parse_int_list(args.representative_compositions))

    run_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []

    for raw in args.root:
        spec = parse_root(raw)
        if not spec.path.is_dir():
            collector.issue(spec.label, spec.path, "root_not_found")
            continue

        run_dirs = discover_run_dirs(spec.path)
        if not run_dirs:
            collector.issue(spec.label, spec.path, "no_run_directories_found")

        for run_dir in run_dirs:
            rel = relative_path(run_dir, spec.path)
            destination = outdir / "Raw" / spec.label / rel
            metadata = parse_path_metadata(run_dir)
            versions = extract_versions_and_model(run_dir)
            selected = collect_matching_files(run_dir)

            counts = {"raw_msd": 0, "fit_or_summary": 0,
                      "diagnostic_plot": 0, "provenance": 0}
            for source, category in selected:
                if collector.copy(
                    source, destination / source.name, spec.label,
                    category, run_dir
                ):
                    counts[category] += 1

            complete = counts["raw_msd"] > 0
            if not complete:
                collector.issue(
                    spec.label, run_dir, "raw_msd_not_found",
                    "No filename matched the configured MSD patterns."
                )

            representative = (
                metadata["temperature_K"] in representative_t and
                metadata["Cu_at_pct"] in representative_cu
            )

            row = {
                "functional_or_label": spec.label,
                "source_root": str(spec.path),
                "run_source": str(run_dir),
                "relative_run_path": str(rel),
                **metadata,
                **versions,
                "raw_msd_files": counts["raw_msd"],
                "fit_or_summary_files": counts["fit_or_summary"],
                "diagnostic_plot_files": counts["diagnostic_plot"],
                "provenance_files": counts["provenance"],
                "complete_for_msd_diagnostic": complete,
                "representative_candidate": representative,
            }
            run_rows.append(row)
            if representative:
                representative_rows.append(row)

    if args.processed_dir:
        processed_rows = collect_processed(
            Path(args.processed_dir).expanduser().resolve(),
            outdir, collector, representative_t, representative_cu
        )
        representative_rows.extend(
            row for row in processed_rows if row["representative"]
        )

    write_csv(outdir / "collection_manifest.csv", collector.files)
    write_csv(outdir / "run_manifest.csv", run_rows)
    write_csv(outdir / "missing_or_skipped.csv", collector.issues)
    write_csv(
        outdir / "representative_case_manifest.csv",
        representative_rows
    )

    versions = sorted({
        str(row["lammps_version"]) for row in run_rows
        if row["lammps_version"]
    })
    deepmd_versions = sorted({
        str(row["deepmd_source_version"]) for row in run_rows
        if row["deepmd_source_version"]
    })

    readme = f"""Cu-Ag diffusion SI diagnostic collection
==========================================

Run directories discovered: {len(run_rows)}
Runs with raw MSD files: {sum(bool(r['complete_for_msd_diagnostic']) for r in run_rows)}
Representative candidates: {sum(bool(r.get('representative_candidate')) for r in run_rows)}
Files manifested: {len(collector.files)}
Missing/skipped records: {len(collector.issues)}

LAMMPS versions found: {', '.join(versions) if versions else 'not identified'}
DeePMD source versions found: {', '.join(deepmd_versions) if deepmd_versions else 'not identified'}

Recommended Figure S8 selection
-------------------------------
Use one Ag-rich, one intermediate, and one Cu-rich liquid for each
functional, preferably at matched temperatures. For each selected case show:
  1. species-resolved MSD versus time;
  2. the accepted long-time fit interval;
  3. the fitted line;
  4. D, R^2, and the MSD displacement over the fit interval.

The representative-case manifest only flags candidates. Final selection
should favor successful fits with large MSD displacement and R^2 >= 0.98.
"""
    (outdir / "README_collection.txt").write_text(readme)

    print(f"Output directory: {outdir}")
    print(f"Run directories discovered: {len(run_rows)}")
    print(f"Files manifested: {len(collector.files)}")
    print(f"Missing/skipped records: {len(collector.issues)}")

    if args.archive and not args.dry_run:
        archive = create_archive(outdir)
        print(f"Archive: {archive}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
