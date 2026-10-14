#!/usr/bin/env python3
"""
Process and plot Cu-Ag Green-Kubo viscosity data for paired PBE/PBEsol DNP runs.

Expected directory layouts:

  PBE_ROOT/T1423/Cu_20/rep01/viscosity_summary.csv
  PBE_ROOT/T1423/Cu_20/rep02/viscosity_summary.csv
  PBEsol_ROOT/T1423/Cu_20/rep01/viscosity_summary.csv

Direct state files are also supported when replicate folders are absent:

  PBE_ROOT/T1423/Cu_20/viscosity_summary.csv

The script reads per-run summary files written by viscosity_gk_post.py, aggregates
replicate means/SEMs, compares PBE and PBEsol, and generates publication-style QC plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

T_DIR_RE = re.compile(r"^T(?P<T>\d+(?:\.\d+)?)$")
CU_DIR_RE = re.compile(r"^Cu_(?P<Cu>\d+(?:\.\d+)?)$")
REP_DIR_RE = re.compile(r"^rep(?P<rep>\d+)$")

# Brillo et al., High Temperatures-High Pressures 47 (2018) 417-441, Table 3.
# Values reported at 1375 K as Ag-Cu compositions and represented here as Cu at. %.
BUILTIN_BRILLO_2018_1375K = {
    100.0: (4.30e-3, 0.86e-3),
    90.0:  (4.20e-3, 0.84e-3),
    80.0:  (3.97e-3, 0.79e-3),
    70.0:  (3.76e-3, 0.75e-3),
    60.0:  (3.36e-3, 0.67e-3),
    50.0:  (2.89e-3, 0.58e-3),
    40.0:  (2.69e-3, 0.54e-3),
    30.0:  (2.69e-3, 0.54e-3),
    20.0:  (2.62e-3, 0.52e-3),
    10.0:  (2.65e-3, 0.53e-3),
    0.0:   (2.85e-3, 0.60e-3),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate and plot PBE/PBEsol Cu-Ag Green-Kubo viscosity summaries."
    )
    p.add_argument(
        "--pbe-root",
        default="/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au/Validation/Viscosity",
        help="Root containing PBE viscosity runs, with T*/Cu_*[/rep*] folders.",
    )
    p.add_argument(
        "--pbesol-root",
        default="/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au_sol/Validation/Viscosity",
        help="Root containing PBEsol viscosity runs, with T*/Cu_*[/rep*] folders.",
    )
    p.add_argument("--pbe-label", default="PBE")
    p.add_argument("--pbesol-label", default="PBEsol")
    p.add_argument("--outdir", default="viscosity_pbe_pbesol_processed")
    p.add_argument("--summary-name", default="viscosity_summary.csv")
    p.add_argument("--eta-column", default="eta_avg_Pa_s")
    p.add_argument("--sem-column", default="eta_avg_tail_sem_Pa_s")
    p.add_argument(
        "--allow-direct-state-files",
        action="store_true",
        help="Allow T*/Cu_*/viscosity_summary.csv direct files. Replicate files are always allowed.",
    )
    p.add_argument(
        "--keep-direct-if-reps",
        action="store_true",
        help="Keep direct T*/Cu_* summaries even if rep*/ summaries exist. Default drops direct summaries when reps exist.",
    )
    p.add_argument(
        "--include-singletons",
        action="store_true",
        default=True,
        help="Include state points with one valid run. Default: on.",
    )
    p.add_argument(
        "--exclude-singletons",
        action="store_true",
        help="Exclude state points with only one valid run.",
    )
    p.add_argument(
        "--temperatures",
        default="",
        help="Optional comma-separated T list to keep, e.g. 1275,1375,1423.",
    )
    p.add_argument(
        "--compositions",
        default="",
        help="Optional comma-separated Cu at.% list to keep, e.g. 0,20,40,50,60,80,100.",
    )
    p.add_argument("--min-eta", type=float, default=0.0, help="Minimum accepted viscosity in Pa*s. Default: >0.")
    p.add_argument("--max-eta", type=float, default=float("nan"), help="Maximum accepted viscosity in Pa*s. Default: no upper filter.")
    p.add_argument(
        "--max-component-cv",
        type=float,
        default=float("nan"),
        help="Optional maximum coefficient of variation among eta_xy/xz/yz components. Useful QC filter.",
    )
    p.add_argument("--exp-csv", default=None, help="Optional experimental/reference viscosity CSV.")
    p.add_argument("--no-built-in-exp", action="store_true", help="Disable built-in Brillo 2018 1375 K overlay.")
    p.add_argument("--units", choices=["mPa_s", "Pa_s"], default="mPa_s", help="Plot units. CSV always stores Pa*s and mPa*s.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show-replicates", action="store_true", help="Overlay individual replicate points on composition plots.")
    return p.parse_args()


def safe_float(x) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_float_list(s: str) -> Optional[set[float]]:
    if not s:
        return None
    out: set[float] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.add(float(tok))
    return out


def near_in_set(x: float, values: Optional[set[float]], tol: float = 1.0e-6) -> bool:
    if values is None:
        return True
    return any(abs(float(x) - float(v)) <= tol for v in values)


def extract_state(path: Path) -> Tuple[float, float, Optional[int]]:
    T = float("nan")
    Cu = float("nan")
    rep: Optional[int] = None
    for part in path.parts:
        mt = T_DIR_RE.match(part)
        if mt:
            T = float(mt.group("T"))
        mc = CU_DIR_RE.match(part)
        if mc:
            Cu = float(mc.group("Cu"))
        mr = REP_DIR_RE.match(part)
        if mr:
            rep = int(mr.group("rep"))
    return T, Cu, rep


def state_key(path: Path) -> Tuple[float, float]:
    T, Cu, _ = extract_state(path)
    return T, Cu


def is_valid_state_path(path: Path) -> bool:
    T, Cu, _ = extract_state(path)
    return math.isfinite(T) and math.isfinite(Cu)


def find_summary_files(
    root: Path,
    summary_name: str,
    allow_direct: bool,
    keep_direct_if_reps: bool,
    keep_T: Optional[set[float]],
    keep_Cu: Optional[set[float]],
) -> List[Path]:
    all_files = sorted(p for p in root.rglob(summary_name) if p.is_file() and is_valid_state_path(p))

    # Filter T/composition first.
    filtered: List[Path] = []
    for p in all_files:
        T, Cu, rep = extract_state(p)
        if not near_in_set(T, keep_T):
            continue
        if not near_in_set(Cu, keep_Cu):
            continue
        if rep is not None:
            filtered.append(p)
        elif allow_direct:
            filtered.append(p)

    if keep_direct_if_reps:
        return filtered

    # Drop direct T*/Cu_*/summary files when rep*/summary files exist for same T/Cu.
    has_rep = defaultdict(bool)
    for p in filtered:
        T, Cu, rep = extract_state(p)
        if rep is not None:
            has_rep[(T, Cu)] = True

    out: List[Path] = []
    for p in filtered:
        T, Cu, rep = extract_state(p)
        if rep is None and has_rep[(T, Cu)]:
            continue
        out.append(p)
    return sorted(out)


def read_csv_first_row(path: Path) -> Dict[str, str]:
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def component_cv(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if arr.size < 2:
        return float("nan")
    mean_abs = float(np.mean(np.abs(arr)))
    if mean_abs == 0:
        return float("inf")
    return float(np.std(arr, ddof=1) / mean_abs)


def row_status(eta: float, min_eta: float, max_eta: float, comp_cv: float, max_component_cv: float) -> str:
    if not math.isfinite(eta):
        return "nonfinite_eta"
    if eta <= min_eta:
        return "below_min_eta"
    if math.isfinite(max_eta) and eta > max_eta:
        return "above_max_eta"
    if math.isfinite(max_component_cv) and math.isfinite(comp_cv) and comp_cv > max_component_cv:
        return "high_component_cv"
    return "ok"


def build_rows_for_functional(
    label: str,
    root: Path,
    files: Sequence[Path],
    eta_column: str,
    sem_column: str,
    min_eta: float,
    max_eta: float,
    max_component_cv: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    auto_rep = defaultdict(int)
    for p in files:
        T, Cu, rep = extract_state(p)
        if rep is None:
            auto_rep[(T, Cu)] += 1
            rep_val = auto_rep[(T, Cu)]
            run_label = "main"
        else:
            rep_val = rep
            run_label = f"rep{rep:02d}"
        try:
            r = read_csv_first_row(p)
        except Exception as exc:
            rows.append({
                "functional": label,
                "T_K": T,
                "Cu_atpct": Cu,
                "replicate": rep_val,
                "run_label": run_label,
                "eta_Pa_s": float("nan"),
                "eta_mPa_s": float("nan"),
                "eta_run_sem_Pa_s": float("nan"),
                "eta_xy_Pa_s": float("nan"),
                "eta_xz_Pa_s": float("nan"),
                "eta_yz_Pa_s": float("nan"),
                "eta_lammps_Pa_s": float("nan"),
                "volume_A3": float("nan"),
                "corr_time_max_ps": float("nan"),
                "component_cv": float("nan"),
                "status": f"read_error:{exc}",
                "folder": str(p.parent),
                "summary_file": str(p),
            })
            continue
        eta = safe_float(r.get(eta_column))
        eta_sem = safe_float(r.get(sem_column))
        eta_xy = safe_float(r.get("eta_xy_Pa_s"))
        eta_xz = safe_float(r.get("eta_xz_Pa_s"))
        eta_yz = safe_float(r.get("eta_yz_Pa_s"))
        cv = component_cv([eta_xy, eta_xz, eta_yz])
        rows.append({
            "functional": label,
            "T_K": T,
            "Cu_atpct": Cu,
            "replicate": rep_val,
            "run_label": run_label,
            "eta_Pa_s": eta,
            "eta_mPa_s": eta * 1000.0 if math.isfinite(eta) else float("nan"),
            "eta_run_sem_Pa_s": eta_sem,
            "eta_xy_Pa_s": eta_xy,
            "eta_xz_Pa_s": eta_xz,
            "eta_yz_Pa_s": eta_yz,
            "eta_lammps_Pa_s": safe_float(r.get("eta_lammps_Pa_s")),
            "volume_A3": safe_float(r.get("volume_A3")),
            "corr_time_max_ps": safe_float(r.get("corr_time_max_ps")),
            "component_cv": cv,
            "status": row_status(eta, min_eta, max_eta, cv, max_component_cv),
            "folder": str(p.parent),
            "summary_file": str(p),
        })
    rows.sort(key=lambda x: (str(x["functional"]), float(x["T_K"]), float(x["Cu_atpct"]), int(x["replicate"])))
    return rows


def sem(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def aggregate_rows(raw_rows: Sequence[Dict[str, object]], include_singletons: bool) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float, float], List[Dict[str, object]]] = defaultdict(list)
    for r in raw_rows:
        if r["status"] != "ok":
            continue
        grouped[(str(r["functional"]), float(r["T_K"]), float(r["Cu_atpct"]))].append(r)

    out: List[Dict[str, object]] = []
    for (func, T, Cu), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        vals = [float(g["eta_Pa_s"]) for g in group]
        vals = [v for v in vals if math.isfinite(v)]
        n = len(vals)
        if n == 0:
            continue
        if n == 1 and not include_singletons:
            continue
        run_sems = [float(g["eta_run_sem_Pa_s"]) for g in group if math.isfinite(float(g["eta_run_sem_Pa_s"]))]
        rep_sem = sem(vals)
        plot_sem = rep_sem if math.isfinite(rep_sem) else (float(np.mean(run_sems)) if run_sems else 0.0)
        comp_cvs = [float(g["component_cv"]) for g in group if math.isfinite(float(g["component_cv"]))]
        out.append({
            "functional": func,
            "T_K": T,
            "Cu_atpct": Cu,
            "n_runs_total": len(group),
            "n_runs_ok": n,
            "eta_mean_Pa_s": float(np.mean(vals)),
            "eta_mean_mPa_s": float(np.mean(vals)) * 1000.0,
            "eta_sem_Pa_s": rep_sem,
            "eta_sem_mPa_s": rep_sem * 1000.0 if math.isfinite(rep_sem) else float("nan"),
            "eta_plot_sem_Pa_s": plot_sem,
            "eta_plot_sem_mPa_s": plot_sem * 1000.0,
            "eta_min_Pa_s": float(np.min(vals)),
            "eta_max_Pa_s": float(np.max(vals)),
            "mean_run_tail_sem_Pa_s": float(np.mean(run_sems)) if run_sems else float("nan"),
            "component_cv_mean": float(np.mean(comp_cvs)) if comp_cvs else float("nan"),
            "run_labels": ";".join(str(g["run_label"]) for g in group),
        })
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def load_exp_csv(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            T = safe_float(row.get("T_K", row.get("temperature_K", row.get("Temperature_K", row.get("temperature")))))
            Cu = safe_float(row.get("Cu_atpct", row.get("Cu_pct", row.get("Cu composition", row.get("Cu_atpct_percent")))))
            eta = safe_float(row.get("eta_Pa_s", row.get("viscosity_Pa_s", row.get("eta", row.get("viscosity")))))
            eta_sem = safe_float(row.get("eta_sem_Pa_s", row.get("viscosity_sem_Pa_s")))
            if not (math.isfinite(T) and math.isfinite(Cu) and math.isfinite(eta)):
                continue
            out.append({
                "label": row.get("label") or row.get("source") or "Experiment",
                "T_K": T,
                "Cu_atpct": Cu,
                "eta_Pa_s": eta,
                "eta_mPa_s": eta * 1000.0,
                "eta_sem_Pa_s": eta_sem,
                "eta_sem_mPa_s": eta_sem * 1000.0 if math.isfinite(eta_sem) else float("nan"),
            })
    return out


def builtin_exp_rows() -> List[Dict[str, object]]:
    out = []
    for Cu, (eta, eta_sem) in BUILTIN_BRILLO_2018_1375K.items():
        out.append({
            "label": "Brillo2018",
            "T_K": 1375.0,
            "Cu_atpct": Cu,
            "eta_Pa_s": eta,
            "eta_mPa_s": eta * 1000.0,
            "eta_sem_Pa_s": eta_sem,
            "eta_sem_mPa_s": eta_sem * 1000.0,
        })
    return out


def ycols(units: str) -> Tuple[str, str, str, str]:
    if units == "Pa_s":
        return "eta_mean_Pa_s", "eta_plot_sem_Pa_s", "eta_Pa_s", "eta_sem_Pa_s"
    return "eta_mean_mPa_s", "eta_plot_sem_mPa_s", "eta_mPa_s", "eta_sem_mPa_s"


def ylabel(units: str) -> str:
    return r"Viscosity, $\eta$ (Pa$\cdot$s)" if units == "Pa_s" else r"Viscosity, $\eta$ (mPa$\cdot$s)"


def plot_composition(agg: Sequence[Dict[str, object]], raw: Sequence[Dict[str, object]], exp: Sequence[Dict[str, object]], outpath: Path, units: str, show_reps: bool, dpi: int) -> None:
    ymean, ysem, yexp, yexpsem = ycols(units)
    funcs = sorted(set(str(r["functional"]) for r in agg))
    temps = sorted(set(float(r["T_K"]) for r in agg))
    if not funcs or not temps:
        return
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    markers = {"PBE": "D", "PBEsol": "s"}
    for T in temps:
        for func in funcs:
            pts = sorted([r for r in agg if str(r["functional"]) == func and float(r["T_K"]) == T], key=lambda r: float(r["Cu_atpct"]))
            if not pts:
                continue
            ax.errorbar(
                [float(p["Cu_atpct"]) for p in pts],
                [float(p[ymean]) for p in pts],
                yerr=[float(p[ysem]) for p in pts],
                marker=markers.get(func, "o"),
                linestyle="--",
                capsize=3,
                label=f"{func} {T:.0f} K",
            )
    if show_reps:
        for r in raw:
            if r.get("status") != "ok":
                continue
            ax.scatter(float(r["Cu_atpct"]), float(r["eta_mPa_s"] if units == "mPa_s" else r["eta_Pa_s"]), s=12, alpha=0.35)
    for label_T in sorted(set((str(e["label"]), float(e["T_K"])) for e in exp)):
        lab, T = label_T
        pts = sorted([e for e in exp if str(e["label"]) == lab and float(e["T_K"]) == T], key=lambda e: float(e["Cu_atpct"]))
        ax.errorbar(
            [float(p["Cu_atpct"]) for p in pts],
            [float(p[yexp]) for p in pts],
            yerr=[0.0 if not math.isfinite(float(p[yexpsem])) else float(p[yexpsem]) for p in pts],
            marker="o",
            linestyle=":",
            capsize=3,
            label=f"{lab} {T:.0f} K",
        )
    ax.set_xlabel("Cu composition (at. %)", fontweight="bold")
    ax.set_ylabel(ylabel(units), fontweight="bold")
    ax.tick_params(direction="in")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_temperature(agg: Sequence[Dict[str, object]], outpath: Path, units: str, dpi: int) -> None:
    ymean, ysem, _, _ = ycols(units)
    funcs = sorted(set(str(r["functional"]) for r in agg))
    comps = sorted(set(float(r["Cu_atpct"]) for r in agg))
    if not funcs or not comps:
        return
    fig, axes = plt.subplots(1, len(funcs), figsize=(6.0 * len(funcs), 5.2), sharey=True)
    if len(funcs) == 1:
        axes = [axes]
    for ax, func in zip(axes, funcs):
        for Cu in comps:
            pts = sorted([r for r in agg if str(r["functional"]) == func and abs(float(r["Cu_atpct"]) - Cu) < 1e-6], key=lambda r: float(r["T_K"]))
            if not pts:
                continue
            ax.errorbar(
                [float(p["T_K"]) for p in pts],
                [float(p[ymean]) for p in pts],
                yerr=[float(p[ysem]) for p in pts],
                marker="o",
                capsize=3,
                label=f"Cu {Cu:.0f}%",
            )
        ax.set_title(func)
        ax.set_xlabel("Temperature (K)", fontweight="bold")
        ax.tick_params(direction="in")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel(units), fontweight="bold")
    axes[-1].legend(fontsize=8, ncol=1, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_delta_ratio(delta_rows: Sequence[Dict[str, object]], outpath: Path, units: str, dpi: int) -> None:
    if not delta_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    temps = sorted(set(float(r["T_K"]) for r in delta_rows))
    dy_col = "delta_PBEsol_minus_PBE_mPa_s" if units == "mPa_s" else "delta_PBEsol_minus_PBE_Pa_s"
    dy_sem_col = "delta_sem_mPa_s" if units == "mPa_s" else "delta_sem_Pa_s"
    for T in temps:
        pts = sorted([r for r in delta_rows if abs(float(r["T_K"]) - T) < 1e-6], key=lambda r: float(r["Cu_atpct"]))
        axes[0].errorbar(
            [float(p["Cu_atpct"]) for p in pts],
            [float(p[dy_col]) for p in pts],
            yerr=[float(p[dy_sem_col]) for p in pts],
            marker="o",
            capsize=3,
            label=f"{T:.0f} K",
        )
        axes[1].plot(
            [float(p["Cu_atpct"]) for p in pts],
            [float(p["ratio_PBE_over_PBEsol"]) for p in pts],
            marker="s",
            linestyle="--",
            label=f"{T:.0f} K",
        )
    axes[0].axhline(0.0, lw=1.0)
    axes[0].set_xlabel("Cu composition (at. %)", fontweight="bold")
    axes[0].set_ylabel("PBEsol - PBE " + ylabel(units).replace("Viscosity, ", ""), fontweight="bold")
    axes[1].axhline(1.0, lw=1.0)
    axes[1].set_xlabel("Cu composition (at. %)", fontweight="bold")
    axes[1].set_ylabel("PBE / PBEsol viscosity", fontweight="bold")
    for ax in axes:
        ax.tick_params(direction="in")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def compute_delta_rows(agg: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[float, float, str], Dict[str, object]] = {}
    for r in agg:
        by_key[(float(r["T_K"]), float(r["Cu_atpct"]), str(r["functional"]))] = r
    out = []
    for T, Cu, func in sorted(by_key):
        if func != "PBEsol":
            continue
        ps = by_key.get((T, Cu, "PBEsol"))
        pe = by_key.get((T, Cu, "PBE"))
        if not ps or not pe:
            continue
        delta = float(ps["eta_mean_Pa_s"]) - float(pe["eta_mean_Pa_s"])
        sem_ps = float(ps["eta_plot_sem_Pa_s"])
        sem_pe = float(pe["eta_plot_sem_Pa_s"])
        delta_sem = math.sqrt((sem_ps if math.isfinite(sem_ps) else 0.0) ** 2 + (sem_pe if math.isfinite(sem_pe) else 0.0) ** 2)
        ratio = float(pe["eta_mean_Pa_s"]) / float(ps["eta_mean_Pa_s"]) if float(ps["eta_mean_Pa_s"]) != 0 else float("nan")
        out.append({
            "T_K": T,
            "Cu_atpct": Cu,
            "eta_PBEsol_Pa_s": float(ps["eta_mean_Pa_s"]),
            "eta_PBE_Pa_s": float(pe["eta_mean_Pa_s"]),
            "delta_PBEsol_minus_PBE_Pa_s": delta,
            "delta_PBEsol_minus_PBE_mPa_s": delta * 1000.0,
            "delta_sem_Pa_s": delta_sem,
            "delta_sem_mPa_s": delta_sem * 1000.0,
            "ratio_PBE_over_PBEsol": ratio,
            "n_PBEsol": int(ps["n_runs_ok"]),
            "n_PBE": int(pe["n_runs_ok"]),
        })
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    keep_T = parse_float_list(args.temperatures)
    keep_Cu = parse_float_list(args.compositions)
    include_singletons = args.include_singletons and not args.exclude_singletons

    roots = [(args.pbe_label, Path(args.pbe_root).resolve()), (args.pbesol_label, Path(args.pbesol_root).resolve())]
    all_raw: List[Dict[str, object]] = []
    discovery = {}
    for label, root in roots:
        files = find_summary_files(
            root=root,
            summary_name=args.summary_name,
            allow_direct=args.allow_direct_state_files,
            keep_direct_if_reps=args.keep_direct_if_reps,
            keep_T=keep_T,
            keep_Cu=keep_Cu,
        )
        discovery[label] = {"root": str(root), "n_files": len(files)}
        rows = build_rows_for_functional(
            label=label,
            root=root,
            files=files,
            eta_column=args.eta_column,
            sem_column=args.sem_column,
            min_eta=args.min_eta,
            max_eta=args.max_eta,
            max_component_cv=args.max_component_cv,
        )
        all_raw.extend(rows)

    if not all_raw:
        raise SystemExit("No viscosity_summary.csv files were found. Check roots or use --allow-direct-state-files.")

    agg = aggregate_rows(all_raw, include_singletons=include_singletons)
    invalid = [r for r in all_raw if r.get("status") != "ok"]
    delta_rows = compute_delta_rows(agg)

    exp_rows: List[Dict[str, object]] = []
    if not args.no_built_in_exp:
        exp_rows.extend(builtin_exp_rows())
    if args.exp_csv:
        exp_rows.extend(load_exp_csv(Path(args.exp_csv)))

    write_csv(outdir / "viscosity_raw.csv", all_raw)
    write_csv(outdir / "viscosity_invalid_or_filtered.csv", invalid)
    write_csv(outdir / "viscosity_aggregate.csv", agg)
    write_csv(outdir / "viscosity_PBEsol_minus_PBE.csv", delta_rows)
    if exp_rows:
        write_csv(outdir / "viscosity_reference_data.csv", exp_rows)

    plot_composition(agg, all_raw, exp_rows, outdir / "viscosity_vs_composition_PBE_PBEsol.png", args.units, args.show_replicates, args.dpi)
    plot_temperature(agg, outdir / "viscosity_vs_temperature_PBE_PBEsol.png", args.units, args.dpi)
    plot_delta_ratio(delta_rows, outdir / "viscosity_delta_ratio_PBE_PBEsol.png", args.units, args.dpi)

    summary = {
        "inputs": discovery,
        "n_raw_rows": len(all_raw),
        "n_ok_raw_rows": sum(1 for r in all_raw if r.get("status") == "ok"),
        "n_invalid_or_filtered_rows": len(invalid),
        "n_aggregate_rows": len(agg),
        "n_delta_rows": len(delta_rows),
        "status_counts": dict(sorted({str(s): sum(1 for r in all_raw if r.get("status") == s) for s in set(r.get("status") for r in all_raw)}.items())),
        "filters": {
            "temperatures": sorted(keep_T) if keep_T else None,
            "compositions": sorted(keep_Cu) if keep_Cu else None,
            "min_eta_Pa_s": args.min_eta,
            "max_eta_Pa_s": args.max_eta if math.isfinite(args.max_eta) else None,
            "max_component_cv": args.max_component_cv if math.isfinite(args.max_component_cv) else None,
            "include_singletons": include_singletons,
            "allow_direct_state_files": args.allow_direct_state_files,
            "drop_direct_if_reps": not args.keep_direct_if_reps,
        },
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
