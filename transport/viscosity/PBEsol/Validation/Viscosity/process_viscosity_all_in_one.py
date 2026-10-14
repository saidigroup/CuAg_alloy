#!/usr/bin/env python3
"""All-in-one Cu-Ag PBE/PBEsol Green-Kubo viscosity workflow.

This script does both steps:
  1) scans PBE/PBEsol Viscosity roots for S0St.dat and creates viscosity_summary.csv
     using a robust parser for LAMMPS fix ave/correlate block output;
  2) aggregates all viscosity_summary.csv files, reports failed/filtered jobs, and
     plots PBE vs PBEsol comparisons.

Expected layout:
  ROOT/T1423/Cu_80/rep01/S0St.dat
  ROOT/T1423/Cu_80/rep01/viscosity_out        # preferred for T and volume
  ROOT/T1423/Cu_80/rep01/in.lmp or final*.lmp # fallback for volume

Direct state files ROOT/T1423/Cu_80/S0St.dat can be used with --allow-direct-state-files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------- constants / regex ----------
KB_SI = 1.380649e-23
BAR_TO_PA = 1.0e5
PS_TO_S = 1.0e-12
A3_TO_M3 = 1.0e-30
CONVERT = BAR_TO_PA ** 2 * PS_TO_S * A3_TO_M3

T_DIR_RE = re.compile(r"^T(?P<T>\d+(?:\.\d+)?)$")
CU_DIR_RE = re.compile(r"^Cu_(?P<Cu>\d+(?:\.\d+)?)$")
REP_DIR_RE = re.compile(r"^rep(?P<rep>\d+)$")

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

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-process and plot PBE/PBEsol Cu-Ag viscosity data in one step.")
    p.add_argument("--pbe-root", default="/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au/Validation/Viscosity")
    p.add_argument("--pbesol-root", default="/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au_sol/Validation/Viscosity")
    p.add_argument("--pbe-label", default="PBE")
    p.add_argument("--pbesol-label", default="PBEsol")
    p.add_argument("--outdir", default="viscosity_pbe_pbesol_processed")
    p.add_argument("--acf-name", default="S0St.dat")
    p.add_argument("--summary-name", default="viscosity_summary.csv")
    p.add_argument("--dt-ps", type=float, default=0.001)
    p.add_argument("--tail-frac", type=float, default=0.20)
    p.add_argument("--temperatures", default="", help="Optional comma-separated temperatures to keep")
    p.add_argument("--compositions", default="", help="Optional comma-separated Cu at.% values to keep")
    p.add_argument("--allow-direct-state-files", action="store_true", help="Also process ROOT/T*/Cu_*/S0St.dat outside rep folders")
    p.add_argument("--keep-direct-if-reps", action="store_true", help="Keep direct T*/Cu_* summaries even when rep*/ summaries exist")
    p.add_argument("--overwrite-summaries", action="store_true", help="Regenerate existing viscosity_summary.csv files")
    p.add_argument("--skip-postprocess", action="store_true", help="Do not regenerate summaries; only aggregate existing viscosity_summary.csv files")
    p.add_argument("--include-singletons", action="store_true", default=True)
    p.add_argument("--exclude-singletons", action="store_true")
    p.add_argument("--min-eta", type=float, default=0.0)
    p.add_argument("--max-eta", type=float, default=float("nan"))
    p.add_argument("--max-component-cv", type=float, default=float("nan"))
    p.add_argument("--units", choices=["mPa_s", "Pa_s"], default="mPa_s")
    p.add_argument("--no-built-in-exp", action="store_true")
    p.add_argument("--exp-csv", default=None)
    p.add_argument("--show-replicates", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()

# ---------- utilities ----------
def safe_float(x) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def parse_float_list(s: str) -> Optional[set[float]]:
    if not s:
        return None
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.add(float(tok))
    return out


def near_in_set(x: float, vals: Optional[set[float]], tol: float = 1e-6) -> bool:
    if vals is None:
        return True
    return any(abs(float(x) - float(v)) <= tol for v in vals)


def extract_state(path: Path) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    T = None; Cu = None; rep = None
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


def find_state_files(root: Path, filename: str, allow_direct: bool, keep_direct_if_reps: bool, keep_T, keep_Cu) -> List[Path]:
    files = []
    for p in root.rglob(filename):
        if not p.is_file():
            continue
        T, Cu, rep = extract_state(p)
        if T is None or Cu is None:
            continue
        if not near_in_set(T, keep_T) or not near_in_set(Cu, keep_Cu):
            continue
        if rep is not None or allow_direct:
            files.append(p)
    if keep_direct_if_reps:
        return sorted(files)
    has_rep = defaultdict(bool)
    for p in files:
        T, Cu, rep = extract_state(p)
        if rep is not None:
            has_rep[(T, Cu)] = True
    out = []
    for p in files:
        T, Cu, rep = extract_state(p)
        if rep is None and has_rep[(T, Cu)]:
            continue
        out.append(p)
    return sorted(out)


def parse_volume_from_lammps_data(path: Path) -> Optional[float]:
    if not path.exists() or not path.is_file():
        return None
    xlo=xhi=ylo=yhi=zlo=zhi=None
    try:
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
                xlo=float(parts[0]); xhi=float(parts[1])
            elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
                ylo=float(parts[0]); yhi=float(parts[1])
            elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
                zlo=float(parts[0]); zhi=float(parts[1])
            if all(v is not None for v in [xlo,xhi,ylo,yhi,zlo,zhi]):
                return (xhi-xlo)*(yhi-ylo)*(zhi-zlo)
    except Exception:
        return None
    return None


def find_log(run_dir: Path) -> Optional[Path]:
    for name in ["viscosity_out", "log.lammps", "viscosity.log", "log"]:
        p = run_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def find_volume(run_dir: Path) -> Optional[float]:
    for name in ["final_viscosity.lmp", "final.lmp", "in.lmp", "final_source.lmp"]:
        v = parse_volume_from_lammps_data(run_dir / name)
        if v is not None and v > 0:
            return v
    return None


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader(); wr.writerows(rows)

# ---------- per-run postprocessing ----------
def parse_lammps_log(log_path: Optional[Path]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if log_path is None or not log_path.exists():
        return out
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"GK_VISCOSITY\s+([0-9eE+\-.]+)\s+Pa\*s\s+at\s+T=([0-9eE+\-.]+)\s+K", text)
    if m:
        out["eta_lammps_Pa_s"] = float(m.group(1)); out["T_K"] = float(m.group(2))
    lines = text.splitlines()
    header_idx = None; headers = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*Step\s+", line):
            header_idx = i; headers = line.split()
    if header_idx is not None and headers is not None:
        data_line = None
        for line in lines[header_idx+1:]:
            s = line.strip()
            if not s: continue
            if s.startswith("Loop time"): break
            if re.match(r"^[0-9eE+\-.\s]+$", s):
                vals = s.split()
                if len(vals) == len(headers):
                    data_line = vals
        if data_line is not None:
            for h, v in zip(headers, data_line):
                try: out[h] = float(v)
                except Exception: pass
    if "Temp" in out and "T_K" not in out:
        out["T_K"] = out["Temp"]
    if "vol" in out:
        out["volume_A3"] = out["vol"]
    elif "Volume" in out:
        out["volume_A3"] = out["Volume"]
    return out


def read_last_acf_block(acf_path: Path) -> Dict[str, object]:
    lines = acf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks: List[Tuple[float, int, int]] = []  # timestep, nrows, header line index
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) == 2:
            try:
                timestep = float(parts[0]); nrows = int(float(parts[1]))
                # next row should look like index time_delta ncount ...
                if nrows > 0 and i + 1 < len(lines) and len(lines[i+1].split()) >= 6:
                    blocks.append((timestep, nrows, i))
            except Exception:
                pass
    if not blocks:
        raise ValueError(f"No fix ave/correlate blocks found in {acf_path}")
    timestep, nrows, idx0 = blocks[-1]
    rows = []
    for j in range(idx0+1, min(idx0+1+nrows, len(lines))):
        parts = lines[j].split()
        if len(parts) < 6:
            continue
        try:
            rows.append([float(x) for x in parts[:6]])
        except Exception:
            continue
    if not rows:
        raise ValueError(f"No six-column ACF rows in final block of {acf_path}")
    arr = np.asarray(rows, dtype=float)
    return {
        "n_blocks": len(blocks),
        "block_timestep": timestep,
        "idx": arr[:,0],
        "time_delta_steps": arr[:,1],
        "ncount": arr[:,2],
        "c_xy": arr[:,3],
        "c_xz": arr[:,4],
        "c_yz": arr[:,5],
    }


def cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    if len(y) < 2:
        return out
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out


def estimate_plateau(y: np.ndarray, frac_tail: float) -> Tuple[float, float, float, int]:
    n = len(y); n_tail = max(5, int(math.ceil(frac_tail * n)))
    tail = np.asarray(y[-n_tail:], dtype=float)
    mean = float(np.mean(tail))
    sd = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
    sem = sd / math.sqrt(len(tail)) if len(tail) > 1 else 0.0
    return mean, sd, sem, n_tail


def write_run_summary(run_dir: Path, acf_path: Path, summary_name: str, dt_ps: float, tail_frac: float, T_path: Optional[float], dry_run: bool) -> Dict[str, object]:
    summary_path = run_dir / summary_name
    row: Dict[str, object] = {"run_dir": str(run_dir), "acf": str(acf_path), "summary": str(summary_path)}
    if dry_run:
        row.update({"status": "dry_run", "reason": "not postprocessed"}); return row
    log_path = find_log(run_dir)
    meta = parse_lammps_log(log_path)
    T = meta.get("T_K", T_path if T_path is not None else float("nan"))
    V = meta.get("volume_A3", float("nan"))
    if not math.isfinite(V):
        V0 = find_volume(run_dir)
        if V0 is not None:
            V = V0; row["volume_source"] = "LAMMPS data box"
        else:
            row["volume_source"] = "missing"
    else:
        row["volume_source"] = str(log_path) if log_path else "log"
    row["log"] = str(log_path) if log_path else ""
    if not math.isfinite(T):
        row.update({"status": "failed", "reason": "temperature not found in log/path"}); return row
    if not math.isfinite(V):
        row.update({"status": "failed", "reason": "volume not found in log or LAMMPS data file"}); return row
    try:
        acf = read_last_acf_block(acf_path)
        t_ps = np.asarray(acf["time_delta_steps"], dtype=float) * dt_ps
        pref = CONVERT * V / (KB_SI * T)
        eta_xy_curve = pref * cumulative_trapz(np.asarray(acf["c_xy"], dtype=float), t_ps)
        eta_xz_curve = pref * cumulative_trapz(np.asarray(acf["c_xz"], dtype=float), t_ps)
        eta_yz_curve = pref * cumulative_trapz(np.asarray(acf["c_yz"], dtype=float), t_ps)
        eta_avg_curve = (eta_xy_curve + eta_xz_curve + eta_yz_curve) / 3.0
        eta_xy, eta_xy_sd, eta_xy_sem, n_tail = estimate_plateau(eta_xy_curve, tail_frac)
        eta_xz, eta_xz_sd, eta_xz_sem, _ = estimate_plateau(eta_xz_curve, tail_frac)
        eta_yz, eta_yz_sd, eta_yz_sem, _ = estimate_plateau(eta_yz_curve, tail_frac)
        eta_avg, eta_avg_sd, eta_avg_sem, _ = estimate_plateau(eta_avg_curve, tail_frac)
        out = {
            "directory": str(run_dir), "temperature_K": T, "volume_A3": V,
            "eta_xy_Pa_s": eta_xy, "eta_xz_Pa_s": eta_xz, "eta_yz_Pa_s": eta_yz,
            "eta_avg_Pa_s": eta_avg, "eta_avg_tail_sd_Pa_s": eta_avg_sd,
            "eta_avg_tail_sem_Pa_s": eta_avg_sem,
            "eta_lammps_Pa_s": meta.get("eta_lammps_Pa_s", float("nan")),
            "corr_time_max_ps": float(t_ps[-1]) if len(t_ps) else float("nan"),
            "n_acf_blocks": int(acf["n_blocks"]), "acf_block_timestep": float(acf["block_timestep"]),
            "acf_n_rows": int(len(t_ps)), "acf_ncount_first": float(acf["ncount"][0]),
            "acf_ncount_last": float(acf["ncount"][-1]), "tail_frac": tail_frac,
            "tail_points": int(n_tail), "dt_ps": dt_ps,
        }
        write_csv(summary_path, [out])
        curve_path = run_dir / "viscosity_integral.csv"
        with curve_path.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["time_ps","eta_xy_Pa_s","eta_xz_Pa_s","eta_yz_Pa_s","eta_avg_Pa_s","ncount"])
            for vals in zip(t_ps, eta_xy_curve, eta_xz_curve, eta_yz_curve, eta_avg_curve, acf["ncount"]):
                wr.writerow(vals)
        plot_path = run_dir / "viscosity_integral.png"
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(t_ps, eta_xy_curve*1000, label="xy")
        ax.plot(t_ps, eta_xz_curve*1000, label="xz")
        ax.plot(t_ps, eta_yz_curve*1000, label="yz")
        ax.plot(t_ps, eta_avg_curve*1000, label="avg", lw=2)
        ax.axhline(eta_avg*1000, ls="--", lw=1)
        ax.set_xlabel("Correlation time (ps)"); ax.set_ylabel(r"Cumulative viscosity (mPa$\cdot$s)")
        ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(plot_path, dpi=200); plt.close(fig)
        row.update({"status": "ok", "reason": "summary written", "temperature_K": T, "volume_A3": V, "eta_avg_Pa_s": eta_avg, "n_acf_blocks": int(acf["n_blocks"]), "acf_block_timestep": float(acf["block_timestep"])})
    except Exception as exc:
        row.update({"status": "failed", "reason": repr(exc)})
    return row

# ---------- aggregate/plot ----------
def read_first_row(path: Path) -> Dict[str, str]:
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.DictReader(fh))
    if not rows: raise ValueError(f"empty CSV: {path}")
    return rows[0]


def component_cv(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if arr.size < 2: return float("nan")
    mean_abs = float(np.mean(np.abs(arr)))
    if mean_abs == 0: return float("inf")
    return float(np.std(arr, ddof=1) / mean_abs)


def status_for_eta(eta: float, cv: float, min_eta: float, max_eta: float, max_cv: float) -> str:
    if not math.isfinite(eta): return "nonfinite_eta"
    if eta <= min_eta: return "below_min_eta"
    if math.isfinite(max_eta) and eta > max_eta: return "above_max_eta"
    if math.isfinite(max_cv) and math.isfinite(cv) and cv > max_cv: return "high_component_cv"
    return "ok"


def build_raw_rows(label: str, root: Path, summary_files: Sequence[Path], min_eta: float, max_eta: float, max_cv: float) -> List[Dict[str, object]]:
    rows = []
    auto = defaultdict(int)
    for p in summary_files:
        T, Cu, rep = extract_state(p)
        if rep is None:
            auto[(T,Cu)] += 1; rep_val = auto[(T,Cu)]; run_label = "main"
        else:
            rep_val = rep; run_label = f"rep{rep:02d}"
        try:
            r = read_first_row(p)
            eta = safe_float(r.get("eta_avg_Pa_s")); run_sem = safe_float(r.get("eta_avg_tail_sem_Pa_s"))
            eta_xy = safe_float(r.get("eta_xy_Pa_s")); eta_xz = safe_float(r.get("eta_xz_Pa_s")); eta_yz = safe_float(r.get("eta_yz_Pa_s"))
            cv = component_cv([eta_xy, eta_xz, eta_yz])
            status = status_for_eta(eta, cv, min_eta, max_eta, max_cv)
            rows.append({"functional": label, "T_K": T, "Cu_atpct": Cu, "replicate": rep_val, "run_label": run_label,
                         "eta_Pa_s": eta, "eta_mPa_s": eta*1000 if math.isfinite(eta) else float("nan"),
                         "eta_run_sem_Pa_s": run_sem, "eta_xy_Pa_s": eta_xy, "eta_xz_Pa_s": eta_xz, "eta_yz_Pa_s": eta_yz,
                         "eta_lammps_Pa_s": safe_float(r.get("eta_lammps_Pa_s")), "volume_A3": safe_float(r.get("volume_A3")),
                         "corr_time_max_ps": safe_float(r.get("corr_time_max_ps")), "component_cv": cv, "status": status,
                         "folder": str(p.parent), "summary_file": str(p)})
        except Exception as exc:
            rows.append({"functional": label, "T_K": T, "Cu_atpct": Cu, "replicate": rep or 0, "run_label": run_label if 'run_label' in locals() else "?", "eta_Pa_s": float("nan"), "eta_mPa_s": float("nan"), "status": f"read_error:{exc}", "folder": str(p.parent), "summary_file": str(p)})
    return sorted(rows, key=lambda x: (str(x["functional"]), float(x["T_K"] or -1), float(x["Cu_atpct"] or -1), int(x["replicate"])))


def sem(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if arr.size <= 1: return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def aggregate(raw: Sequence[Dict[str, object]], include_singletons: bool) -> List[Dict[str, object]]:
    groups = defaultdict(list)
    for r in raw:
        if r.get("status") == "ok":
            groups[(str(r["functional"]), float(r["T_K"]), float(r["Cu_atpct"]))].append(r)
    out = []
    for (func,T,Cu), g in sorted(groups.items()):
        vals = [float(r["eta_Pa_s"]) for r in g if math.isfinite(float(r["eta_Pa_s"]))]
        if not vals: continue
        if len(vals) == 1 and not include_singletons: continue
        rep_sem = sem(vals)
        run_sems = [float(r["eta_run_sem_Pa_s"]) for r in g if "eta_run_sem_Pa_s" in r and math.isfinite(float(r["eta_run_sem_Pa_s"]))]
        plot_sem = rep_sem if math.isfinite(rep_sem) else (float(np.mean(run_sems)) if run_sems else 0.0)
        cvs = [float(r["component_cv"]) for r in g if "component_cv" in r and math.isfinite(float(r["component_cv"]))]
        out.append({"functional": func, "T_K": T, "Cu_atpct": Cu, "n_runs_ok": len(vals), "eta_mean_Pa_s": float(np.mean(vals)), "eta_mean_mPa_s": float(np.mean(vals))*1000, "eta_sem_Pa_s": rep_sem, "eta_sem_mPa_s": rep_sem*1000 if math.isfinite(rep_sem) else float("nan"), "eta_plot_sem_Pa_s": plot_sem, "eta_plot_sem_mPa_s": plot_sem*1000, "eta_min_Pa_s": float(np.min(vals)), "eta_max_Pa_s": float(np.max(vals)), "component_cv_mean": float(np.mean(cvs)) if cvs else float("nan"), "run_labels": ";".join(str(r["run_label"]) for r in g)})
    return out


def builtin_exp_rows() -> List[Dict[str, object]]:
    return [{"label":"Brillo2018", "T_K":1375.0, "Cu_atpct":Cu, "eta_Pa_s":eta, "eta_mPa_s":eta*1000, "eta_sem_Pa_s":sem, "eta_sem_mPa_s":sem*1000} for Cu,(eta,sem) in BUILTIN_BRILLO_2018_1375K.items()]


def load_exp_csv(path: Path) -> List[Dict[str, object]]:
    out=[]
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            T=safe_float(r.get("T_K", r.get("temperature_K", r.get("Temperature_K"))))
            Cu=safe_float(r.get("Cu_atpct", r.get("Cu_pct", r.get("Cu composition"))))
            eta=safe_float(r.get("eta_Pa_s", r.get("viscosity_Pa_s", r.get("eta"))))
            es=safe_float(r.get("eta_sem_Pa_s", r.get("viscosity_sem_Pa_s")))
            if math.isfinite(T) and math.isfinite(Cu) and math.isfinite(eta):
                out.append({"label":r.get("label") or r.get("source") or "Experiment", "T_K":T, "Cu_atpct":Cu, "eta_Pa_s":eta, "eta_mPa_s":eta*1000, "eta_sem_Pa_s":es, "eta_sem_mPa_s":es*1000 if math.isfinite(es) else float("nan")})
    return out


def ycols(units: str):
    return ("eta_mean_Pa_s", "eta_plot_sem_Pa_s", "eta_Pa_s", "eta_sem_Pa_s") if units == "Pa_s" else ("eta_mean_mPa_s", "eta_plot_sem_mPa_s", "eta_mPa_s", "eta_sem_mPa_s")


def ylabel(units: str) -> str:
    return r"Viscosity, $\eta$ (Pa$\cdot$s)" if units == "Pa_s" else r"Viscosity, $\eta$ (mPa$\cdot$s)"


def compute_delta(agg_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    d = {(float(r["T_K"]), float(r["Cu_atpct"]), str(r["functional"])): r for r in agg_rows}
    out=[]
    for (T,Cu,func), ps in sorted(d.items()):
        if func != "PBEsol": continue
        pe=d.get((T,Cu,"PBE"))
        if not pe: continue
        delta=float(ps["eta_mean_Pa_s"])-float(pe["eta_mean_Pa_s"])
        ds=math.sqrt(float(ps["eta_plot_sem_Pa_s"])**2 + float(pe["eta_plot_sem_Pa_s"])**2)
        ratio=float(pe["eta_mean_Pa_s"])/float(ps["eta_mean_Pa_s"]) if float(ps["eta_mean_Pa_s"]) != 0 else float("nan")
        out.append({"T_K":T,"Cu_atpct":Cu,"eta_PBEsol_Pa_s":float(ps["eta_mean_Pa_s"]),"eta_PBE_Pa_s":float(pe["eta_mean_Pa_s"]),"delta_PBEsol_minus_PBE_Pa_s":delta,"delta_PBEsol_minus_PBE_mPa_s":delta*1000,"delta_sem_Pa_s":ds,"delta_sem_mPa_s":ds*1000,"ratio_PBE_over_PBEsol":ratio,"n_PBEsol":int(ps["n_runs_ok"]),"n_PBE":int(pe["n_runs_ok"])})
    return out


def plot_composition(agg_rows, raw_rows, exp_rows, path: Path, units: str, show_reps: bool, dpi: int):
    if not agg_rows: return
    ymean,ysem,yexp,yexpsem=ycols(units)
    fig, ax = plt.subplots(figsize=(9,6))
    funcs=sorted(set(str(r["functional"]) for r in agg_rows))
    temps=sorted(set(float(r["T_K"]) for r in agg_rows))
    markers={"PBE":"D","PBEsol":"s"}
    for T in temps:
        for func in funcs:
            pts=sorted([r for r in agg_rows if str(r["functional"])==func and abs(float(r["T_K"])-T)<1e-6], key=lambda r: float(r["Cu_atpct"]))
            if pts:
                ax.errorbar([float(p["Cu_atpct"]) for p in pts], [float(p[ymean]) for p in pts], yerr=[float(p[ysem]) for p in pts], marker=markers.get(func,"o"), linestyle="--", capsize=3, label=f"{func} {T:.0f} K")
    if show_reps:
        for r in raw_rows:
            if r.get("status") == "ok": ax.scatter(float(r["Cu_atpct"]), float(r["eta_mPa_s"] if units=="mPa_s" else r["eta_Pa_s"]), s=12, alpha=0.35)
    for lab,T in sorted(set((str(e["label"]), float(e["T_K"])) for e in exp_rows)):
        pts=sorted([e for e in exp_rows if str(e["label"])==lab and abs(float(e["T_K"])-T)<1e-6], key=lambda e: float(e["Cu_atpct"]))
        ax.errorbar([float(p["Cu_atpct"]) for p in pts], [float(p[yexp]) for p in pts], yerr=[0 if not math.isfinite(float(p[yexpsem])) else float(p[yexpsem]) for p in pts], marker="o", linestyle=":", capsize=3, label=f"{lab} {T:.0f} K")
    ax.set_xlabel("Cu composition (at. %)", fontweight="bold"); ax.set_ylabel(ylabel(units), fontweight="bold")
    ax.tick_params(direction="in"); ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)


def plot_temperature(agg_rows, path: Path, units: str, dpi: int):
    if not agg_rows: return
    ymean,ysem,_,_=ycols(units)
    funcs=sorted(set(str(r["functional"]) for r in agg_rows))
    fig, axes=plt.subplots(1,len(funcs),figsize=(6*len(funcs),5.2),sharey=True)
    if len(funcs)==1: axes=[axes]
    comps=sorted(set(float(r["Cu_atpct"]) for r in agg_rows))
    for ax,func in zip(axes, funcs):
        for Cu in comps:
            pts=sorted([r for r in agg_rows if str(r["functional"])==func and abs(float(r["Cu_atpct"])-Cu)<1e-6], key=lambda r: float(r["T_K"]))
            if pts:
                ax.errorbar([float(p["T_K"]) for p in pts], [float(p[ymean]) for p in pts], yerr=[float(p[ysem]) for p in pts], marker="o", capsize=3, label=f"Cu {Cu:.0f}%")
        ax.set_title(func); ax.set_xlabel("Temperature (K)", fontweight="bold"); ax.tick_params(direction="in"); ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel(units), fontweight="bold"); axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02,1), loc="upper left")
    fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def plot_delta(delta_rows, path: Path, units: str, dpi: int):
    if not delta_rows: return
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    dy="delta_PBEsol_minus_PBE_mPa_s" if units=="mPa_s" else "delta_PBEsol_minus_PBE_Pa_s"
    dsem="delta_sem_mPa_s" if units=="mPa_s" else "delta_sem_Pa_s"
    for T in sorted(set(float(r["T_K"]) for r in delta_rows)):
        pts=sorted([r for r in delta_rows if abs(float(r["T_K"])-T)<1e-6], key=lambda r: float(r["Cu_atpct"]))
        axes[0].errorbar([float(p["Cu_atpct"]) for p in pts], [float(p[dy]) for p in pts], yerr=[float(p[dsem]) for p in pts], marker="o", capsize=3, label=f"{T:.0f} K")
        axes[1].plot([float(p["Cu_atpct"]) for p in pts], [float(p["ratio_PBE_over_PBEsol"]) for p in pts], marker="s", linestyle="--", label=f"{T:.0f} K")
    axes[0].axhline(0,lw=1); axes[1].axhline(1,lw=1)
    axes[0].set_xlabel("Cu composition (at. %)", fontweight="bold"); axes[1].set_xlabel("Cu composition (at. %)", fontweight="bold")
    axes[0].set_ylabel("PBEsol - PBE " + ylabel(units).replace("Viscosity, ",""), fontweight="bold"); axes[1].set_ylabel("PBE / PBEsol viscosity", fontweight="bold")
    for ax in axes: ax.tick_params(direction="in"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)

# ---------- main ----------
def main():
    args=parse_args(); outdir=Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    keep_T=parse_float_list(args.temperatures); keep_Cu=parse_float_list(args.compositions)
    include_singletons=args.include_singletons and not args.exclude_singletons
    roots=[(args.pbe_label,Path(args.pbe_root).resolve()),(args.pbesol_label,Path(args.pbesol_root).resolve())]

    post_rows=[]
    if not args.skip_postprocess:
        for label,root in roots:
            acfs=find_state_files(root,args.acf_name,args.allow_direct_state_files,args.keep_direct_if_reps,keep_T,keep_Cu)
            for acf in acfs:
                T,Cu,rep=extract_state(acf); run_dir=acf.parent; summary=run_dir/args.summary_name
                base={"functional":label,"T_K":T,"Cu_atpct":Cu,"run_label":"main" if rep is None else f"rep{rep:02d}","run_dir":str(run_dir),"acf":str(acf),"summary":str(summary)}
                if summary.exists() and not args.overwrite_summaries:
                    base.update({"status":"skipped_existing","reason":"summary exists; use --overwrite-summaries to regenerate"}); post_rows.append(base); continue
                res=write_run_summary(run_dir,acf,args.summary_name,args.dt_ps,args.tail_frac,T,args.dry_run)
                base.update(res); post_rows.append(base)
        write_csv(outdir/"viscosity_postprocess_report.csv", post_rows)

    raw_rows=[]; discovery={}
    for label,root in roots:
        sums=find_state_files(root,args.summary_name,args.allow_direct_state_files,args.keep_direct_if_reps,keep_T,keep_Cu)
        discovery[label]={"root":str(root),"n_summary_files":len(sums)}
        raw_rows.extend(build_raw_rows(label,root,sums,args.min_eta,args.max_eta,args.max_component_cv))
    agg_rows=aggregate(raw_rows,include_singletons)
    invalid=[r for r in raw_rows if r.get("status")!="ok"]
    delta_rows=compute_delta(agg_rows)
    exp=[]
    if not args.no_built_in_exp: exp.extend(builtin_exp_rows())
    if args.exp_csv: exp.extend(load_exp_csv(Path(args.exp_csv)))

    write_csv(outdir/"viscosity_raw.csv", raw_rows)
    write_csv(outdir/"viscosity_invalid_or_filtered.csv", invalid)
    write_csv(outdir/"viscosity_aggregate.csv", agg_rows)
    write_csv(outdir/"viscosity_PBEsol_minus_PBE.csv", delta_rows)
    if exp: write_csv(outdir/"viscosity_reference_data.csv", exp)
    plot_composition(agg_rows,raw_rows,exp,outdir/"viscosity_vs_composition_PBE_PBEsol.png",args.units,args.show_replicates,args.dpi)
    plot_temperature(agg_rows,outdir/"viscosity_vs_temperature_PBE_PBEsol.png",args.units,args.dpi)
    plot_delta(delta_rows,outdir/"viscosity_delta_ratio_PBE_PBEsol.png",args.units,args.dpi)

    summary={"postprocess_status_counts":{},"aggregate_status_counts":{},"inputs":discovery,"n_raw_rows":len(raw_rows),"n_ok_raw_rows":sum(1 for r in raw_rows if r.get("status")=="ok"),"n_invalid_or_filtered_rows":len(invalid),"n_aggregate_rows":len(agg_rows),"n_delta_rows":len(delta_rows),"filters":{"temperatures":sorted(keep_T) if keep_T else None,"compositions":sorted(keep_Cu) if keep_Cu else None,"min_eta_Pa_s":args.min_eta,"max_eta_Pa_s":args.max_eta if math.isfinite(args.max_eta) else None,"max_component_cv":args.max_component_cv if math.isfinite(args.max_component_cv) else None,"allow_direct_state_files":args.allow_direct_state_files,"drop_direct_if_reps":not args.keep_direct_if_reps,"include_singletons":include_singletons}}
    for r in post_rows: summary["postprocess_status_counts"][str(r.get("status"))]=summary["postprocess_status_counts"].get(str(r.get("status")),0)+1
    for r in raw_rows: summary["aggregate_status_counts"][str(r.get("status"))]=summary["aggregate_status_counts"].get(str(r.get("status")),0)+1
    with (outdir/"summary.json").open("w",encoding="utf-8") as fh: json.dump(summary,fh,indent=2)
    print(json.dumps(summary,indent=2)); print(f"\nWrote outputs to: {outdir}")

if __name__=="__main__": main()
