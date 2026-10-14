#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dp_CuAg_only.py

Filter + export DeepMD datasets from VASP runs, but ONLY for:
  - Cu only
  - Ag only
  - Cu+Ag

Any run containing elements outside {Cu, Ag} is skipped entirely.

Defaults mirror dp_UOx_v10_full.py:
  --glob                          ./**/vasprun.xml
  --base-out                      ../clean_data/dpdata_CuAg_i1
  --max-set-size                  1000
  --elec-filter                   vasprun_oszicar
  --strict-elec                   keep_if_unknown

Filtering:
  - electronic prefilter (vasprun/oszicar)
  - force atom-excess
  - energy excess
  - force k-sigma
  - energy k-sigma
  - combined second pass (force excess, energy floor)
  - optional combined k-sigma filters

Exports:
  base_out/<mp_id>/<composition>/<bulk|surface>/<tempblock>/
  base_out/validation/<mp_id>/<composition>/<bulk|surface>/<tempblock>/

If --rmv-dupli is enabled, duplicate frames are excluded from both training
and validation. The validation set is randomly sampled from unique surviving
frames only, using --val-fraction.

Composition folder is in priority order: Cu then Ag, e.g.:
  Cu108Ag0   (Cu-only)
  Cu0Ag108   (Ag-only)
  Cu54Ag54   (binary)
"""

import os
import re
import csv
import argparse
import tempfile
from pathlib import Path
from glob import glob
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from dpdata import LabeledSystem


# ---------------- Defaults ----------------
DEFAULT_MAX_SET_SIZE = 1000
DEFAULT_STRICT_ELEC = "keep_if_unknown"

DEFAULT_ATOM_EXCESS_THRESHOLD = 12.0
DEFAULT_ENERGY_EXCESS_THRESHOLD = 2.0

DEFAULT_K_SIGMA_FRAMES = 2.3
DEFAULT_ENERGY_K_SIGMA_FRAMES = 2.3
DEFAULT_SECOND_PASS_EXCESS_FLOOR = 7.0
DEFAULT_ENERGY_FLOOR_EV_PER_ATOM = 0.5

MIN_FRAMES_TO_FILTER = 5
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_VAL_DIRNAME = "validation"

# Hard gate for this script:
ALLOWED_ELEMENTS = {"Cu", "Ag"}
priority_elements = ["Cu", "Ag"]  # fixed DeepMD type_map order


# ---------------- Basic helpers ----------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip("/"))

def make_repaired_vasprun(orig_path: str) -> Optional[str]:
    """Minimal repair of truncated vasprun.xml (same as UOx script)."""
    try:
        with open(orig_path, "rb") as f:
            b = f.read()
        if b.find(b"<modeling") == -1:
            return None
        last_calc_end = b.rfind(b"</calculation>")
        if last_calc_end == -1:
            return None
        repaired_bytes = b[: last_calc_end + len(b"</calculation>")]
        if repaired_bytes.rfind(b"</modeling>") == -1:
            repaired_bytes += b"\n</modeling>\n"
        tmpdir = tempfile.mkdtemp(prefix="repaired_vasprun_")
        repaired_path = os.path.join(tmpdir, "vasprun_repaired.xml")
        with open(repaired_path, "wb") as g:
            g.write(repaired_bytes)
        return repaired_path
    except Exception:
        return None

def load_labeled_system_any(run_dir: str) -> Optional[LabeledSystem]:
    """Prefer vasprun.xml (repair if needed). OUTCAR fallback."""
    vp = os.path.join(run_dir, "vasprun.xml")
    if os.path.exists(vp):
        try:
            return LabeledSystem(vp)
        except Exception:
            rp = make_repaired_vasprun(vp)
            if rp:
                try:
                    return LabeledSystem(rp)
                except Exception:
                    pass
    outcar = os.path.join(run_dir, "OUTCAR")
    if os.path.exists(outcar) and os.path.getsize(outcar) > 0:
        try:
            return LabeledSystem(outcar, fmt="vasp/outcar")
        except Exception:
            pass
    return None


# ---------------- Logging ----------------
def log_excluded_frames(log_dir: Path, run_dir: str, rows: List[Dict[str, Any]], header: List[str], suffix: str):
    if not rows:
        return
    _ensure_dir(log_dir)
    base = sanitize_filename(run_dir) + suffix
    fpath = log_dir / f"{base}.csv"
    write_header = not fpath.exists()
    with open(fpath, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, "") for k in header])


# ---------------- Structure & temperature ----------------
def detect_structure_category(run_dir: str) -> str:
    s = run_dir.lower()
    return "surface" if any(t in s for t in ("surf", "surface", "slab")) else "bulk"

def detect_materials_project_id(run_dir: str) -> str:
    """Return first Materials Project-style id (mp-*) found in the run path."""
    m = re.search(r"(?i)(?:^|[/_\\.-])(mp-[0-9]+)(?=$|[/_\\.-])", str(run_dir))
    if m:
        return m.group(1).lower()
    m = re.search(r"(?i)(mp-[0-9]+)", str(run_dir))
    return m.group(1).lower() if m else "NO_MP"

_TEMP_PATTERNS = [
    re.compile(r'(?:^|/)(?:T|temp)(\d{1,6})(?:K)?(?:/|$)', re.IGNORECASE),
    re.compile(r'(?:^|_)(?:T|temp)(\d{1,6})(?:K)?(?:_|$)', re.IGNORECASE),
]
def parse_temperature_K(run_dir: str) -> Optional[int]:
    s = run_dir.replace("\\", "/")
    for pat in _TEMP_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

def temp_block_label(temp_K: Optional[int], block_size: int) -> str:
    if temp_K is None:
        return "TUNKNOWN"
    lo = (temp_K // block_size) * block_size
    hi = lo + block_size
    return f"T{lo}-{hi}K"


# ---------------- Energies / natoms ----------------
def _get_energies(ls: LabeledSystem) -> Optional[np.ndarray]:
    E = ls.data.get("energies", None)
    if E is None:
        E = ls.data.get("energy", None)
    if E is None:
        return None
    return np.asarray(E).reshape(-1)

def _get_natoms(ls: LabeledSystem) -> int:
    F = ls.data.get("forces", None)
    if F is not None:
        F = np.asarray(F)
        if F.ndim == 3:
            return int(F.shape[1])
    an = ls.data.get("atom_numbs", None)
    if an is not None:
        return int(np.sum(np.asarray(an)))
    raise ValueError("Cannot infer natoms (no forces/atom_numbs).")


# ---------------- Electronic convergence (verbatim behavior) ----------------
def _parse_incar_params(incar_path: str) -> Tuple[Optional[float], Optional[int]]:
    if not os.path.isfile(incar_path):
        return None, None
    ediff = None
    nelm = None
    try:
        with open(incar_path, "r", errors="ignore") as f:
            for line in f:
                s = line.split("!")[0].strip()
                if not s or "=" not in s:
                    continue
                k, v = [x.strip() for x in s.split("=", 1)]
                k = k.upper()
                if k == "EDIFF":
                    try:
                        ediff = float(v)
                    except Exception:
                        pass
                elif k == "NELM":
                    try:
                        nelm = int(float(v))
                    except Exception:
                        pass
    except Exception:
        return None, None
    return ediff, nelm

def _parse_vasprun_params_and_flags(vasprun_path: str, debug: bool = False) -> Optional[Tuple[float, int, List[Optional[bool]]]]:
    if not os.path.isfile(vasprun_path) or os.path.getsize(vasprun_path) == 0:
        return None
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(vasprun_path)
        root = tree.getroot()
    except Exception:
        return None

    def find_param(name: str) -> Optional[str]:
        el = root.find(f".//i[@name='{name}']")
        return el.text.strip() if (el is not None and el.text) else None

    try:
        ediff = float(find_param("EDIFF"))
        nelm = int(float(find_param("NELM")))
    except Exception:
        return None

    energy_keys = ("e_0_energy", "e_wo_entrp", "e_fr_energy", "e_energy")
    flags: List[Optional[bool]] = []

    calcs = root.findall(".//calculation")
    for calc in calcs:
        scsteps = calc.findall("scstep")
        nsteps = len(scsteps)
        if nsteps == 0:
            flags.append(None)
            continue
        if nsteps >= nelm:
            flags.append(False)
            continue

        energies: List[float] = []
        for st in scsteps:
            e_node = st.find("energy")
            if e_node is None:
                continue
            val = None
            for k in energy_keys:
                iel = e_node.find(f"i[@name='{k}']")
                if iel is not None and iel.text:
                    try:
                        val = float(iel.text)
                        break
                    except Exception:
                        pass
            if val is not None:
                energies.append(val)

        if len(energies) >= 2:
            flags.append(abs(energies[-1] - energies[-2]) <= ediff)
        elif len(energies) == 1:
            flags.append(True)
        else:
            flags.append(None)

    if debug:
        nT = sum(f is True for f in flags)
        nF = sum(f is False for f in flags)
        nN = sum(f is None for f in flags)
        print(f"{Path(vasprun_path).parent}: elec-debug EDIFF={ediff} NELM={nelm} flags: True={nT} False={nF} None={nN} (nflags={len(flags)})")

    return ediff, nelm, flags

def _parse_oszicar_flags(oszicar_path: str, ediff: Optional[float], nelm: Optional[int]) -> Optional[List[Optional[bool]]]:
    if not os.path.isfile(oszicar_path) or os.path.getsize(oszicar_path) == 0:
        return None

    re_iter = re.compile(r'^\s*(?:DAV|RMM):\s*(\d+)\b', re.IGNORECASE)
    re_rms  = re.compile(r'rms\s*=\s*([0-9Ee.+-]+)', re.IGNORECASE)

    flags: List[Optional[bool]] = []
    have = False
    max_iter = 0
    last_rms = None

    def close():
        nonlocal have, max_iter, last_rms, flags
        if not have:
            return
        if ediff is not None and nelm is not None and last_rms is not None:
            try:
                flags.append((max_iter < nelm) and (float(last_rms) <= float(ediff)))
            except Exception:
                flags.append(None)
        elif nelm is not None:
            flags.append(max_iter < nelm)
        else:
            flags.append(None)
        have = False
        max_iter = 0
        last_rms = None

    try:
        with open(oszicar_path, "r", errors="ignore") as f:
            for line in f:
                if line.lstrip().startswith("T="):
                    close()
                    continue
                m = re_iter.match(line)
                if m:
                    have = True
                    it = int(m.group(1))
                    if it > max_iter:
                        max_iter = it
                m2 = re_rms.search(line)
                if m2:
                    have = True
                    try:
                        last_rms = float(m2.group(1))
                    except Exception:
                        last_rms = None
        close()
    except Exception:
        return None

    return flags if flags else None

def prefilter_nonconverged(
    ls: LabeledSystem,
    run_dir: str,
    log_dir: Path,
    elec_filter_mode: str,
    strict_elec: str,
    elec_debug: bool,
) -> Tuple[LabeledSystem, List[int]]:
    n = len(ls)
    if n == 0:
        return ls, []

    incar_ediff, incar_nelm = _parse_incar_params(os.path.join(run_dir, "INCAR"))

    flags = None
    ediff_used = None
    nelm_used = None

    if elec_filter_mode in ("vasprun_oszicar", "vasprun_only"):
        vr = _parse_vasprun_params_and_flags(os.path.join(run_dir, "vasprun.xml"), debug=elec_debug)
        if vr is not None:
            ediff_used, nelm_used, flags = vr

    if (flags is None) and (elec_filter_mode in ("vasprun_oszicar", "oszicar_only")):
        ediff_used = incar_ediff
        nelm_used = incar_nelm
        flags = _parse_oszicar_flags(os.path.join(run_dir, "OSZICAR"), ediff_used, nelm_used)

    if not flags:
        print(f"{run_dir}: elec-prefilter kept {n}/{n} frames (no usable flags)")
        rows = [dict(run_dir=run_dir, frame_index="", reason="elec_no_flags_keep")]
        log_excluded_frames(log_dir, run_dir, rows, ["run_dir","frame_index","reason"], suffix="__elec")
        return ls, []

    m = min(n, len(flags))
    drop: List[int] = []
    reasons: List[str] = []

    for i in range(m):
        fi = flags[i]
        if fi is False:
            drop.append(i); reasons.append("elec_failed")
        elif fi is None and strict_elec == "drop_if_unknown":
            drop.append(i); reasons.append("elec_unknown")

    if strict_elec == "drop_if_unknown" and m < n:
        for j in range(m, n):
            drop.append(j); reasons.append("elec_trailing_no_flag")

    if not drop:
        print(f"{run_dir}: elec-prefilter kept {n}/{n} frames (no drops)")
        return ls, []

    keep_idx = [i for i in range(n) if i not in set(drop)]

    rows = []
    for idx, reason in zip(drop, reasons):
        rows.append(dict(run_dir=run_dir, frame_index=int(idx), reason=reason))
    log_excluded_frames(log_dir, run_dir, rows, ["run_dir","frame_index","reason"], suffix="__elec")

    n_failed = sum(r == "elec_failed" for r in reasons)
    n_unk = sum(r == "elec_unknown" for r in reasons)
    n_tr = sum(r == "elec_trailing_no_flag" for r in reasons)
    print(f"{run_dir}: elec-prefilter kept {len(keep_idx)}/{n} frames (dropped {len(drop)}: {n_failed} failed, {n_unk} unknown, {n_tr} trailing)")
    return ls.sub_system(keep_idx), drop


# ---------------- Force filters ----------------
def filter_frames_atom_excess(ls: LabeledSystem, run_dir: str, threshold_excess: float, log_dir: Path) -> Tuple[LabeledSystem, List[int]]:
    n = len(ls)
    if n == 0:
        return ls, []
    F = ls.data.get("forces", None)
    if F is None:
        print(f"{run_dir}: atom-excess filter skipped (no forces)")
        return ls, []
    F = np.asarray(F)
    mags = np.linalg.norm(F, axis=2)
    avg_atom = mags.mean(axis=1)
    max_atom = mags.max(axis=1)
    cutoff = avg_atom + float(threshold_excess)
    drop_mask = max_atom > cutoff
    drop_idx = np.nonzero(drop_mask)[0].tolist()

    if not drop_idx:
        print(f"{run_dir}: atom-excess filter removed 0/{n} (threshold={threshold_excess:g} eV/Å)")
        return ls, []

    rows = []
    header = ["run_dir","frame_index","avg_atom_force_eV_per_A","max_atom_force_eV_per_A","threshold_excess_eV_per_A","cutoff_eV_per_A","reason"]
    for j in drop_idx:
        rows.append(dict(
            run_dir=run_dir, frame_index=int(j),
            avg_atom_force_eV_per_A=float(avg_atom[j]),
            max_atom_force_eV_per_A=float(max_atom[j]),
            threshold_excess_eV_per_A=float(threshold_excess),
            cutoff_eV_per_A=float(cutoff[j]),
            reason="max_atom>avg_atom+threshold",
        ))
    log_excluded_frames(log_dir, run_dir, rows, header, suffix="__atom_excess")

    keep_idx = [i for i in range(n) if i not in set(drop_idx)]
    out = ls.sub_system(keep_idx)
    print(f"{run_dir}: atom-excess filter kept {len(out)}/{n} (removed {len(drop_idx)}; threshold={threshold_excess:g} eV/Å)")
    return out, drop_idx

def filter_frames_force_ksigma(ls: LabeledSystem, run_dir: str, k_sigma_frames: float, log_dir: Path) -> Tuple[LabeledSystem, List[int]]:
    n = len(ls)
    if n < MIN_FRAMES_TO_FILTER:
        print(f"{run_dir}: skipping k-sigma filter (only {n} frames < {MIN_FRAMES_TO_FILTER})")
        return ls, []
    F = ls.data.get("forces", None)
    if F is None:
        print(f"{run_dir}: k-sigma filter skipped (no forces)")
        return ls, []
    F = np.asarray(F)
    mags = np.linalg.norm(F, axis=2)
    frame_rms = np.sqrt((mags**2).mean(axis=1))
    mu = float(frame_rms.mean())
    sigma = float(frame_rms.std())
    if not np.isfinite(sigma) or sigma == 0.0:
        print(f"{run_dir}: sigma(frame_RMS)=0/nonfinite — k-sigma filter skipped")
        return ls, []
    cutoff = mu + float(k_sigma_frames)*sigma
    drop_idx = np.nonzero(frame_rms > cutoff)[0].tolist()
    if not drop_idx:
        print(f"{run_dir}: k-sigma filter removed 0/{n}; rms_cutoff={cutoff:.4g}")
        return ls, []

    rows = []
    header = ["run_dir","frame_index","frame_rms_eV_per_A","mu_rms","sigma_rms","rms_cutoff","reason"]
    for j in drop_idx:
        rows.append(dict(
            run_dir=run_dir, frame_index=int(j),
            frame_rms_eV_per_A=float(frame_rms[j]),
            mu_rms=mu, sigma_rms=sigma, rms_cutoff=cutoff,
            reason="frame_rms>mu+Kσ",
        ))
    log_excluded_frames(log_dir, run_dir, rows, header, suffix="__ksigma_force")

    keep_idx = [i for i in range(n) if i not in set(drop_idx)]
    out = ls.sub_system(keep_idx)
    print(f"{run_dir}: k-sigma filter kept {len(out)}/{n} (removed {len(drop_idx)}); rms_cutoff={cutoff:.4g}")
    return out, drop_idx

def second_pass_excess_filter(ls: LabeledSystem, threshold_excess: float) -> Tuple[LabeledSystem, int]:
    n = len(ls)
    if n == 0:
        return ls, 0
    F = ls.data.get("forces", None)
    if F is None:
        return ls, 0
    F = np.asarray(F)
    mags = np.linalg.norm(F, axis=2)
    frame_rms = np.sqrt((mags**2).mean(axis=1))
    max_atom = mags.max(axis=1)
    excess = max_atom - frame_rms
    drop_mask = excess > float(threshold_excess)
    removed = int(np.count_nonzero(drop_mask))
    keep_idx = np.nonzero(~drop_mask)[0].tolist()
    return ls.sub_system(keep_idx), removed


# ---------------- Energy filters ----------------
def filter_frames_energy_excess(ls: LabeledSystem, run_dir: str, threshold_excess_eV_per_atom: float, log_dir: Path) -> Tuple[LabeledSystem, List[int]]:
    n = len(ls)
    if n == 0:
        return ls, []
    E = _get_energies(ls)
    if E is None:
        print(f"{run_dir}: energy-excess filter skipped (no energies)")
        return ls, []
    nat = _get_natoms(ls)
    epa = E / float(nat)
    mu = float(np.mean(epa))
    cutoff = mu + float(threshold_excess_eV_per_atom)
    drop_idx = np.nonzero(epa > cutoff)[0].tolist()

    if not drop_idx:
        print(f"{run_dir}: energy-excess filter kept {n}/{n} (removed 0); cutoff={cutoff:.4g} eV/atom")
        return ls, []

    rows = []
    header = ["run_dir","frame_index","energy_eV","energy_per_atom_eV","mu_energy_per_atom_eV","cutoff_eV_per_atom","reason"]
    for j in drop_idx:
        rows.append(dict(
            run_dir=run_dir, frame_index=int(j),
            energy_eV=float(E[j]),
            energy_per_atom_eV=float(epa[j]),
            mu_energy_per_atom_eV=mu,
            cutoff_eV_per_atom=cutoff,
            reason="E/atom>mu+threshold",
        ))
    log_excluded_frames(log_dir, run_dir, rows, header, suffix="__energy_excess")

    keep_idx = [i for i in range(n) if i not in set(drop_idx)]
    out = ls.sub_system(keep_idx)
    print(f"{run_dir}: energy-excess filter kept {len(out)}/{n} (removed {len(drop_idx)}); cutoff={cutoff:.4g} eV/atom")
    return out, drop_idx

def filter_frames_energy_ksigma(ls: LabeledSystem, run_dir: str, k_sigma_energy: float, log_dir: Path, two_sided: bool) -> Tuple[LabeledSystem, List[int]]:
    n = len(ls)
    if n < MIN_FRAMES_TO_FILTER:
        print(f"{run_dir}: skipping energy k-sigma filter (only {n} frames < {MIN_FRAMES_TO_FILTER})")
        return ls, []
    E = _get_energies(ls)
    if E is None:
        print(f"{run_dir}: energy k-sigma skipped (no energies)")
        return ls, []
    nat = _get_natoms(ls)
    epa = E / float(nat)
    mu = float(np.mean(epa))
    sigma = float(np.std(epa))
    if sigma <= 0.0 or (not np.isfinite(sigma)):
        print(f"{run_dir}: energy sigma=0/nonfinite — energy k-sigma skipped")
        return ls, []

    if two_sided:
        mask = np.abs(epa - mu) > float(k_sigma_energy)*sigma
        cutoff = float("nan")
        reason = "|E/atom-mu|>Kσ"
    else:
        cutoff = mu + float(k_sigma_energy)*sigma
        mask = epa > cutoff
        reason = "E/atom>mu+Kσ"

    drop_idx = np.nonzero(mask)[0].tolist()
    if not drop_idx:
        print(f"{run_dir}: energy k-sigma kept {n}/{n} (removed 0); mu={mu:.4g}, sigma={sigma:.4g}")
        return ls, []

    rows = []
    header = ["run_dir","frame_index","energy_eV","energy_per_atom_eV","mu_energy_per_atom_eV","sigma_energy_per_atom_eV","cutoff_eV_per_atom","reason"]
    for j in drop_idx:
        rows.append(dict(
            run_dir=run_dir, frame_index=int(j),
            energy_eV=float(E[j]),
            energy_per_atom_eV=float(epa[j]),
            mu_energy_per_atom_eV=mu,
            sigma_energy_per_atom_eV=sigma,
            cutoff_eV_per_atom=float(cutoff) if np.isfinite(cutoff) else "",
            reason=reason,
        ))
    log_excluded_frames(log_dir, run_dir, rows, header, suffix="__energy_ksigma")

    keep_idx = [i for i in range(n) if i not in set(drop_idx)]
    out = ls.sub_system(keep_idx)
    print(f"{run_dir}: energy k-sigma kept {len(out)}/{n} (removed {len(drop_idx)}); mu={mu:.4g}, sigma={sigma:.4g}")
    return out, drop_idx

def second_pass_energy_floor_filter(ls: LabeledSystem, floor_eV_per_atom: float) -> Tuple[LabeledSystem, int, float, float, float]:
    n = len(ls)
    if n == 0:
        return ls, 0, float("nan"), float("nan"), float("nan")
    E = _get_energies(ls)
    if E is None:
        return ls, 0, float("nan"), float("nan"), float("nan")
    nat = _get_natoms(ls)
    epa = E / float(nat)
    mu = float(np.mean(epa))
    sigma = float(np.std(epa))
    cutoff = mu + float(floor_eV_per_atom)
    mask = epa > cutoff
    removed = int(np.count_nonzero(mask))
    keep_idx = np.nonzero(~mask)[0].tolist()
    return ls.sub_system(keep_idx), removed, mu, sigma, cutoff


# ---------------- Stats & split ----------------
def force_stats(ls: LabeledSystem) -> Dict[str, float]:
    out = dict(
        atom_mag_mean=float("nan"),
        atom_mag_std=float("nan"),
        frame_rms_mean=float("nan"),
        frame_rms_std=float("nan"),
        n_frames=int(len(ls)),
    )
    if len(ls) == 0:
        return out
    F = ls.data.get("forces", None)
    if F is None:
        return out
    F = np.asarray(F)
    mags = np.linalg.norm(F, axis=2)
    out["atom_mag_mean"] = float(mags.mean())
    out["atom_mag_std"] = float(mags.std())
    frame_rms = np.sqrt((mags**2).mean(axis=1))
    out["frame_rms_mean"] = float(frame_rms.mean())
    out["frame_rms_std"] = float(frame_rms.std())
    return out

def energy_stats(ls: LabeledSystem) -> Dict[str, float]:
    out = dict(energy_per_atom_mean=float("nan"), energy_per_atom_std=float("nan"))
    if len(ls) == 0:
        return out
    E = _get_energies(ls)
    if E is None:
        return out
    try:
        nat = _get_natoms(ls)
    except Exception:
        return out
    epa = E / float(nat)
    out["energy_per_atom_mean"] = float(np.mean(epa))
    out["energy_per_atom_std"] = float(np.std(epa))
    return out

def split_train_validation(ls: LabeledSystem, val_fraction: float) -> Tuple[LabeledSystem, LabeledSystem]:
    n = len(ls)
    if n < 2 or val_fraction <= 0:
        return ls, ls.sub_system([])
    n_val = int(round(float(val_fraction)*n))
    n_val = max(1, n_val)
    n_val = min(n_val, n-1)
    val_idx = list(range(0, n_val))
    train_idx = list(range(n_val, n))
    return ls.sub_system(train_idx), ls.sub_system(val_idx)


# ---------------- Frame-level duplicate detection ----------------
def _hash_array_for_frame(arr: Any, frame_index: int, decimals: int) -> bytes:
    """Return stable bytes for one frame of a numeric or integer array."""
    a = np.asarray(arr)
    if a.ndim > 0 and a.shape[0] > frame_index:
        a = a[frame_index]
    if np.issubdtype(a.dtype, np.floating):
        a = np.round(a.astype(np.float64), int(decimals))
    else:
        a = a.astype(np.int64, copy=False) if np.issubdtype(a.dtype, np.integer) else a
    return np.ascontiguousarray(a).tobytes()


def frame_hash(ls: LabeledSystem, frame_index: int, decimals: int) -> str:
    """Hash a DeepMD frame using atom metadata plus cell/coord/energy/force/virial data.

    This catches literal duplicated/restarted frames. It intentionally does not
    detect symmetry-equivalent, translated, or atom-permuted structures.
    """
    import hashlib
    h = hashlib.sha256()

    for key in ("atom_names", "atom_numbs", "atom_types"):
        if key in ls.data:
            arr = np.asarray(ls.data[key])
            h.update(key.encode()); h.update(b"=")
            h.update(str(arr.tolist()).encode()); h.update(b";")

    for key in ("cells", "coords", "energies", "forces", "virials"):
        if key not in ls.data:
            h.update(key.encode()); h.update(b"=MISSING;")
            continue
        h.update(key.encode()); h.update(b"=")
        h.update(_hash_array_for_frame(ls.data[key], frame_index, decimals)); h.update(b";")
    return h.hexdigest()


def split_unique_duplicate_frames(
    ls: LabeledSystem,
    seen_hashes: Dict[str, str],
    group_label: str,
    decimals: int,
) -> Tuple[LabeledSystem, LabeledSystem, List[Dict[str, Any]]]:
    """Split frames into unique candidates and duplicate frames to exclude."""
    unique_idx: List[int] = []
    dup_idx: List[int] = []
    duplicate_rows: List[Dict[str, Any]] = []

    for i in range(len(ls)):
        fh = frame_hash(ls, i, decimals)
        if fh in seen_hashes:
            dup_idx.append(i)
            duplicate_rows.append(dict(
                group=group_label,
                frame_index=int(i),
                hash=fh,
                duplicate_of=seen_hashes[fh],
            ))
        else:
            unique_idx.append(i)
            seen_hashes[fh] = f"{group_label}:frame{i}"

    return ls.sub_system(unique_idx), ls.sub_system(dup_idx), duplicate_rows


def split_random_validation_backfill(
    train_ls: LabeledSystem,
    n_backfill: int,
    rng: np.random.Generator,
) -> Tuple[LabeledSystem, LabeledSystem, List[int]]:
    """Move n_backfill random unique frames from training into validation."""
    n = len(train_ls)
    if n <= 0 or n_backfill <= 0:
        return train_ls, train_ls.sub_system([]), []
    n_pick = min(int(n_backfill), max(0, n - 1))
    if n_pick <= 0:
        return train_ls, train_ls.sub_system([]), []
    pick = sorted(int(x) for x in rng.choice(n, size=n_pick, replace=False).tolist())
    pick_set = set(pick)
    keep = [i for i in range(n) if i not in pick_set]
    return train_ls.sub_system(keep), train_ls.sub_system(pick), pick


# ---------------- Composition gate ----------------
def allowed_cuag_only(ls: LabeledSystem) -> Tuple[bool, List[str]]:
    names = ls.data.get("atom_names", None)
    if not names:
        return False, []
    elems = sorted(set(str(x) for x in names))
    ok = set(elems).issubset(ALLOWED_ELEMENTS)
    return ok, elems


# ---------------- Main ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build DeepMD datasets from VASP runs (Cu/Ag only) with electronic+force+energy filtering and temperature bucketing.")
    ap.add_argument("--glob", dest="glob_pat", default="./**/vasprun.xml")
    ap.add_argument("--base-out", default="../../clean_data/dpdata_CuAg_i8")
    ap.add_argument("--max-set-size", type=int, default=DEFAULT_MAX_SET_SIZE)

    ap.add_argument("--elec-filter", choices=["vasprun_oszicar","vasprun_only","oszicar_only","off"], default="vasprun_oszicar")
    ap.add_argument("--strict-elec", choices=["keep_if_unknown","drop_if_unknown"], default=DEFAULT_STRICT_ELEC)
    ap.add_argument("--elec-debug", action="store_true", default=False)

    ap.add_argument("--atom-excess-threshold", type=float, default=DEFAULT_ATOM_EXCESS_THRESHOLD)
    ap.add_argument("--energy-excess-threshold", type=float, default=DEFAULT_ENERGY_EXCESS_THRESHOLD)
    ap.add_argument("--k-sigma-frames", type=float, default=DEFAULT_K_SIGMA_FRAMES)
    ap.add_argument("--energy-k-sigma-frames", type=float, default=DEFAULT_ENERGY_K_SIGMA_FRAMES)
    ap.add_argument("--energy-two-sided", action="store_true", default=False)

    ap.add_argument("--second-pass-excess-floor", type=float, default=DEFAULT_SECOND_PASS_EXCESS_FLOOR)
    ap.add_argument("--energy-floor-eV-per-atom", type=float, default=DEFAULT_ENERGY_FLOOR_EV_PER_ATOM)

    ap.add_argument("--k-sigma-combined", type=float, default=None)
    ap.add_argument("--energy-k-sigma-combined", type=float, default=None)

    ap.add_argument("--temp-block-size", type=int, default=200)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--val-dirname", default=DEFAULT_VAL_DIRNAME)

    ap.add_argument("--rmv-dupli", "--rmv_dupli", dest="rmv_dupli", action="store_true", default=False,
                    help="Remove duplicate frames from both training and validation exports. Validation is sampled only from unique surviving frames.")
    ap.add_argument("--dedup-decimals", type=int, default=8,
                    help="Decimal places used when hashing numeric frame arrays for duplicate detection.")
    ap.add_argument("--dedup-scope", choices=["global", "group"], default="global",
                    help="global removes duplicates across all output buckets; group removes duplicates only within each mp/composition/structure/T bucket.")
    ap.add_argument("--random-seed", type=int, default=9999999,
                    help="Random seed for validation backfill sampling when --rmv-dupli is active.")

    args = ap.parse_args()

    base_out = Path(args.base_out)
    _ensure_dir(base_out)
    log_dir = base_out / "_filter_logs"
    _ensure_dir(log_dir)

    # Discover runs
    vasprun_files = glob(args.glob_pat, recursive=True)
    run_dirs = sorted({str(Path(f).parent) for f in vasprun_files})

    grouped: Dict[Tuple[str, str, str, str, str], List[Tuple[LabeledSystem, Dict[str,Any]]]] = {}
    per_run_rows: List[Dict[str,Any]] = []
    tot_before = 0
    tot_after_first = 0
    n_skipped_composition = 0

    for d in run_dirs:
        if any(skip in d for skip in ("broken_vaspruns", "incomplete_xml_runs")):
            continue

        ls = load_labeled_system_any(d)
        if ls is None or len(ls) == 0:
            print(f"{d}: could not load or empty")
            continue

        ok_comp, elems = allowed_cuag_only(ls)
        if not ok_comp:
            n_skipped_composition += 1
            rows = [dict(run_dir=d, frame_index="", reason=f"composition_reject elems={','.join(elems) if elems else 'UNKNOWN'}")]
            log_excluded_frames(log_dir, d, rows, ["run_dir","frame_index","reason"], suffix="__composition")
            print(f"{d}: skipped (contains elements outside Cu/Ag): {elems}")
            continue

        n0 = len(ls)
        tot_before += n0

        # Electronic prefilter
        if args.elec_filter != "off":
            ls1, drop_elec = prefilter_nonconverged(
                ls, d, log_dir,
                elec_filter_mode=args.elec_filter,
                strict_elec=args.strict_elec,
                elec_debug=args.elec_debug,
            )
        else:
            ls1, drop_elec = ls, []

        # Force atom-excess
        ls2, drop_atom = filter_frames_atom_excess(ls1, d, args.atom_excess_threshold, log_dir)

        # Energy excess
        ls3, drop_eex = filter_frames_energy_excess(ls2, d, args.energy_excess_threshold, log_dir)

        # Force k-sigma
        ls4, drop_fks = filter_frames_force_ksigma(ls3, d, args.k_sigma_frames, log_dir)

        # Energy k-sigma
        ls5, drop_eks = filter_frames_energy_ksigma(ls4, d, args.energy_k_sigma_frames, log_dir, two_sided=args.energy_two_sided)

        n_after = len(ls5)
        tot_after_first += n_after

        row = dict(
            run_dir=d,
            frames_before=n0,
            frames_after=n_after,
            frames_after_elec=len(ls1),
            frames_after_atom_force=len(ls2),
            frames_after_energy_excess=len(ls3),
            frames_after_force_ksigma=len(ls4),
            removed_elec=len(drop_elec),
            removed_atom_excess=len(drop_atom),
            removed_energy_excess=len(drop_eex),
            removed_force_ksigma=len(drop_fks),
            removed_energy_ksigma=len(drop_eks),
            removed_total=n0 - n_after,
        )
        per_run_rows.append(row)

        if n_after == 0:
            continue

        # Composition folder in fixed order (Cu then Ag)
        atoms = ls5.data["atom_names"]
        counts = ls5.data["atom_numbs"]
        elem_counts = dict(zip(atoms, counts))
        composition_folder = "".join(f"{el}{int(elem_counts.get(el,0))}" for el in priority_elements)

        mp_bucket = detect_materials_project_id(d)
        structure_cat = detect_structure_category(d)
        tblock = temp_block_label(parse_temperature_K(d), args.temp_block_size)

        row.update(dict(mp_id=mp_bucket, composition=composition_folder, structure=structure_cat, temp_block=tblock))

        has_virials = ("virials" in ls5.data)
        grouped.setdefault((mp_bucket, composition_folder, structure_cat, tblock, "v" if has_virials else "nv"), []).append((ls5, row))

    # Per-run summary
    per_run_csv = log_dir / "summary_per_run_first_pass.csv"
    with open(per_run_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_dir", "mp_id", "composition", "structure", "temp_block",
            "frames_before","frames_after",
            "frames_after_elec","frames_after_atom_force","frames_after_energy_excess","frames_after_force_ksigma",
            "removed_total",
            "removed_elec","removed_atom_excess","removed_energy_excess","removed_force_ksigma","removed_energy_ksigma",
        ])
        for r in per_run_rows:
            w.writerow([
                r["run_dir"], r.get("mp_id", ""), r.get("composition", ""), r.get("structure", ""), r.get("temp_block", ""),
                r["frames_before"], r["frames_after"],
                r["frames_after_elec"], r["frames_after_atom_force"], r["frames_after_energy_excess"], r["frames_after_force_ksigma"],
                r["removed_total"],
                r["removed_elec"], r["removed_atom_excess"], r["removed_energy_excess"], r["removed_force_ksigma"], r["removed_energy_ksigma"],
            ])

    print(f"\nPer-run first-pass summary CSV written to: {per_run_csv}")
    print(f"Total frames: before={tot_before}, after_first_pass={tot_after_first}")
    print(f"Runs skipped due to non-(Cu/Ag) composition: {n_skipped_composition}")

    # Combined stage
    combined_rows: List[Dict[str,Any]] = []
    duplicate_rows_all: List[Dict[str, Any]] = []
    random_validation_rows_all: List[Dict[str, Any]] = []
    seen_global: Dict[str, str] = {}
    rng = np.random.default_rng(int(args.random_seed))

    for (mpid, comp, struct, tblock, vtag), items in grouped.items():
        ls_list = [x[0] for x in items]
        run_rows = [x[1] for x in items]

        frames_initial = int(sum(r["frames_before"] for r in run_rows))
        frames_after_elec = int(sum(r["frames_after_elec"] for r in run_rows))
        frames_after_atom_force = int(sum(r["frames_after_atom_force"] for r in run_rows))
        frames_after_energy_excess = int(sum(r["frames_after_energy_excess"] for r in run_rows))
        frames_after_force_ksigma = int(sum(r["frames_after_force_ksigma"] for r in run_rows))
        frames_after_energy_ksigma = int(sum(r["frames_after"] for r in run_rows))
        comp_out = (comp + "_v") if vtag == "v" else comp

        combined = ls_list[0]
        for s_part in ls_list[1:]:
            combined.extend(s_part)

        n_before_2nd = len(combined)
        combined.shuffle()

        combined, removed_2nd_force = second_pass_excess_filter(combined, args.second_pass_excess_floor)
        n_after_2nd_force = len(combined)

        combined, removed_2nd_energy, mu_e2, sig_e2, cut_e2 = second_pass_energy_floor_filter(combined, args.energy_floor_eV_per_atom)
        n_after_2nd_energy = len(combined)

        # Optional combined filters are retained as no-ops for parity with v3.
        removed_combined_ks = 0
        mu_fk = sig_fk = cut_fk = float("nan")
        if args.k_sigma_combined is not None and len(combined) > 0:
            pass

        removed_combined_eks = 0
        mu_ek = sig_ek = cut_ek = float("nan")
        if args.energy_k_sigma_combined is not None and len(combined) > 0:
            pass

        stats_f = force_stats(combined)
        stats_e = energy_stats(combined)

        if len(combined) == 0:
            print(f"{mpid}/{comp}/{struct}/{tblock}: all frames removed — skipping export")
            combined_rows.append(dict(
                mp_id=mpid, composition=comp_out, virials_tag=vtag, structure=struct, temp_block=tblock,
                frames_initial=frames_initial,
                frames_after_elec=frames_after_elec,
                frames_after_atom_force=frames_after_atom_force,
                frames_after_energy_excess=frames_after_energy_excess,
                frames_after_force_ksigma=frames_after_force_ksigma,
                frames_after_energy_ksigma=frames_after_energy_ksigma,
                frames_before_second_pass=n_before_2nd,
                frames_after_second_pass_force=n_after_2nd_force,
                removed_second_pass_force=removed_2nd_force,
                floor_eV_per_A=float(args.second_pass_excess_floor),
                frames_after_second_pass_energy=n_after_2nd_energy,
                removed_second_pass_energy=removed_2nd_energy,
                energy_floor_eV_per_atom=float(args.energy_floor_eV_per_atom),
                atom_mag_mean=stats_f["atom_mag_mean"],
                atom_mag_std=stats_f["atom_mag_std"],
                frame_rms_mean=stats_f["frame_rms_mean"],
                frame_rms_std=stats_f["frame_rms_std"],
                energy_per_atom_mean=stats_e["energy_per_atom_mean"],
                energy_per_atom_std=stats_e["energy_per_atom_std"],
                frames_train=0, frames_valid=0, frames_exported=0,
                rmv_dupli=bool(args.rmv_dupli),
                dedup_scope=args.dedup_scope if args.rmv_dupli else "off",
                dedup_decimals=int(args.dedup_decimals) if args.rmv_dupli else "",
                frames_validation_duplicates=0,
                frames_duplicates_excluded=0,
                frames_validation_random=0,
                validation_target_frames=0,
            ))
            continue

        duplicate_rows: List[Dict[str, Any]] = []
        ndup = 0
        nrand = 0
        validation_target_frames = 0

        if args.rmv_dupli:
            group_label = "/".join([str(mpid), str(comp_out), str(struct), str(tblock), str(vtag)])
            seen_for_group = seen_global if args.dedup_scope == "global" else {}

            # Duplicates are excluded from exported train/validation data.
            # Validation is sampled only from unique surviving frames to avoid leakage.
            unique_ls, dup_ls, duplicate_rows = split_unique_duplicate_frames(
                combined, seen_for_group, group_label, int(args.dedup_decimals)
            )
            duplicate_rows_all.extend(duplicate_rows)
            ndup = len(dup_ls)

            n_unique = len(unique_ls)
            if args.val_fraction > 0 and n_unique >= 2:
                validation_target_frames = int(round(float(args.val_fraction) * n_unique))
                validation_target_frames = max(1, validation_target_frames)
                validation_target_frames = min(validation_target_frames, n_unique - 1)
            else:
                validation_target_frames = 0

            train_ls, rand_val_ls, rand_pick_idx = split_random_validation_backfill(unique_ls, validation_target_frames, rng)
            val_ls = rand_val_ls
            nrand = len(rand_val_ls)
            for j in rand_pick_idx:
                random_validation_rows_all.append(dict(
                    group=group_label,
                    original_unique_frame_index=int(j),
                    reason="random_unique_validation",
                    val_fraction=float(args.val_fraction),
                    validation_target_frames=int(validation_target_frames),
                    duplicate_frames_excluded_in_group=int(ndup),
                ))

            if len(train_ls) > 1:
                train_ls.shuffle()
        else:
            combined.shuffle()
            train_ls, val_ls = split_train_validation(combined, args.val_fraction)
            validation_target_frames = len(val_ls)

        # Export train with mp-* binning: mp_id / composition / structure / temp block.
        out_path = base_out / mpid / comp_out / struct / tblock
        _ensure_dir(out_path)
        train_ls.to_deepmd_raw(str(out_path), type_map=priority_elements, set_size=args.max_set_size)
        train_ls.to_deepmd_npy(str(out_path), type_map=priority_elements, set_size=args.max_set_size)
        ntr = len(train_ls)
        nsets = (ntr + args.max_set_size - 1)//args.max_set_size
        print(f"Wrote TRAIN to: {out_path}  -> {ntr} frames in {nsets} set(s) (max {args.max_set_size}/set)")

        # Export validation using the same mp-* binning below validation/.
        nva = len(val_ls)
        if nva > 0:
            vpath = base_out / args.val_dirname / mpid / comp_out / struct / tblock
            _ensure_dir(vpath)
            val_ls.to_deepmd_raw(str(vpath), type_map=priority_elements, set_size=args.max_set_size)
            val_ls.to_deepmd_npy(str(vpath), type_map=priority_elements, set_size=args.max_set_size)
            vsets = (nva + args.max_set_size - 1)//args.max_set_size
            if args.rmv_dupli:
                print(f"Wrote VALID to: {vpath}  -> {nva} unique-random frames; excluded {ndup} duplicate frames from this group")
            else:
                print(f"Wrote VALID to: {vpath}  -> {nva} frames in {vsets} set(s) (max {args.max_set_size}/set)")
        else:
            print(f"Validation split produced 0 frames for {mpid}/{comp_out}/{struct}/{tblock} (val_fraction={args.val_fraction:g})")

        combined_rows.append(dict(
            mp_id=mpid, composition=comp_out, virials_tag=vtag, structure=struct, temp_block=tblock,
            frames_initial=frames_initial,
            frames_after_elec=frames_after_elec,
            frames_after_atom_force=frames_after_atom_force,
            frames_after_energy_excess=frames_after_energy_excess,
            frames_after_force_ksigma=frames_after_force_ksigma,
            frames_after_energy_ksigma=frames_after_energy_ksigma,
            frames_before_second_pass=n_before_2nd,
            frames_after_second_pass_force=n_after_2nd_force,
            removed_second_pass_force=removed_2nd_force,
            floor_eV_per_A=float(args.second_pass_excess_floor),
            frames_after_second_pass_energy=n_after_2nd_energy,
            removed_second_pass_energy=removed_2nd_energy,
            energy_floor_eV_per_atom=float(args.energy_floor_eV_per_atom),
            atom_mag_mean=stats_f["atom_mag_mean"],
            atom_mag_std=stats_f["atom_mag_std"],
            frame_rms_mean=stats_f["frame_rms_mean"],
            frame_rms_std=stats_f["frame_rms_std"],
            energy_per_atom_mean=stats_e["energy_per_atom_mean"],
            energy_per_atom_std=stats_e["energy_per_atom_std"],
            frames_train=ntr, frames_valid=nva, frames_exported=len(train_ls)+len(val_ls),
            rmv_dupli=bool(args.rmv_dupli),
            dedup_scope=args.dedup_scope if args.rmv_dupli else "off",
            dedup_decimals=int(args.dedup_decimals) if args.rmv_dupli else "",
            frames_validation_duplicates=0,
            frames_duplicates_excluded=ndup,
            frames_validation_random=nrand,
            validation_target_frames=validation_target_frames,
        ))

    combined_csv = log_dir / "summary_combined_sets.csv"
    fieldnames = [
        "mp_id", "composition","virials_tag","structure","temp_block",
        "frames_initial","frames_after_elec","frames_after_atom_force","frames_after_energy_excess",
        "frames_after_force_ksigma","frames_after_energy_ksigma",
        "frames_before_second_pass","frames_after_second_pass_force","removed_second_pass_force","floor_eV_per_A",
        "frames_after_second_pass_energy","removed_second_pass_energy","energy_floor_eV_per_atom",
        "atom_mag_mean","atom_mag_std","frame_rms_mean","frame_rms_std",
        "energy_per_atom_mean","energy_per_atom_std",
        "frames_train","frames_valid","frames_exported",
        "rmv_dupli", "dedup_scope", "dedup_decimals",
        "frames_validation_duplicates", "frames_duplicates_excluded", "frames_validation_random", "validation_target_frames",
    ]
    with open(combined_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in combined_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nCombined-set summary CSV written to: {combined_csv}")

    # Final aggregate reporting across all exported groups.
    def _sum_int(rows, key):
        total = 0
        for rr in rows:
            try:
                total += int(rr.get(key, 0) or 0)
            except Exception:
                pass
        return total

    total_groups = len(combined_rows)
    total_train = _sum_int(combined_rows, "frames_train")
    total_valid = _sum_int(combined_rows, "frames_valid")
    total_exported = _sum_int(combined_rows, "frames_exported")
    total_after_second_energy = _sum_int(combined_rows, "frames_after_second_pass_energy")
    total_dups_in_validation = _sum_int(combined_rows, "frames_validation_duplicates")
    total_dups_excluded = _sum_int(combined_rows, "frames_duplicates_excluded")
    total_rand = _sum_int(combined_rows, "frames_validation_random")
    total_val_target = _sum_int(combined_rows, "validation_target_frames")
    valid_fraction_actual = (float(total_valid) / float(total_exported)) if total_exported > 0 else float("nan")
    duplicate_fraction_surviving = (float(total_dups_excluded) / float(total_after_second_energy)) if total_after_second_energy > 0 else float("nan")

    final_totals = {
        "n_groups_exported": total_groups,
        "frames_after_all_filters_before_dedup_split": total_after_second_energy,
        "frames_train_final": total_train,
        "frames_validation_final": total_valid,
        "frames_exported_final": total_exported,
        "frames_validation_duplicates": total_dups_in_validation,
        "frames_duplicates_excluded": total_dups_excluded,
        "frames_validation_random_unique": total_rand,
        "validation_target_frames_sum": total_val_target,
        "actual_validation_fraction": valid_fraction_actual,
        "duplicate_fraction_of_surviving_frames": duplicate_fraction_surviving,
        "rmv_dupli": bool(args.rmv_dupli),
        "dedup_scope": args.dedup_scope if args.rmv_dupli else "off",
        "dedup_decimals": int(args.dedup_decimals) if args.rmv_dupli else "",
        "requested_val_fraction": float(args.val_fraction),
    }
    totals_csv = log_dir / "summary_final_totals.csv"
    with open(totals_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final_totals.keys()))
        w.writeheader()
        w.writerow(final_totals)

    print("\n=== Final export totals ===")
    print(f"Groups exported:                         {total_groups}")
    print(f"Frames after filters before split:       {total_after_second_energy}")
    print(f"Final training frames:                   {total_train}")
    print(f"Final validation frames:                 {total_valid}")
    print(f"Final exported frames:                   {total_exported}")
    if total_exported > 0:
        print(f"Actual validation fraction:              {100.0 * valid_fraction_actual:.2f}%")
    if args.rmv_dupli:
        print(f"Validation duplicate frames:             {total_dups_in_validation}")
        print(f"Duplicate frames excluded:               {total_dups_excluded}")
        print(f"Validation random unique frames:         {total_rand}")
        print(f"Requested validation target frame sum:   {total_val_target}")
        if total_after_second_energy > 0:
            print(f"Duplicate fraction of surviving frames:  {100.0 * duplicate_fraction_surviving:.2f}%")
        if total_valid != total_val_target:
            print("NOTE: validation may differ slightly from target because each non-empty group keeps at least one training frame.")
    print(f"Final totals CSV written to:             {totals_csv}")

    if args.rmv_dupli:
        dup_csv = log_dir / "duplicate_frames_excluded.csv"
        with open(dup_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "frame_index", "hash", "duplicate_of"], extrasaction="ignore")
            w.writeheader()
            for r in duplicate_rows_all:
                w.writerow(r)

        rand_csv = log_dir / "random_unique_validation.csv"
        with open(rand_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["group", "original_unique_frame_index", "reason", "val_fraction", "validation_target_frames", "duplicate_frames_excluded_in_group"],
                extrasaction="ignore",
            )
            w.writeheader()
            for r in random_validation_rows_all:
                w.writerow(r)

        print(f"Duplicate-frame exclusion CSV written to: {dup_csv}")
        print(f"Random unique validation CSV written to: {rand_csv}")
        print(f"Duplicate-scope: {args.dedup_scope}; dedup decimals: {args.dedup_decimals}")
        print(f"Total duplicate frames excluded: {len(duplicate_rows_all)}")
        print(f"Total random unique frames moved to validation: {len(random_validation_rows_all)}")

    print(
        "\nDone.\n"
        f"  Grouping: Materials Project id / composition / structure / temperature bin; non-MP paths use NO_MP\n"
        f"  Duplicate removal: {'on' if args.rmv_dupli else 'off'}\n"
        f"  Output dir: {base_out}\n"
    )


if __name__ == "__main__":
    main()
