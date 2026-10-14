#!/usr/bin/env python3
"""Process and plot Cu-Ag PBE/PBEsol enthalpy scan outputs.

Expected setup layout:
  ./PBE/Cu_0/T800/enthalpy_avg.dat
  ./PBEsol/Cu_20/T1200/enthalpy_avg.dat

The preferred input file is enthalpy_avg.dat written by enthalpy.in:
  timestep temp press hpa pepa kepa etpa vpa density

Outputs:
  enthalpy_raw.csv
  enthalpy_summary.csv
  enthalpy_delta_PBEsol_minus_PBE.csv
  enthalpy_mixing.csv
  several PNG plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

FUNC_NAMES = {"PBE", "PBEsol"}
CU_RE = re.compile(r"^Cu_(\d+(?:\.\d+)?)$")
T_RE = re.compile(r"^T(\d+(?:\.\d+)?)$")
REP_RE = re.compile(r"^rep\d+$", re.I)

AVG_COLUMNS = [
    "step", "temp_K", "press_bar", "h_eV_atom", "pe_eV_atom", "ke_eV_atom",
    "etotal_eV_atom", "vol_A3_atom", "density_g_cm3"
]

# Manuscript plotting style matched to Figures 2/3.
FUNCTIONAL_STYLE = {
    "PBE": {"color": "tab:blue", "marker": "s", "linestyle": ":", "label": "PBE"},
    "PBEsol": {"color": "tab:red", "marker": "D", "linestyle": ":", "label": "PBEsol"},
}
EXPERIMENT_STYLE = {"color": "green", "marker": "o", "linestyle": ":", "label": "Fitzner exp. 1375 K"}
J_PER_MOL_PER_MEV_ATOM = 96.4853321233


def functional_style(func):
    return FUNCTIONAL_STYLE.get(str(func), {"color": "0.25", "marker": "o", "linestyle": ":", "label": str(func)})


def fitzner_cuag_liquid_hmix_j_mol(cu_atpct):
    """Fitzner et al. 1999 liquid Cu-Ag mixing enthalpy expression at 1375 +/- 2 K."""
    x_cu = float(cu_atpct) / 100.0
    x_ag = 1.0 - x_cu
    if x_cu <= 0.0 or x_cu >= 1.0:
        return 0.0
    return x_cu * x_ag * (16959.0 - 7620.0 * x_ag + 4287.0 * x_ag * x_ag)


def make_fitzner_experimental_mixing(comps=None):
    comps = [0, 20, 40, 50, 60, 80, 100] if comps is None else comps
    rows = []
    for comp in comps:
        h_j_mol = fitzner_cuag_liquid_hmix_j_mol(comp)
        rows.append({
            "source": "Fitzner et al. 1999",
            "T_K": 1375.0,
            "phase": "liquid",
            "Cu_atpct": float(comp),
            "h_mix_J_mol": h_j_mol,
            "h_mix_meV_atom": h_j_mol / J_PER_MOL_PER_MEV_ATOM,
            "formula": "XCu*XAg*(16959 - 7620*XAg + 4287*XAg^2) J/mol",
        })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Process Cu-Ag enthalpy/density scan outputs.")
    p.add_argument("--root", default=".", help="Root containing PBE/ and PBEsol/ folders. Default: .")
    p.add_argument("--outdir", default="enthalpy_processed", help="Output directory")
    p.add_argument("--functionals", default="PBE,PBEsol", help="Comma-separated functionals to process")
    p.add_argument("--frac-last", type=float, default=0.50, help="Fraction of enthalpy_avg.dat tail to average. Default: 0.50")
    p.add_argument("--min-points", type=int, default=20, help="Minimum tail points required for ok status")
    p.add_argument("--raw-name", default="enthalpy_raw.csv")
    p.add_argument("--summary-name", default="enthalpy_summary.csv")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--only-ok", action="store_true", help="Use only ok rows in plots/derived tables. Default behavior already excludes non-finite values.")
    p.add_argument("--formats", default="png,tiff", help="Comma-separated figure formats to write, e.g. png,tiff,pdf")
    p.add_argument("--dpi", type=int, default=600, help="DPI for raster figure outputs. Default: 600")
    p.add_argument("--no-experimental-mixing", action="store_true",
                   help="Do not plot/write the Fitzner et al. liquid Cu-Ag experimental mixing-enthalpy reference.")
    p.add_argument("--no-figure4", action="store_true",
                   help="Do not write the manuscript-style two-panel Figure 4 density/enthalpy plot.")
    p.add_argument("--figure4-name", default="figure4_density_enthalpy",
                   help="Base filename for the two-panel Figure 4 density/enthalpy plot.")
    p.add_argument("--figure4-enthalpy", choices=["mixing", "absolute"], default="mixing",
                   help="Panel B quantity: 'mixing' uses Delta H_mix in meV/atom; 'absolute' uses H in eV/atom.")
    p.add_argument("--temperature-cmap", default="viridis",
                   help="Matplotlib colormap used to color simulation series by temperature.")
    p.add_argument("--failed-csv", default="failed_enthalpy_runs.csv", help="CSV of non-ok run folders")
    p.add_argument("--failed-dirs", default="failed_enthalpy_dirs.txt", help="Plain-text list of non-ok run directories")
    return p.parse_args()


def as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def comp_label(c):
    return f"Cu{int(c) if float(c).is_integer() else c:g}"


def discover_runs(root: Path, functionals):
    runs = []
    for func in functionals:
        fdir = root / func
        if not fdir.is_dir():
            continue
        for cdir in sorted(fdir.iterdir()):
            m = CU_RE.match(cdir.name)
            if not cdir.is_dir() or not m:
                continue
            comp = float(m.group(1))
            for tdir in sorted(cdir.iterdir()):
                mt = T_RE.match(tdir.name)
                if not tdir.is_dir() or not mt:
                    continue
                temp = float(mt.group(1))
                # Direct run in T folder. Include every expected T folder so missing/failed jobs are reported.
                runs.append((func, comp, temp, "main", tdir))
                # Optional replicate subfolders.
                for sub in sorted(tdir.iterdir()):
                    if sub.is_dir() and REP_RE.match(sub.name):
                        runs.append((func, comp, temp, sub.name, sub))
    return runs


def read_metadata(folder: Path):
    p = folder / "enthalpy_setup.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def read_enthalpy_avg(path: Path):
    raw = path.read_bytes().replace(b"\x00", b"")
    from io import StringIO
    df = pd.read_csv(StringIO(raw.decode("utf-8", errors="ignore")), sep=r"\s+", comment="#", header=None, engine="python")
    if df.empty:
        raise ValueError("empty enthalpy_avg.dat")
    if df.shape[1] < 5:
        raise ValueError(f"expected at least 5 columns, found {df.shape[1]}")
    # Assign known columns; preserve extras if present.
    cols = AVG_COLUMNS[:df.shape[1]] + [f"extra_{i}" for i in range(max(0, df.shape[1] - len(AVG_COLUMNS)))]
    df.columns = cols[:df.shape[1]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def tail_stats(df: pd.DataFrame, frac_last: float, min_points: int):
    n = len(df)
    if n == 0:
        raise ValueError("no data rows")
    start = int(math.floor(max(0.0, min(1.0, 1.0 - frac_last)) * n))
    tail = df.iloc[start:].copy()
    if len(tail) < min_points:
        raise ValueError(f"only {len(tail)} tail points; need {min_points}")
    out = {"n_points": int(n), "n_tail": int(len(tail)), "tail_start_step": float(tail["step"].iloc[0]) if "step" in tail else np.nan}
    for col in ["temp_K", "press_bar", "h_eV_atom", "pe_eV_atom", "ke_eV_atom", "etotal_eV_atom", "vol_A3_atom", "density_g_cm3"]:
        if col not in tail.columns:
            out[f"{col}_mean"] = np.nan
            out[f"{col}_sd"] = np.nan
            out[f"{col}_sem"] = np.nan
            continue
        vals = tail[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if vals.size == 0:
            mean = sd = sem = np.nan
        elif vals.size == 1:
            mean, sd, sem = float(vals[0]), np.nan, np.nan
        else:
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            sem = float(sd / math.sqrt(vals.size))
        out[f"{col}_mean"] = mean
        out[f"{col}_sd"] = sd
        out[f"{col}_sem"] = sem
    return out


def process_run(func, comp, temp, run_label, folder, args):
    meta = read_metadata(folder)
    row = {
        "functional": func,
        "Cu_atpct": comp,
        "T_folder_K": temp,
        "run_label": run_label,
        "folder": str(folder),
        "phase": meta.get("phase", "unknown"),
        "Tm_threshold_K": as_float(meta.get("Tm_threshold_K", np.nan)),
        "model": meta.get("model", ""),
        "source_structure": meta.get("source_structure", ""),
        "status": "missing_enthalpy_avg",
    }
    p = folder / "enthalpy_avg.dat"
    if not p.is_file():
        return row
    try:
        df = read_enthalpy_avg(p)
        stats = tail_stats(df, args.frac_last, args.min_points)
        row.update(stats)
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = f"error: {exc}"
    return row


def aggregate(raw: pd.DataFrame):
    rows = []
    group_cols = ["functional", "Cu_atpct", "T_folder_K", "phase"]
    numeric_props = [
        "temp_K", "press_bar", "h_eV_atom", "pe_eV_atom", "ke_eV_atom", "etotal_eV_atom",
        "vol_A3_atom", "density_g_cm3"
    ]
    for key, g in raw.groupby(group_cols, dropna=False):
        func, comp, temp, phase = key
        ok = g[g["status"].eq("ok")].copy()
        base = {
            "functional": func,
            "Cu_atpct": comp,
            "T_folder_K": temp,
            "phase": phase,
            "n_runs_total": int(len(g)),
            "n_runs_ok": int(len(ok)),
            "status": "ok" if len(ok) else "no_data",
            "run_labels": ",".join(map(str, g["run_label"].tolist())),
        }
        for prop in numeric_props:
            c = f"{prop}_mean"
            if len(ok) == 0 or c not in ok.columns:
                base[f"{prop}_mean"] = np.nan
                base[f"{prop}_sd"] = np.nan
                base[f"{prop}_sem"] = np.nan
                continue
            vals = ok[c].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            if vals.size == 0:
                mean = sd = sem = np.nan
            elif vals.size == 1:
                mean = float(vals[0])
                # fall back to within-run SEM if available
                sem_col = f"{prop}_sem"
                sem = float(ok[sem_col].iloc[0]) if sem_col in ok and np.isfinite(ok[sem_col].iloc[0]) else np.nan
                sd = np.nan
            else:
                mean = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1))
                sem = float(sd / math.sqrt(vals.size))
            base[f"{prop}_mean"] = mean
            base[f"{prop}_sd"] = sd
            base[f"{prop}_sem"] = sem
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["functional", "T_folder_K", "Cu_atpct", "phase"])


def compute_delta(summary: pd.DataFrame):
    pbe = summary[(summary.functional == "PBE") & (summary.status == "ok")].copy()
    sol = summary[(summary.functional == "PBEsol") & (summary.status == "ok")].copy()
    keep = ["Cu_atpct", "T_folder_K", "phase", "h_eV_atom_mean", "density_g_cm3_mean", "vol_A3_atom_mean", "pe_eV_atom_mean"]
    m = sol[keep].merge(pbe[keep], on=["Cu_atpct", "T_folder_K", "phase"], suffixes=("_PBEsol", "_PBE"), how="inner")
    for prop in ["h_eV_atom", "pe_eV_atom", "density_g_cm3", "vol_A3_atom"]:
        a = f"{prop}_mean_PBEsol"
        b = f"{prop}_mean_PBE"
        if a in m and b in m:
            m[f"delta_{prop}_PBEsol_minus_PBE"] = m[a] - m[b]
    if "delta_h_eV_atom_PBEsol_minus_PBE" in m:
        m["delta_h_meV_atom_PBEsol_minus_PBE"] = 1000.0 * m["delta_h_eV_atom_PBEsol_minus_PBE"]
    return m.sort_values(["T_folder_K", "Cu_atpct", "phase"])


def compute_mixing(summary: pd.DataFrame):
    rows = []
    ok = summary[(summary.status == "ok") & summary["h_eV_atom_mean"].notna()].copy()
    for (func, temp, phase), g in ok.groupby(["functional", "T_folder_K", "phase"], dropna=False):
        h0 = g[g.Cu_atpct == 0.0]
        h100 = g[g.Cu_atpct == 100.0]
        if h0.empty or h100.empty:
            continue
        h_ag = float(h0["h_eV_atom_mean"].iloc[0])
        h_cu = float(h100["h_eV_atom_mean"].iloc[0])
        for _, r in g.iterrows():
            x = float(r.Cu_atpct) / 100.0
            h_ref = x * h_cu + (1.0 - x) * h_ag
            hmix = float(r.h_eV_atom_mean) - h_ref
            rows.append({
                "functional": func,
                "T_folder_K": temp,
                "phase": phase,
                "Cu_atpct": float(r.Cu_atpct),
                "h_mix_eV_atom": hmix,
                "h_mix_meV_atom": 1000.0 * hmix,
                "h_eV_atom_mean": float(r.h_eV_atom_mean),
                "h_ref_eV_atom": h_ref,
            })
    return pd.DataFrame(rows).sort_values(["functional", "T_folder_K", "phase", "Cu_atpct"]) if rows else pd.DataFrame()


def marker_for_phase(phase):
    # Retained for backward compatibility. Main manuscript styling uses functional markers.
    return "s" if str(phase).lower() == "solid" else "o" if str(phase).lower() == "liquid" else "x"


def parse_formats(spec):
    return [x.strip().lower().lstrip('.') for x in str(spec).split(',') if x.strip()]


def save_figure(fig, outdir: Path, filename: str, formats, dpi: int):
    base = outdir / Path(filename).with_suffix('').name
    for fmt in formats:
        outfile = base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
            kwargs["dpi"] = dpi
        fig.savefig(outfile, **kwargs)


def style_axis(ax, xkind=None):
    ax.tick_params(axis="both", which="major", direction="in", top=False, right=False)
    ax.tick_params(axis="both", which="minor", direction="in", top=False, right=False, length=2.5)
    if xkind == "composition":
        ax.set_xlim(-3.0, 103.0)
        ax.xaxis.set_minor_locator(MultipleLocator(10))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def legend_outside_right(ax, fontsize=7):
    ax.legend(frameon=False, fontsize=fontsize, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def functional_marker(func):
    """Functional encoding for Figure 4 when color is reserved for temperature."""
    s = str(func).strip().lower()
    if s == "pbe" or ("pbe" in s and "sol" not in s):
        return "s"
    if "pbesol" in s or "pbe sol" in s:
        return "D"
    return "o"


def functional_linestyle(func):
    # Keep dotted manuscript style, but make PBE/PBEsol distinguishable if printed grayscale.
    s = str(func).strip().lower()
    if s == "pbe" or ("pbe" in s and "sol" not in s):
        return ":"
    if "pbesol" in s or "pbe sol" in s:
        return "--"
    return ":"


def temperature_color_lookup(temps, cmap_name="viridis", include_exp_1375=False):
    vals = sorted(float(t) for t in temps if np.isfinite(float(t)))
    if include_exp_1375 and 1375.0 not in vals:
        vals.append(1375.0)
        vals = sorted(vals)
    if not vals:
        return {}
    cmap = plt.get_cmap(cmap_name)
    if len(vals) == 1:
        return {vals[0]: cmap(0.60)}
    return {t: cmap(i / (len(vals) - 1)) for i, t in enumerate(vals)}


def make_temperature_legend_handles(temp_colors, temps_used):
    from matplotlib.lines import Line2D
    handles = []
    for t in sorted(float(x) for x in temps_used):
        if t in temp_colors:
            label = f"{int(t)} K" if float(t).is_integer() else f"{t:g} K"
            handles.append(Line2D([0], [0], color=temp_colors[t], marker="o", linestyle="-",
                                  linewidth=1.4, markersize=5.5, label=label))
    return handles


def make_functional_legend_handles(functionals, include_exp=False):
    from matplotlib.lines import Line2D
    handles = []
    for func in [f for f in ["PBE", "PBEsol"] if f in set(functionals)]:
        handles.append(Line2D([0], [0], color="0.20", marker=functional_marker(func),
                              linestyle=functional_linestyle(func), linewidth=1.3, markersize=6.0,
                              label=func))
    if include_exp:
        handles.append(Line2D([0], [0], color="0.20", marker="o", linestyle="-",
                              linewidth=1.3, markersize=5.5, label="Fitzner exp."))
    return handles


def plot_figure4_density_enthalpy(summary: pd.DataFrame, mix: pd.DataFrame, outdir: Path,
                                  formats, dpi, include_experiment=True,
                                  filename="figure4_density_enthalpy", enthalpy_mode="mixing",
                                  cmap_name="viridis"):
    """Make manuscript Figure 4 as two panels: A density, B enthalpy.

    Color encodes temperature. Marker/line style encodes functional.
    Panel B defaults to mixing enthalpy because this is the physically
    meaningful comparison across functionals and against Fitzner et al.
    """
    if not HAVE_MPL:
        return

    dens = summary[(summary.status == "ok") & summary["density_g_cm3_mean"].notna()].copy()

    if enthalpy_mode == "mixing":
        enth = mix.copy() if mix is not None and not mix.empty else pd.DataFrame()
        ycol = "h_mix_meV_atom"
        ylab = r"$\mathbf{\Delta H_{mix}\ (meV/atom)}$"
    else:
        enth = summary[(summary.status == "ok") & summary["h_eV_atom_mean"].notna()].copy()
        ycol = "h_eV_atom_mean"
        ylab = r"$\mathbf{H\ (eV/atom)}$"

    if dens.empty and enth.empty and not include_experiment:
        return

    temps = []
    if not dens.empty:
        temps.extend(dens.T_folder_K.dropna().astype(float).unique().tolist())
    if not enth.empty:
        tcol = "T_folder_K" if "T_folder_K" in enth.columns else None
        if tcol:
            temps.extend(enth[tcol].dropna().astype(float).unique().tolist())
    temp_colors = temperature_color_lookup(temps, cmap_name=cmap_name, include_exp_1375=(include_experiment and enthalpy_mode == "mixing"))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)

    # Panel A: density
    ax = axes[0]
    if not dens.empty:
        for func in [f for f in ["PBE", "PBEsol"] if f in set(dens.functional)]:
            df = dens[dens.functional == func]
            for t in sorted(df.T_folder_K.dropna().unique()):
                sub = df[df.T_folder_K == t].sort_values("Cu_atpct")
                if sub.empty:
                    continue
                c = temp_colors.get(float(t), "0.35")
                ax.errorbar(
                    sub.Cu_atpct, sub.density_g_cm3_mean,
                    yerr=sub.density_g_cm3_sem if "density_g_cm3_sem" in sub else None,
                    color=c, marker=functional_marker(func), linestyle=functional_linestyle(func),
                    linewidth=1.2, markersize=5.5, capsize=2.5, alpha=0.95,
                    label=f"{func} {int(t)} K",
                )
    ax.set_xlabel(r"$\mathbf{Cu\ composition\ (at.\%)}$")
    ax.set_ylabel(r"$\mathbf{Density\ (g/cm^3)}$")
    style_axis(ax, xkind="composition")
    ax.text(0.02, 0.98, "A)", transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")

    # Panel B: enthalpy or mixing enthalpy
    ax = axes[1]
    if not enth.empty:
        for func in [f for f in ["PBE", "PBEsol"] if f in set(enth.functional)]:
            df = enth[enth.functional == func]
            for t in sorted(df.T_folder_K.dropna().unique()):
                # Preserve separate phase curves only if both phases are present for a functional/T.
                group_cols = ["phase"] if "phase" in df.columns else []
                if group_cols:
                    iterator = df[df.T_folder_K == t].groupby(group_cols, dropna=False)
                else:
                    iterator = [(None, df[df.T_folder_K == t])]
                for phase, sub in iterator:
                    sub = sub.sort_values("Cu_atpct")
                    if sub.empty or ycol not in sub:
                        continue
                    c = temp_colors.get(float(t), "0.35")
                    ax.plot(
                        sub.Cu_atpct, sub[ycol],
                        color=c, marker=functional_marker(func), linestyle=functional_linestyle(func),
                        linewidth=1.2, markersize=5.5, alpha=0.95,
                        label=f"{func} {int(t)} K",
                    )
    if include_experiment and enthalpy_mode == "mixing":
        exp = make_fitzner_experimental_mixing()
        c = temp_colors.get(1375.0, "0.20")
        ax.plot(
            exp.Cu_atpct, exp.h_mix_meV_atom,
            color=c, marker="o", linestyle="-", linewidth=1.4, markersize=5.5,
            markerfacecolor="none", markeredgewidth=1.2, zorder=10,
            label="Fitzner exp. 1375 K",
        )
        ax.axhline(0.0, color="0.55", linewidth=0.8)
    ax.set_xlabel(r"$\mathbf{Cu\ composition\ (at.\%)}$")
    ax.set_ylabel(ylab)
    style_axis(ax, xkind="composition")
    ax.text(0.02, 0.98, "B)", transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")

    # Separate compact legends: temperature colors and source/functional symbols.
    temps_used = sorted(set([float(t) for t in temps if np.isfinite(float(t))]))
    if include_experiment and enthalpy_mode == "mixing":
        temps_used = sorted(set(temps_used + [1375.0]))
    temp_handles = make_temperature_legend_handles(temp_colors, temps_used)
    func_handles = make_functional_legend_handles(
        list(set(summary.functional.dropna().astype(str))) if "functional" in summary else [],
        include_exp=(include_experiment and enthalpy_mode == "mixing")
    )

    if temp_handles:
        leg1 = axes[1].legend(handles=temp_handles, title="Temperature", frameon=False,
                              fontsize=8, title_fontsize=9, loc="upper left",
                              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        axes[1].add_artist(leg1)
    if func_handles:
        axes[1].legend(handles=func_handles, title="Series", frameon=False,
                       fontsize=8, title_fontsize=9, loc="lower left",
                       bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
    save_figure(fig, outdir, filename, formats, dpi)
    plt.close(fig)


def plot_vs_temperature(summary: pd.DataFrame, outdir: Path, ycol: str, ylabel: str, filename: str, formats, dpi):
    if not HAVE_MPL:
        return
    data = summary[(summary.status == "ok") & summary[ycol].notna()].copy()
    if data.empty:
        return
    funcs = [f for f in ["PBE", "PBEsol"] if f in set(data.functional)]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for func in funcs:
        df = data[data.functional == func]
        st = functional_style(func)
        for comp in sorted(df.Cu_atpct.unique()):
            sub = df[df.Cu_atpct == comp].sort_values("T_folder_K")
            if sub.empty:
                continue
            # Use the Figure 2/3 functional marker and color. Composition remains in the label.
            ax.plot(
                sub.T_folder_K, sub[ycol],
                color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                linewidth=1.1, markersize=5.0, alpha=0.85,
                label=f"{func} Cu{int(comp)}%",
            )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    style_axis(ax, xkind="temperature")
    legend_outside_right(ax, fontsize=6.5)
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.16, top=0.95)
    save_figure(fig, outdir, filename, formats, dpi)
    plt.close(fig)


def plot_vs_composition(summary: pd.DataFrame, outdir: Path, ycol: str, ylabel: str, filename: str, formats, dpi):
    if not HAVE_MPL:
        return
    data = summary[(summary.status == "ok") & summary[ycol].notna()].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for func in ["PBE", "PBEsol"]:
        df = data[data.functional == func]
        if df.empty:
            continue
        st = functional_style(func)
        for t in sorted(df.T_folder_K.unique()):
            sub = df[df.T_folder_K == t].sort_values("Cu_atpct")
            if sub.empty:
                continue
            ax.plot(
                sub.Cu_atpct, sub[ycol],
                color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                linewidth=1.1, markersize=5.0, alpha=0.85,
                label=f"{func} {int(t)} K",
            )
    ax.set_xlabel("Cu composition (at.%)")
    ax.set_ylabel(ylabel)
    style_axis(ax, xkind="composition")
    legend_outside_right(ax, fontsize=6.5)
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.16, top=0.95)
    save_figure(fig, outdir, filename, formats, dpi)
    plt.close(fig)


def plot_mixing(mix: pd.DataFrame, outdir: Path, formats, dpi, include_experiment=True):
    if not HAVE_MPL:
        return
    if mix.empty and not include_experiment:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    if not mix.empty:
        for func in ["PBE", "PBEsol"]:
            df = mix[mix.functional == func]
            if df.empty:
                continue
            st = functional_style(func)
            for (t, phase), sub in df.groupby(["T_folder_K", "phase"], dropna=False):
                sub = sub.sort_values("Cu_atpct")
                phase_label = str(phase)
                ax.plot(
                    sub.Cu_atpct, sub.h_mix_meV_atom,
                    color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                    linewidth=1.2, markersize=5.5, alpha=0.90,
                    label=f"{func} {int(t)} K {phase_label}",
                )
    if include_experiment:
        exp = make_fitzner_experimental_mixing()
        ax.plot(
            exp.Cu_atpct, exp.h_mix_meV_atom,
            color=EXPERIMENT_STYLE["color"], marker=EXPERIMENT_STYLE["marker"],
            linestyle=EXPERIMENT_STYLE["linestyle"], linewidth=1.3, markersize=5.5,
            label=EXPERIMENT_STYLE["label"], zorder=10,
        )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Cu composition (at.%)")
    ax.set_ylabel(r"$\Delta H_{mix}$ (meV/atom)")
    style_axis(ax, xkind="composition")
    legend_outside_right(ax, fontsize=6.5)
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.16, top=0.95)
    save_figure(fig, outdir, "mixing_enthalpy_vs_composition", formats, dpi)
    plt.close(fig)


def write_failed_lists(raw: pd.DataFrame, outdir: Path, csv_name: str, txt_name: str):
    failed = raw[~raw.status.eq("ok")].copy()
    if failed.empty:
        (outdir / txt_name).write_text("")
        failed.to_csv(outdir / csv_name, index=False)
        return failed
    cols = [c for c in ["functional", "Cu_atpct", "T_folder_K", "run_label", "folder", "phase", "status", "model", "source_structure"] if c in failed.columns]
    failed[cols].sort_values(["functional", "Cu_atpct", "T_folder_K", "run_label"]).to_csv(outdir / csv_name, index=False)
    dirs = failed["folder"].dropna().astype(str).tolist()
    (outdir / txt_name).write_text("\n".join(dirs) + "\n")
    return failed

def main():
    args = parse_args()
    if not (0.0 < args.frac_last <= 1.0):
        raise SystemExit("--frac-last must be in (0, 1]")
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    functionals = [x.strip() for x in args.functionals.split(',') if x.strip()]
    formats = parse_formats(args.formats)
    include_experiment = not args.no_experimental_mixing

    runs = discover_runs(root, functionals)
    if not runs:
        raise SystemExit(f"No enthalpy run folders found under {root}")

    raw_rows = [process_run(*r, args=args) for r in runs]
    raw = pd.DataFrame(raw_rows).sort_values(["functional", "T_folder_K", "Cu_atpct", "run_label"])
    summary = aggregate(raw)
    delta = compute_delta(summary)
    mix = compute_mixing(summary)

    raw.to_csv(outdir / args.raw_name, index=False)
    summary.to_csv(outdir / args.summary_name, index=False)
    delta.to_csv(outdir / "enthalpy_delta_PBEsol_minus_PBE.csv", index=False)
    if not mix.empty:
        mix.to_csv(outdir / "enthalpy_mixing.csv", index=False)
    if include_experiment:
        make_fitzner_experimental_mixing().to_csv(outdir / "enthalpy_mixing_experiment_fitzner_1375K.csv", index=False)
    failed = write_failed_lists(raw, outdir, args.failed_csv, args.failed_dirs)

    status_payload = {
        "raw_rows": int(len(raw)),
        "summary_rows": int(len(summary)),
        "raw_status_counts": raw.status.value_counts(dropna=False).to_dict(),
        "summary_status_counts": summary.status.value_counts(dropna=False).to_dict(),
        "failed_rows": int(len(failed)),
        "failed_csv": args.failed_csv,
        "failed_dirs": args.failed_dirs,
        "figure4_name": args.figure4_name,
        "figure4_enthalpy": args.figure4_enthalpy,
    }
    (outdir / "summary.json").write_text(json.dumps(status_payload, indent=2))

    if not args.no_plots:
        if not args.no_figure4:
            plot_figure4_density_enthalpy(
                summary, mix, outdir, formats, args.dpi,
                include_experiment=include_experiment,
                filename=args.figure4_name,
                enthalpy_mode=args.figure4_enthalpy,
                cmap_name=args.temperature_cmap,
            )
        plot_vs_temperature(summary, outdir, "h_eV_atom_mean", "Enthalpy (eV/atom)", "enthalpy_vs_temperature", formats, args.dpi)
        plot_vs_temperature(summary, outdir, "pe_eV_atom_mean", "Potential energy (eV/atom)", "potential_energy_vs_temperature", formats, args.dpi)
        plot_vs_temperature(summary, outdir, "density_g_cm3_mean", r"Density (g/cm$^3$)", "density_vs_temperature", formats, args.dpi)
        plot_vs_temperature(summary, outdir, "vol_A3_atom_mean", r"Volume ($\AA^3$/atom)", "volume_vs_temperature", formats, args.dpi)
        plot_vs_composition(summary, outdir, "h_eV_atom_mean", "Enthalpy (eV/atom)", "enthalpy_vs_composition", formats, args.dpi)
        plot_vs_composition(summary, outdir, "density_g_cm3_mean", r"Density (g/cm$^3$)", "density_vs_composition", formats, args.dpi)
        plot_mixing(mix, outdir, formats, args.dpi, include_experiment=include_experiment)

    print(f"Wrote outputs to {outdir}")
    print(f"Raw rows: {len(raw)}")
    print(f"Summary rows: {len(summary)}")
    print("Raw status counts:", raw.status.value_counts(dropna=False).to_dict())
    print("Summary status counts:", summary.status.value_counts(dropna=False).to_dict())
    print(f"Failed/non-ok run directories written to: {outdir / args.failed_dirs}")
    print(f"Failed/non-ok run CSV written to: {outdir / args.failed_csv}")
    if mix.empty:
        print("Mixing enthalpy table not written: missing same-phase endpoint references.")


if __name__ == "__main__":
    main()
