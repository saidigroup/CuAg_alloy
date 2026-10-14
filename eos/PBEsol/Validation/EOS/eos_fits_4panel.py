#!/usr/bin/env python3
"""
Collect DFT/DNP EOS data, fit Birch-Murnaghan curves, and generate the new
4-panel Figure 1 layout:

    A) PBE DFT
    B) DNP-PBE
    C) PBEsol DFT
    D) DNP-PBEsol

The script still writes the pointwise comparison CSV and fit-summary CSV.
It also writes a DNP-DFT pointwise error SI figure and a 4-panel main EOS figure.
"""
from pathlib import Path
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

# ---------------- USER SETTINGS ----------------
ROOTS = {
    "PBE": Path("PBE"),
    "PBEsol": Path("PBEsol"),
}

# Missing directories are skipped, so this works for either 3- or 5-composition datasets.
SYSTEMS = ["Cu_0", "Cu_25", "Cu_50", "Cu_75", "Cu_100"]

SYSTEM_LABELS = {
    "Cu_0": r"Ag",
    "Cu_25": r"Cu$_{25}$Ag$_{75}$",
    "Cu_50": r"Cu$_{50}$Ag$_{50}$",
    "Cu_75": r"Cu$_{75}$Ag$_{25}$",
    "Cu_100": r"Cu",
}

NATOMS = {
    "Cu_0": 32,
    "Cu_25": 32,
    "Cu_50": 32,
    "Cu_75": 32,
    "Cu_100": 32,
}

PANEL_DEFS = [
    ("PBE", "DFT", "A) PBE"),
    ("PBE", "DNP", "B) DNP-PBE"),
    ("PBEsol", "DFT", "C) PBEsol"),
    ("PBEsol", "DNP", "D) DNP-PBEsol"),
]

WRITE_POINT_CSV = "eos_pointwise_comparison.csv"
WRITE_FIT_CSV = "eos_fit_summary.csv"

MAKE_DELTAE_PLOTS = True
MAKE_EOS_FIT_PLOTS = True
MAKE_DFT_ONLY_FIT_PLOT = False

DELTAE_PLOT = "EOS_deltaE_by_composition.png"
FIT_PLOT = "EOS_fit_4panel.png"
DFT_ONLY_FIT_PLOT = "EOS_fit_DFT_only_4panel.png"

# Plot raw points/fits as volume per atom and relative energy in meV/atom.
PLOT_VOLUME_PER_ATOM = True
PLOT_ENERGY_MEV_PER_ATOM = True

DPI = 300

COMPOSITION_COLORS = {
    "Cu_0": "#1f77b4",
    "Cu_25": "#ff7f0e",
    "Cu_50": "#2ca02c",
    "Cu_75": "#d62728",
    "Cu_100": "#9467bd",
}

# Match existing method symbols from the previous scripts.
METHOD_MARKERS = {
    "DFT": "^",
    "DNP": "s",
}

METHOD_LINESTYLES = {
    "DFT": "-",
    "DNP": "--",
}
# ------------------------------------------------


def parse_vfactor_from_name(name: str):
    m = re.match(r"V_([0-9.]+)$", name)
    if not m:
        return None
    return float(m.group(1))


def poscar_volume(path: Path):
    lines = path.read_text().splitlines()
    if len(lines) < 5:
        return None

    scale = float(lines[1].split()[0])
    a = [float(x) for x in lines[2].split()[:3]]
    b = [float(x) for x in lines[3].split()[:3]]
    c = [float(x) for x in lines[4].split()[:3]]

    a = [scale * x for x in a]
    b = [scale * x for x in b]
    c = [scale * x for x in c]

    vol = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    return abs(vol)


def lammps_data_volume(path: Path):
    if not path.exists():
        return None
    lines = path.read_text(errors="ignore").splitlines()
    xlo = xhi = ylo = yhi = zlo = zhi = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.endswith("xlo xhi"):
            parts = s.split()
            xlo, xhi = float(parts[0]), float(parts[1])
        elif s.endswith("ylo yhi"):
            parts = s.split()
            ylo, yhi = float(parts[0]), float(parts[1])
        elif s.endswith("zlo zhi"):
            parts = s.split()
            zlo, zhi = float(parts[0]), float(parts[1])
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        return None
    return abs((xhi - xlo) * (yhi - ylo) * (zhi - zlo))


def read_vasp_energy(vdir: Path):
    oszicar = vdir / "OSZICAR"
    if not oszicar.exists():
        return None
    text = oszicar.read_text(errors="ignore")
    matches = re.findall(r"^\s*\d+\s+F=\s*([-\+\d.Ee]+)", text, flags=re.MULTILINE)
    if matches:
        return float(matches[-1])
    return None


def read_vasp_volume(vdir: Path):
    contcar = vdir / "CONTCAR"
    if contcar.exists():
        return poscar_volume(contcar)
    poscar = vdir / "POSCAR"
    if poscar.exists():
        return poscar_volume(poscar)
    return None


def find_lammps_output(vdir: Path):
    for name in ["out_relaxed", "relaxed.out", "log.lammps", "out.lammps"]:
        f = vdir / name
        if f.exists():
            return f
    return None


def read_lammps_energy(vdir: Path):
    out = find_lammps_output(vdir)
    if out is None:
        return None
    text = out.read_text(errors="ignore")
    patterns = [
        r"Total energy \(eV\)\s*=\s*([-\+\d.Ee]+)\s*;",
        r"Cohesive energy \(eV/atom\)\s*=\s*([-\+\d.Ee]+)\s*;",
        r"Cohesive energy \(eV\)\s*=\s*([-\+\d.Ee]+)\s*;",
    ]
    for pat in patterns:
        m = re.findall(pat, text)
        if m:
            return float(m[-1])
    return None


def read_lammps_volume(vdir: Path):
    fin_lmp = vdir / "fin.lmp"
    if fin_lmp.exists():
        return lammps_data_volume(fin_lmp)
    return None


def collect_points(root: Path, system: str):
    system_dir = root / system
    rows = []
    if not system_dir.exists():
        return rows
    vdirs = sorted(
        [d for d in system_dir.iterdir() if d.is_dir() and d.name.startswith("V_")],
        key=lambda d: parse_vfactor_from_name(d.name)
        if parse_vfactor_from_name(d.name) is not None
        else 999,
    )
    for vdir in vdirs:
        vf = parse_vfactor_from_name(vdir.name)
        rows.append({
            "folder": vdir.name,
            "vf": vf,
            "vasp_E": read_vasp_energy(vdir),
            "vasp_V": read_vasp_volume(vdir),
            "dnp_E": read_lammps_energy(vdir),
            "dnp_V": read_lammps_volume(vdir),
            "lmp_file": (vdir / "fin.lmp").name if (vdir / "fin.lmp").exists() else None,
        })
    return rows


def birch_murnaghan(V, E0, B0, BP, V0):
    eta = (V0 / V) ** (2.0 / 3.0) - 1.0
    return E0 + (9.0 * V0 * B0 / 16.0) * (BP * eta**3 + (6.0 - 4.0 * eta) * eta**2)


def initial_guess(vols, enes):
    idx = int(np.argmin(enes))
    E0 = float(enes[idx])
    V0 = float(vols[idx])
    if len(vols) >= 3:
        p = np.polyfit(vols, enes, 2)
        a = p[0]
        B0_guess = max(2 * a * V0, 0.01) if a > 0 else 0.5
    else:
        B0_guess = 0.5
    return [E0, B0_guess, 4.0, V0]


def fit_bm(vols, enes):
    vols = np.asarray(vols, dtype=float)
    enes = np.asarray(enes, dtype=float)
    if curve_fit is None or len(vols) < 4:
        return None
    p0 = initial_guess(vols, enes)
    lower = [min(enes) - 5.0, 1e-5, 0.5, min(vols) * 0.9]
    upper = [max(enes) + 5.0, 10.0, 10.0, max(vols) * 1.1]
    try:
        popt, _ = curve_fit(
            birch_murnaghan,
            vols,
            enes,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50000,
        )
        return {
            "E0_eV": float(popt[0]),
            "B0_eV_per_A3": float(popt[1]),
            "B0prime": float(popt[2]),
            "V0_A3": float(popt[3]),
            "B0_GPa": float(popt[1] * 160.21766208),
        }
    except Exception as exc:
        print(f"WARNING: Birch-Murnaghan fit failed: {exc}")
        return None


def write_pointwise_csv(all_rows, outfile):
    fields = [
        "model", "system", "folder", "vf",
        "vasp_E_eV", "vasp_V_A3", "dnp_E_eV", "dnp_V_A3",
        "deltaE_eV", "deltaE_meV_per_atom", "lmp_file",
    ]
    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)


def write_fit_csv(rows, outfile):
    fields = [
        "model", "system", "method",
        "E0_eV", "V0_A3", "B0_eV_per_A3", "B0_GPa", "B0prime",
    ]
    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_x(vols, system):
    n = NATOMS[system]
    return vols / n if PLOT_VOLUME_PER_ATOM else vols


def plot_y_relative(enes, system, emin=None):
    n = NATOMS[system]
    if emin is None:
        emin = float(np.min(enes))
    y = np.asarray(enes, dtype=float) - float(emin)
    return 1000.0 * y / n if PLOT_ENERGY_MEV_PER_ATOM else y


def make_deltae_plots(all_rows):
    available_systems = [s for s in SYSTEMS if any(r["system"] == s and r["deltaE_eV"] is not None for r in all_rows)]
    if not available_systems:
        return
    ncols = min(3, len(available_systems))
    nrows = int(np.ceil(len(available_systems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False, sharey=True)
    axes_flat = axes.ravel()

    model_colors = {"PBE": "#1f77b4", "PBEsol": "#d62728"}
    model_markers = {"PBE": "s", "PBEsol": "D"}

    for ax, system in zip(axes_flat, available_systems):
        for model in ["PBE", "PBEsol"]:
            rows = [r for r in all_rows if r["model"] == model and r["system"] == system and r["deltaE_eV"] is not None]
            rows = sorted(rows, key=lambda r: r["vf"])
            if not rows:
                continue
            x = [r["vf"] for r in rows]
            y = [r["deltaE_meV_per_atom"] for r in rows]
            ax.plot(x, y, marker=model_markers[model], linestyle="-", color=model_colors[model], linewidth=1.7, markersize=6, label=model)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        ax.text(0.03, 0.94, SYSTEM_LABELS.get(system, system), transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        ax.set_xlabel("Relative volume factor", fontweight="bold")
        ax.set_ylabel(r"$E_{DNP}-E_{DFT}$ (meV/atom)", fontweight="bold")
        ax.tick_params(direction="in", which="both")

    for ax in axes_flat[len(available_systems):]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(DELTAE_PLOT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_fit_plots(fit_inputs, fit_results):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (model, method, panel_title) in zip(axes, PANEL_DEFS):
        marker = METHOD_MARKERS[method]
        linestyle = METHOD_LINESTYLES[method]
        for system in SYSTEMS:
            key = (model, system, method)
            pts = fit_inputs.get(key, [])
            fit = fit_results.get(key)
            if not pts:
                continue
            vols = np.array([p[0] for p in pts], dtype=float)
            enes = np.array([p[1] for p in pts], dtype=float)
            x = plot_x(vols, system)
            emin = float(np.min(enes))
            y = plot_y_relative(enes, system, emin=emin)
            color = COMPOSITION_COLORS.get(system, "black")
            ax.plot(x, y, marker=marker, linestyle="None", color=color, markersize=6.5, markeredgewidth=0.7, label=SYSTEM_LABELS.get(system, system))
            if fit is not None:
                vgrid = np.linspace(vols.min(), vols.max(), 300)
                egrid = birch_murnaghan(vgrid, fit["E0_eV"], fit["B0_eV_per_A3"], fit["B0prime"], fit["V0_A3"])
                xfit = plot_x(vgrid, system)
                yfit = plot_y_relative(egrid, system, emin=emin)
                ax.plot(xfit, yfit, linestyle=linestyle, color=color, linewidth=1.7)

        ax.text(0.03, 0.95, panel_title, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=13)
        ax.set_xlabel(r"Volume per atom ($\AA^3$/atom)" if PLOT_VOLUME_PER_ATOM else r"Volume ($\AA^3$)", fontweight="bold")
        ax.set_ylabel(r"$E - E_{\min}$ (meV/atom)" if PLOT_ENERGY_MEV_PER_ATOM else r"$E - E_{\min}$ (eV)", fontweight="bold")
        ax.tick_params(direction="in", which="both")
        ax.grid(False)

    handles, labels, seen = [], [], set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll)
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="upper center", ncol=min(5, max(1, len(labels))), frameon=False, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIT_PLOT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_dft_only_fit_plots(fit_inputs, fit_results):
    old_panel_defs = PANEL_DEFS[:]
    # Retained for optional manual use: set MAKE_DFT_ONLY_FIT_PLOT = True if needed.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharex=True, sharey=True)
    for ax, model in zip(axes, ["PBE", "PBEsol"]):
        for system in SYSTEMS:
            key = (model, system, "DFT")
            pts = fit_inputs.get(key, [])
            fit = fit_results.get(key)
            if not pts:
                continue
            vols = np.array([p[0] for p in pts], dtype=float)
            enes = np.array([p[1] for p in pts], dtype=float)
            color = COMPOSITION_COLORS.get(system, "black")
            emin = float(np.min(enes))
            ax.plot(plot_x(vols, system), plot_y_relative(enes, system, emin=emin), marker=METHOD_MARKERS["DFT"], linestyle="None", color=color, markersize=6, label=SYSTEM_LABELS.get(system, system))
            if fit is not None:
                vgrid = np.linspace(vols.min(), vols.max(), 300)
                egrid = birch_murnaghan(vgrid, fit["E0_eV"], fit["B0_eV_per_A3"], fit["B0prime"], fit["V0_A3"])
                ax.plot(plot_x(vgrid, system), plot_y_relative(egrid, system, emin=emin), linestyle="-", color=color, linewidth=1.7)
        ax.text(0.03, 0.95, f"{model} DFT", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        ax.set_xlabel(r"Volume per atom ($\AA^3$/atom)" if PLOT_VOLUME_PER_ATOM else r"Volume ($\AA^3$)", fontweight="bold")
        ax.set_ylabel(r"$E - E_{\min}$ (meV/atom)" if PLOT_ENERGY_MEV_PER_ATOM else r"$E - E_{\min}$ (eV)", fontweight="bold")
        ax.tick_params(direction="in", which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, max(1, len(labels))), frameon=False, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(DFT_ONLY_FIT_PLOT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    all_rows = []
    fit_summary_rows = []
    fit_inputs = {}
    fit_results = {}

    print("\nEOS pointwise comparison")
    print("-" * 110)
    print(f"{'MODEL':<8} {'SYSTEM':<8} {'VF':>6} {'ΔE (eV)':>12} {'ΔE (meV/atom)':>16}")
    print("-" * 110)

    for model, root in ROOTS.items():
        for system in SYSTEMS:
            rows = collect_points(root, system)
            n = NATOMS[system]
            dft_pts = []
            dnp_pts = []
            for r in rows:
                deltaE = None
                deltaE_meV_atom = None
                if r["vasp_E"] is not None and r["dnp_E"] is not None:
                    deltaE = r["dnp_E"] - r["vasp_E"]
                    deltaE_meV_atom = 1000.0 * deltaE / n
                outrow = {
                    "model": model,
                    "system": system,
                    "folder": r["folder"],
                    "vf": r["vf"],
                    "vasp_E_eV": r["vasp_E"],
                    "vasp_V_A3": r["vasp_V"],
                    "dnp_E_eV": r["dnp_E"],
                    "dnp_V_A3": r["dnp_V"],
                    "deltaE_eV": deltaE,
                    "deltaE_meV_per_atom": deltaE_meV_atom,
                    "lmp_file": r["lmp_file"],
                }
                all_rows.append(outrow)
                if deltaE is not None:
                    print(f"{model:<8} {system:<8} {r['vf']:6.3f} {deltaE:12.6f} {deltaE_meV_atom:16.3f}")
                if r["vasp_E"] is not None and r["vasp_V"] is not None:
                    dft_pts.append((r["vasp_V"], r["vasp_E"]))
                if r["dnp_E"] is not None and r["dnp_V"] is not None:
                    dnp_pts.append((r["dnp_V"], r["dnp_E"]))

            fit_inputs[(model, system, "DFT")] = dft_pts
            fit_inputs[(model, system, "DNP")] = dnp_pts
            dft_fit = fit_bm([x for x, _ in dft_pts], [y for _, y in dft_pts])
            dnp_fit = fit_bm([x for x, _ in dnp_pts], [y for _, y in dnp_pts])
            fit_results[(model, system, "DFT")] = dft_fit
            fit_results[(model, system, "DNP")] = dnp_fit
            for method, fit in [("DFT", dft_fit), ("DNP", dnp_fit)]:
                if fit is not None:
                    fit_summary_rows.append({"model": model, "system": system, "method": method, **fit})

    write_pointwise_csv(all_rows, WRITE_POINT_CSV)
    write_fit_csv(fit_summary_rows, WRITE_FIT_CSV)

    print("\nBirch-Murnaghan fit summary")
    print("-" * 130)
    print(f"{'MODEL':<8} {'SYSTEM':<8} {'METHOD':<6} {'E0 (eV)':>12} {'V0 (A^3)':>12} {'B0 (eV/A^3)':>14} {'B0 (GPa)':>12} {'B0prime':>10}")
    print("-" * 130)
    for row in fit_summary_rows:
        print(
            f"{row['model']:<8} {row['system']:<8} {row['method']:<6} "
            f"{row['E0_eV']:12.6f} {row['V0_A3']:12.6f} "
            f"{row['B0_eV_per_A3']:14.6f} {row['B0_GPa']:12.3f} {row['B0prime']:10.4f}"
        )

    if MAKE_DELTAE_PLOTS:
        make_deltae_plots(all_rows)
    if MAKE_EOS_FIT_PLOTS:
        make_fit_plots(fit_inputs, fit_results)
    if MAKE_DFT_ONLY_FIT_PLOT:
        make_dft_only_fit_plots(fit_inputs, fit_results)

    print(f"\nWrote {WRITE_POINT_CSV}")
    print(f"Wrote {WRITE_FIT_CSV}")
    if MAKE_DELTAE_PLOTS:
        print(f"Wrote {DELTAE_PLOT}")
    if MAKE_EOS_FIT_PLOTS:
        print(f"Wrote {FIT_PLOT}")
    if MAKE_DFT_ONLY_FIT_PLOT:
        print(f"Wrote {DFT_ONLY_FIT_PLOT}")
    if curve_fit is None:
        print("\nWARNING: scipy not found, so no Birch-Murnaghan fits were performed.")


if __name__ == "__main__":
    main()
