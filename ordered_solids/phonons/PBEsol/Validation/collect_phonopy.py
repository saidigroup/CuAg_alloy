#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

CAMPAIGNS = {
    "4x4x4": ROOT / "PBEsol_phonopy_4x4x4",
    "6x6x6": ROOT / "PBEsol_phonopy_6x6x6",
}

COMPS = [0, 25, 50, 75, 100]
SELECTED_T = [900, 970, 1000, 1100, 1200, 1400, 1500]

OUT = ROOT / "collected_phonopy"
OUT.mkdir(exist_ok=True)


def save_csv(filename, rows):
    if not rows:
        return

    with open(OUT / filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(label, comp):
    folder = CAMPAIGNS[label] / f"Cu_{comp}"
    meta_file = folder / "phonopy_metadata.json"
    data_file = folder / "thermal_properties.dat"

    if not meta_file.is_file() or not data_file.is_file():
        return None, "MISSING"

    try:
        with open(meta_file) as f:
            meta = json.load(f)

        data = np.loadtxt(data_file, comments="#", ndmin=2)

        if data.shape[1] != 7:
            raise ValueError(
                f"Expected 7 columns, found {data.shape[1]}"
            )

        if not np.all(np.isfinite(data)):
            raise ValueError("Nonfinite thermal-property values")

        if not np.all(np.diff(data[:, 0]) > 0):
            raise ValueError("Temperature grid is not increasing")

        expected_dim = int(label[0])

        if meta["supercell_matrix"] != [expected_dim] * 3:
            raise ValueError("Incorrect supercell dimensions")

        if meta["reference_atoms"] * expected_dim**3 != \
                meta["supercell_atoms"]:
            raise ValueError("Supercell atom-count mismatch")

        if abs(meta["Cu_at_percent"] - comp) > 0.01:
            raise ValueError("Composition mismatch")

        return {
            "metadata": meta,
            "data": data,
            "folder": folder,
        }, "COMPLETE"

    except Exception as exc:
        return None, f"INVALID: {exc}"


def same_grid(a, b):
    return (
        a.shape == b.shape
        and np.allclose(a[:, 0], b[:, 0], atol=1e-8, rtol=0)
    )


runs = {}
status_rows = []

print("=" * 75)
print("Cu-Ag PBEsol phonon collection")
print("=" * 75)

for label in CAMPAIGNS:
    for comp in COMPS:
        result, status = load_run(label, comp)

        runs[(label, comp)] = result

        row = {
            "supercell": label,
            "Cu_at_percent": comp,
            "status": status,
            "reference_atoms": "",
            "supercell_atoms": "",
            "displacements": "",
            "minimum_frequency_THz": "",
            "imaginary_modes_below_minus_0p1_THz": "",
            "model": "",
        }

        if result is not None:
            m = result["metadata"]
            row.update({
                "reference_atoms": m["reference_atoms"],
                "supercell_atoms": m["supercell_atoms"],
                "displacements": m["number_of_displacements"],
                "minimum_frequency_THz":
                    m["minimum_frequency_THz"],
                "imaginary_modes_below_minus_0p1_THz":
                    m["number_modes_below_minus_0p10_THz"],
                "model": m["model"],
            })

        status_rows.append(row)

        print(f"{label:6s} Cu_{comp:<3d}: {status}")

save_csv("run_status.csv", status_rows)


# ------------------------------------------------------------
# Check model consistency.
# Do not assume differently named model paths are identical.
# ------------------------------------------------------------

models = {
    result["metadata"]["model"]
    for result in runs.values()
    if result is not None
}

model_consistent = len(models) == 1

print("\nPotential paths:")

for model in sorted(models):
    print(" ", model)

if not model_consistent:
    print(
        "\nWARNING: Different model paths detected. "
        "Verify that they refer to the same potential before "
        "interpreting supercell convergence."
    )


# ------------------------------------------------------------
# Collect full thermal-property tables.
# All quantities below are per atom except where noted.
#
# Input columns:
# 0 T (K)
# 1 F (kJ/mol reference cell)
# 2 F (eV/atom)
# 3 S (J/mol reference cell/K)
# 4 S (J/mol atom/K)
# 5 Cv (J/mol reference cell/K)
# 6 Cv (J/mol atom/K)
# ------------------------------------------------------------

thermal_rows = []

for (label, comp), result in runs.items():
    if result is None:
        continue

    for line in result["data"]:
        thermal_rows.append({
            "supercell": label,
            "Cu_at_percent": comp,
            "T_K": line[0],
            "Fvib_eV_atom": line[2],
            "Svib_J_mol_atom_K": line[4],
            "Cv_J_mol_atom_K": line[6],
        })

save_csv("thermal_properties_all.csv", thermal_rows)


# ------------------------------------------------------------
# Vibrational mixing properties.
#
# Ag endpoint = Cu_0
# Cu endpoint = Cu_100
#
# Delta F = F(x) - (1-x) F(Ag) - x F(Cu)
#
# Same convention for entropy and heat capacity.
# ------------------------------------------------------------

mix_rows = []

mixing_by_campaign = {}

for label in CAMPAIGNS:
    ag = runs.get((label, 0))
    cu = runs.get((label, 100))

    if ag is None or cu is None:
        print(
            f"\n{label}: Cannot calculate mixing properties: "
            "missing endpoint."
        )
        continue

    a = ag["data"]
    c = cu["data"]

    if not same_grid(a, c):
        print(f"\n{label}: Endpoint temperature grids differ.")
        continue

    for comp in [25, 50, 75]:
        result = runs.get((label, comp))

        if result is None:
            continue

        d = result["data"]

        if not same_grid(a, d):
            print(
                f"{label} Cu_{comp}: Temperature grid mismatch."
            )
            continue

        x = comp / 100.0

        df = d[:, 2] - (1-x)*a[:, 2] - x*c[:, 2]
        ds = d[:, 4] - (1-x)*a[:, 4] - x*c[:, 4]
        dcv = d[:, 6] - (1-x)*a[:, 6] - x*c[:, 6]

        mixing_by_campaign[(label, comp)] = {
            "T": d[:, 0],
            "F": df,
            "S": ds,
            "Cv": dcv,
        }

        for i, T in enumerate(d[:, 0]):
            mix_rows.append({
                "supercell": label,
                "Cu_at_percent": comp,
                "T_K": T,
                "Delta_Fvib_mix_meV_atom": 1000*df[i],
                "Delta_Svib_mix_J_mol_atom_K": ds[i],
                "Delta_Cv_mix_J_mol_atom_K": dcv[i],
            })

save_csv("vibrational_mixing_all.csv", mix_rows)


# ------------------------------------------------------------
# Compare 4x4x4 and 6x6x6 results.
#
# These are numerical differences only. Model consistency is
# reported independently, and must be confirmed.
# ------------------------------------------------------------

conv_rows = []

for comp in COMPS:
    r4 = runs.get(("4x4x4", comp))
    r6 = runs.get(("6x6x6", comp))

    if r4 is None or r6 is None:
        continue

    a = r4["data"]
    b = r6["data"]

    if not same_grid(a, b):
        print(
            f"Cu_{comp}: Cannot compare supercells: "
            "temperature grids differ."
        )
        continue

    for i, T in enumerate(a[:, 0]):
        conv_rows.append({
            "Cu_at_percent": comp,
            "T_K": T,
            "Fvib_4x4x4_meV_atom": 1000*a[i, 2],
            "Fvib_6x6x6_meV_atom": 1000*b[i, 2],
            "Delta_F_6minus4_meV_atom":
                1000*(b[i, 2]-a[i, 2]),
            "Delta_S_6minus4_J_mol_atom_K":
                b[i, 4]-a[i, 4],
        })

save_csv("supercell_convergence_all.csv", conv_rows)


# ------------------------------------------------------------
# Convergence of vibrational MIXING contributions.
# This is more relevant than absolute F convergence alone.
# ------------------------------------------------------------

mix_conv_rows = []

for comp in [25, 50, 75]:
    a = mixing_by_campaign.get(("4x4x4", comp))
    b = mixing_by_campaign.get(("6x6x6", comp))

    if a is None or b is None:
        continue

    if not np.array_equal(a["T"], b["T"]):
        continue

    for i, T in enumerate(a["T"]):
        mix_conv_rows.append({
            "Cu_at_percent": comp,
            "T_K": T,
            "Delta_Fmix_4x4x4_meV_atom": 1000*a["F"][i],
            "Delta_Fmix_6x6x6_meV_atom": 1000*b["F"][i],
            "Difference_6minus4_meV_atom":
                1000*(b["F"][i]-a["F"][i]),
            "Delta_Smix_4x4x4_J_mol_atom_K": a["S"][i],
            "Delta_Smix_6x6x6_J_mol_atom_K": b["S"][i],
            "Difference_S_6minus4_J_mol_atom_K":
                b["S"][i]-a["S"][i],
        })

save_csv(
    "mixing_convergence_all.csv",
    mix_conv_rows,
)


# ------------------------------------------------------------
# Selected-temperature summary.
# ------------------------------------------------------------

selected_rows = []

for row in mix_rows:
    if any(abs(row["T_K"] - T) < 0.001 for T in SELECTED_T):
        selected_rows.append(row)

save_csv(
    "selected_temperature_summary.csv",
    selected_rows,
)


# ------------------------------------------------------------
# Write report.
# ------------------------------------------------------------

report = [
    "Cu-Ag PBEsol phonon collection",
    "================================",
    "",
    f"Potential-path consistency: {model_consistent}",
    "",
]

report.extend(
    f"{r['supercell']} Cu_{r['Cu_at_percent']}: {r['status']}"
    for r in status_rows
)

report.extend([
    "",
    "Note: Vibrational mixing properties are harmonic",
    "contributions for the supplied solid structures.",
    "They do not include configurational entropy,",
    "anharmonicity, or liquid free energies.",
    "",
    "Supercell convergence must be assessed using",
    "both absolute and mixing vibrational properties.",
])

(OUT / "collection_report.txt").write_text(
    "\n".join(report) + "\n"
)

print("\n" + "=" * 75)
print("Collection completed")
print("=" * 75)
print("Results directory:", OUT)
print("Complete calculations:", sum(
    result is not None for result in runs.values()
))
print("Missing/invalid calculations:", sum(
    result is None for result in runs.values()
))
print("Potential paths consistent:", model_consistent)
