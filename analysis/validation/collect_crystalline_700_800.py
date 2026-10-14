#!/usr/bin/env python3

"""
Collect finite-temperature enthalpies for ordered crystalline Cu-Ag.

Campaign:
    Enthalpy/crystalline_ordered/
        PBE, PBEsol
        Cu_0, Cu_25, Cu_50, Cu_75, Cu_100
        T900, T1000, T1200, T1300

Reads the final completed LAMMPS thermo block from enthalpy_out.
Uses its final 50% for averaging.

Outputs:
    collected_crystalline_ordered/
        run_status.csv
        enthalpy_summary.csv
        mixing_enthalpy.csv
        functional_comparison.csv
        collection_report.txt

Assumptions:
    units metal
    LAMMPS pressure in bar
    energy in eV
    volume in Angstrom^3
    6912 atoms

No result is accepted from an incomplete LAMMPS run.
"""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


FUNCTIONALS = ("PBE", "PBEsol")
COMPOSITIONS = (0, 25, 50, 75, 100)
TEMPERATURES = (700, 800)

NATOMS = 6912

# 1 bar * 1 Angstrom^3, converted to eV.
BAR_A3_TO_EV = 6.241509074e-7

# Number of contiguous blocks for a simple block-SEM estimate.
NBLOCKS = 10


def write_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_thermo(path):
    """
    Extract completed standard LAMMPS thermo tables.

    Each table must have a Step column and either:
      Enthalpy
    or:
      TotEng, Press, Volume

    Only tables terminated by a 'Loop time' line are accepted.
    """

    blocks = []

    header = None
    current = []

    def finish(completed):
        nonlocal header, current

        if header is not None and current:
            blocks.append({
                "header": header,
                "rows": current,
                "completed": completed,
            })

        header = None
        current = []

    with path.open(errors="replace") as f:

        for line in f:

            s = line.strip()

            if s.startswith("Loop time"):
                finish(True)
                continue

            fields = s.split()

            if not fields:
                continue

            # Standard LAMMPS thermo table header.
            if fields[0] == "Step" and len(fields) >= 2:
                finish(False)
                header = fields
                continue

            if header is None:
                continue

            if len(fields) != len(header):
                continue

            try:
                values = [float(x) for x in fields]
            except ValueError:
                continue

            if all(math.isfinite(x) for x in values):
                current.append(values)

    # A table reaching EOF without Loop time is incomplete.
    finish(False)

    valid = [
        block
        for block in blocks
        if block["completed"]
    ]

    if not valid:
        raise ValueError(
            "No completed LAMMPS thermo block found"
        )

    # The last completed block is expected to be production.
    block = valid[-1]

    return (
        block["header"],
        np.asarray(block["rows"], dtype=float),
        len(valid),
    )


def column(header, data, names):
    """Find a thermo column using case-insensitive matching."""

    lower = [x.lower() for x in header]

    for name in names:
        if name.lower() in lower:
            return data[:, lower.index(name.lower())]

    return None


def extract_enthalpy(header, data):

    h = column(
        header,
        data,
        ("Enthalpy",),
    )

    if h is not None:
        return h / NATOMS, "Enthalpy"

    # Fallback: H = Etot + PV.
    energy = column(
        header,
        data,
        ("TotEng",),
    )

    pressure = column(
        header,
        data,
        ("Press",),
    )

    volume = column(
        header,
        data,
        ("Volume", "Vol"),
    )

    if (
        energy is None
        or pressure is None
        or volume is None
    ):
        raise ValueError(
            "Missing Enthalpy or TotEng/Press/Volume "
            f"thermo columns. Found: {header}"
        )

    h = energy + pressure * volume * BAR_A3_TO_EV

    return h / NATOMS, "TotEng+PV"


def block_sem(values, nblocks=NBLOCKS):

    blocks = np.array_split(values, nblocks)

    means = np.asarray([
        np.mean(b)
        for b in blocks
        if len(b) >= 2
    ])

    if len(means) < 3:
        raise ValueError(
            "Insufficient data for block SEM"
        )

    return (
        np.std(means, ddof=1)
        / np.sqrt(len(means))
    )


def process_run(folder, target_temperature, tail_fraction):

    logfile = folder / "enthalpy_out"

    if not logfile.is_file():
        raise FileNotFoundError(
            f"Missing enthalpy_out: {logfile}"
        )

    if logfile.stat().st_size == 0:
        raise ValueError("Empty enthalpy_out")

    header, data, completed_blocks = parse_thermo(logfile)

    if len(data) < 40:
        raise ValueError(
            f"Too few thermo samples: {len(data)}"
        )

    step = column(header, data, ("Step",))
    temp = column(header, data, ("Temp",))
    press = column(header, data, ("Press",))

    if step is None or temp is None:
        raise ValueError(
            "Thermo table must contain Step and Temp"
        )

    if np.any(np.diff(step) <= 0):
        raise ValueError(
            "Non-increasing timestep sequence"
        )

    h, method = extract_enthalpy(header, data)

    # Analyze only the final fraction of the final
    # completed thermo block.
    start = int(len(data) * (1.0 - tail_fraction))

    h = h[start:]
    temp = temp[start:]
    step = step[start:]

    if press is not None:
        press = press[start:]

    if len(h) < 30:
        raise ValueError(
            "Too few production samples"
        )

    h_mean = float(np.mean(h))
    h_sem = float(block_sem(h))

    t_mean = float(np.mean(temp))

    p_mean = (
        float(np.mean(press))
        if press is not None
        else float("nan")
    )

    # Simple stationarity diagnostic:
    # compare the first and last third of the
    # selected averaging window.
    third = len(h) // 3

    drift = float(
        np.mean(h[-third:])
        - np.mean(h[:third])
    )

    # Check average temperature.
    if abs(t_mean - target_temperature) > 75:
        raise ValueError(
            f"Temperature mismatch: "
            f"target={target_temperature}, "
            f"observed={t_mean:.2f} K"
        )

    if (
        math.isfinite(p_mean)
        and abs(p_mean) > 2000
    ):
        raise ValueError(
            f"Large mean pressure: {p_mean:.2f} bar"
        )

    # Treat conspicuous drift as a warning rather
    # than silently accepting the result.
    drift_warning = (
        abs(drift) > max(
            0.005,        # 5 meV/atom
            5.0 * h_sem,
        )
    )

    return {
        "H_eV_atom": h_mean,
        "H_SEM_eV_atom": h_sem,
        "H_SEM_meV_atom": 1000.0 * h_sem,
        "T_mean_K": t_mean,
        "P_mean_bar": p_mean,
        "drift_meV_atom": 1000.0 * drift,
        "drift_warning": drift_warning,
        "n_samples": len(h),
        "step_start": int(step[0]),
        "step_end": int(step[-1]),
        "completed_thermo_blocks": completed_blocks,
        "enthalpy_method": method,
        "thermo_header": " ".join(header),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Enthalpy/crystalline_ordered"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("collected_crystalline_700_800"),
    )

    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.50,
    )

    args = parser.parse_args()

    if not 0 < args.tail_fraction <= 1:
        parser.error(
            "--tail-fraction must be between 0 and 1"
        )

    if not args.root.is_dir():
        parser.error(
            f"Campaign directory missing: {args.root}"
        )

    args.out.mkdir(parents=True, exist_ok=True)

    status_rows = []
    summary_rows = []

    accepted = {}

    print("=" * 76)
    print("CRYSTALLINE ORDERED Cu-Ag ENTHALPY COLLECTOR")
    print("=" * 76)

    # --------------------------------------------------
    # 1. Read all 40 runs.
    # --------------------------------------------------

    for functional in FUNCTIONALS:

        for T in TEMPERATURES:

            for comp in COMPOSITIONS:

                folder = (
                    args.root
                    / functional
                    / f"Cu_{comp}"
                    / f"T{T}"
                )

                key = (functional, T, comp)

                try:

                    result = process_run(
                        folder,
                        T,
                        args.tail_fraction,
                    )

                    if result["drift_warning"]:
                        state = "WARNING_DRIFT"
                    else:
                        state = "COMPLETE"

                    # Only stationary results are used for mixing.
                    if state == "COMPLETE":
                        accepted[key] = result

                    summary_rows.append({
                        "functional": functional,
                        "T_target_K": T,
                        "Cu_at_percent": comp,
                        "status": state,
                        "directory": str(folder),
                        **result,
                    })

                except Exception as exc:

                    state = f"PENDING_OR_INVALID: {exc}"

                status_rows.append({
                    "functional": functional,
                    "T_K": T,
                    "Cu_at_percent": comp,
                    "status": state,
                    "directory": str(folder),
                })

                print(
                    f"{functional:<7} "
                    f"T{T:<5} "
                    f"Cu_{comp:<3} "
                    f"{state}"
                )

    write_csv(
        args.out / "run_status.csv",
        status_rows,
        [
            "functional",
            "T_K",
            "Cu_at_percent",
            "status",
            "directory",
        ],
    )

    summary_columns = [
        "functional",
        "T_target_K",
        "Cu_at_percent",
        "status",
        "H_eV_atom",
        "H_SEM_eV_atom",
        "H_SEM_meV_atom",
        "T_mean_K",
        "P_mean_bar",
        "drift_meV_atom",
        "drift_warning",
        "n_samples",
        "step_start",
        "step_end",
        "completed_thermo_blocks",
        "enthalpy_method",
        "thermo_header",
        "directory",
    ]

    write_csv(
        args.out / "enthalpy_summary.csv",
        summary_rows,
        summary_columns,
    )

    # --------------------------------------------------
    # 2. Mixing enthalpy.
    # --------------------------------------------------

    mixing_rows = []

    for functional in FUNCTIONALS:

        for T in TEMPERATURES:

            ag = accepted.get((functional, T, 0))
            cu = accepted.get((functional, T, 100))

            if ag is None or cu is None:

                print(
                    f"SKIP mixing: {functional} T{T} "
                    "missing valid elemental reference"
                )

                continue

            for comp in (25, 50, 75):

                alloy = accepted.get(
                    (functional, T, comp)
                )

                if alloy is None:
                    continue

                x = comp / 100.0

                h_mix = (
                    alloy["H_eV_atom"]
                    - (1-x) * ag["H_eV_atom"]
                    - x * cu["H_eV_atom"]
                )

                # Independent-run uncertainty propagation.
                sem = math.sqrt(
                    alloy["H_SEM_eV_atom"]**2
                    + (1-x)**2
                    * ag["H_SEM_eV_atom"]**2
                    + x**2
                    * cu["H_SEM_eV_atom"]**2
                )

                mixing_rows.append({
                    "functional": functional,
                    "T_K": T,
                    "Cu_at_percent": comp,
                    "H_alloy_eV_atom":
                        alloy["H_eV_atom"],
                    "H_Ag_eV_atom":
                        ag["H_eV_atom"],
                    "H_Cu_eV_atom":
                        cu["H_eV_atom"],
                    "Delta_Hmix_meV_atom":
                        1000.0 * h_mix,
                    "Delta_Hmix_SEM_meV_atom":
                        1000.0 * sem,
                })

    mix_columns = [
        "functional",
        "T_K",
        "Cu_at_percent",
        "H_alloy_eV_atom",
        "H_Ag_eV_atom",
        "H_Cu_eV_atom",
        "Delta_Hmix_meV_atom",
        "Delta_Hmix_SEM_meV_atom",
    ]

    write_csv(
        args.out / "mixing_enthalpy.csv",
        mixing_rows,
        mix_columns,
    )

    # --------------------------------------------------
    # 3. PBE versus PBEsol.
    # --------------------------------------------------

    indexed = {
        (r["functional"], r["T_K"], r["Cu_at_percent"]): r
        for r in mixing_rows
    }

    comparisons = []

    for T in TEMPERATURES:

        for comp in (25, 50, 75):

            pbe = indexed.get(
                ("PBE", T, comp)
            )

            pbesol = indexed.get(
                ("PBEsol", T, comp)
            )

            if pbe is None or pbesol is None:
                continue

            difference = (
                pbe["Delta_Hmix_meV_atom"]
                - pbesol["Delta_Hmix_meV_atom"]
            )

            sem = math.sqrt(
                pbe["Delta_Hmix_SEM_meV_atom"]**2
                + pbesol["Delta_Hmix_SEM_meV_atom"]**2
            )

            comparisons.append({
                "T_K": T,
                "Cu_at_percent": comp,
                "PBE_Delta_Hmix_meV_atom":
                    pbe["Delta_Hmix_meV_atom"],
                "PBEsol_Delta_Hmix_meV_atom":
                    pbesol["Delta_Hmix_meV_atom"],
                "PBE_minus_PBEsol_meV_atom":
                    difference,
                "Difference_SEM_meV_atom": sem,
            })

    write_csv(
        args.out / "functional_comparison.csv",
        comparisons,
        [
            "T_K",
            "Cu_at_percent",
            "PBE_Delta_Hmix_meV_atom",
            "PBEsol_Delta_Hmix_meV_atom",
            "PBE_minus_PBEsol_meV_atom",
            "Difference_SEM_meV_atom",
        ],
    )

    # --------------------------------------------------
    # 4. Report.
    # --------------------------------------------------

    n_complete = sum(
        r["status"] == "COMPLETE"
        for r in status_rows
    )

    n_drift = sum(
        r["status"] == "WARNING_DRIFT"
        for r in status_rows
    )

    report = [
        "Crystalline ordered Cu-Ag enthalpy collection",
        "=" * 55,
        f"Campaign: {args.root}",
        f"Accepted runs: {n_complete}/20",
        f"Drift warnings: {n_drift}",
        f"Mixing enthalpy rows: {len(mixing_rows)}/12",
        f"Functional comparison rows: {len(comparisons)}/6",
        f"Tail fraction: {args.tail_fraction}",
        f"Number of blocks for SEM: {NBLOCKS}",
        "",
        "IMPORTANT:",
        "The final completed LAMMPS thermo block is assumed",
        "to correspond to production.",
        "The final fraction of that block is averaged.",
        "Block SEM is an approximate sampling uncertainty;",
        "it does not incorporate independent-run variability.",
        "Crystalline order and phase retention are NOT verified.",
        "Do not report transformed or melted structures as",
        "ordered-crystalline mixing enthalpies.",
        "These finite-T enthalpies are not a static-energy",
        "baseline for direct addition to harmonic Fvib.",
    ]

    (args.out / "collection_report.txt").write_text(
        "\n".join(report) + "\n"
    )

    print()
    print("=" * 76)
    print("COLLECTION FINISHED")
    print("=" * 76)
    print(f"Accepted runs: {n_complete}/20")
    print(f"Drift warnings: {n_drift}")
    print(f"Mixing results: {len(mixing_rows)}/12")
    print(f"Functional comparisons: {len(comparisons)}/6")
    print(f"Output directory: {args.out}")
    print("=" * 76)



# === CRYSTALLINE PHASE FILTER V1 ===

def apply_phase_filter(outdir):

    import csv
    import math

    outdir = Path(outdir)

    # All classifications correspond to the user's
    # visual inspection of the final configurations.
    #
    # Pure-element references remained crystalline
    # for both functionals at all four temperatures.
    #
    # All Cu50 alloys melted.
    #
    # Exceptions to otherwise ordered Cu25/Cu75:
    #   PBEsol Cu25: 1200, 1300 K disordered
    #   PBE Cu75:    1300 K disordered
    #   PBEsol Cu75: 1300 K disordered

    functionals = ("PBE", "PBEsol")
    temperatures = (900, 1000, 1200, 1300)
    compositions = (0, 25, 50, 75, 100)

    phase = {}

    for functional in functionals:
        for T in temperatures:
            for comp in compositions:

                key = (functional, T, comp)

                if comp in (0, 100):
                    state = "CRYSTALLINE_ELEMENT"

                elif comp == 50:
                    state = "MELTED"

                else:
                    state = "ORDERED_CRYSTAL"

                phase[key] = state

    for T in (1200, 1300):
        phase[("PBEsol", T, 25)] = "DISORDERED_CRYSTAL"

    phase[("PBE", 1300, 75)] = "DISORDERED_CRYSTAL"
    phase[("PBEsol", 1300, 75)] = "DISORDERED_CRYSTAL"

    assert len(phase) == 40
    assert sum(
        value == "ORDERED_CRYSTAL"
        for value in phase.values()
    ) == 12
    assert sum(
        value == "DISORDERED_CRYSTAL"
        for value in phase.values()
    ) == 4
    assert sum(
        value == "MELTED"
        for value in phase.values()
    ) == 8

    def read_csv(name):
        path = outdir / name

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    def write_csv(name, rows, columns):

        with (outdir / name).open("w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=columns,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

    # --------------------------------------------------
    # Load the ORIGINAL numerical collection.
    # --------------------------------------------------

    status = read_csv("run_status.csv")
    enthalpy = read_csv("enthalpy_summary.csv")
    mixing = read_csv("mixing_enthalpy.csv")
    comparison = read_csv("functional_comparison.csv")

    if len(status) != 40:
        raise RuntimeError(
            f"Expected 40 run-status rows; found {len(status)}"
        )

    if len(enthalpy) != 40:
        raise RuntimeError(
            f"Expected 40 enthalpy rows; found {len(enthalpy)}"
        )

    if len(mixing) != 24:
        raise RuntimeError(
            f"Expected 24 mixing rows; found {len(mixing)}. "
            "Complete the numerical collection first."
        )

    if len(comparison) != 12:
        raise RuntimeError(
            f"Expected 12 functional comparisons; "
            f"found {len(comparison)}"
        )

    def key(row, temperature_column):
        return (
            row["functional"],
            int(float(row[temperature_column])),
            int(float(row["Cu_at_percent"])),
        )

    # Verify full coverage and unique keys before
    # generating any phase-filtered files.

    status_keys = [
        key(row, "T_K")
        for row in status
    ]

    enthalpy_keys = [
        key(row, "T_target_K")
        for row in enthalpy
    ]

    if (
        len(set(status_keys)) != 40
        or set(status_keys) != set(phase)
    ):
        raise RuntimeError(
            "Status keys do not match the 40 "
            "structurally classified calculations."
        )

    if (
        len(set(enthalpy_keys)) != 40
        or set(enthalpy_keys) != set(phase)
    ):
        raise RuntimeError(
            "Enthalpy keys do not match the "
            "40 structurally classified calculations."
        )

    # --------------------------------------------------
    # Complete phase-state table.
    # --------------------------------------------------

    phase_rows = []

    for row in status:

        k = key(row, "T_K")
        state = phase[k]

        numeric_ok = row["status"] == "COMPLETE"

        if state == "ORDERED_CRYSTAL":
            category = "ORDERED_ALLOY"

        elif state == "CRYSTALLINE_ELEMENT":
            category = "ELEMENT_REFERENCE"

        else:
            category = "EXCLUDED_FROM_ORDERED_MIXING"

        phase_rows.append({
            **row,
            "phase_state": state,
            "analysis_category": category,
            "numerically_complete": numeric_ok,
            "visual_inspection": "FINAL_CONFIGURATION",
            "production_window_verified": False,
        })

    write_csv(
        "phase_validation.csv",
        phase_rows,
        [
            "functional",
            "T_K",
            "Cu_at_percent",
            "status",
            "phase_state",
            "analysis_category",
            "numerically_complete",
            "visual_inspection",
            "production_window_verified",
            "directory",
        ],
    )

    # --------------------------------------------------
    # Annotate all original mixing enthalpies.
    # Preserve every numerical value.
    # --------------------------------------------------

    annotated = []
    ordered = []
    excluded = []

    for row in mixing:

        k = key(row, "T_K")
        state = phase[k]

        functional, T, comp = k

        # Both pure-element references must remain
        # crystalline for an ordered-alloy comparison.

        ag_state = phase[(functional, T, 0)]
        cu_state = phase[(functional, T, 100)]

        refs_valid = (
            ag_state == "CRYSTALLINE_ELEMENT"
            and cu_state == "CRYSTALLINE_ELEMENT"
        )

        eligible = (
            state == "ORDERED_CRYSTAL"
            and refs_valid
        )

        if eligible:
            reason = "ORDERED_ALLOY_AND_CRYSTALLINE_REFERENCES"

        elif state == "MELTED":
            reason = "ALLOY_MELTED"

        elif state == "DISORDERED_CRYSTAL":
            reason = "CHEMICAL_ORDER_LOST"

        else:
            reason = "INVALID_REFERENCE_OR_UNCLASSIFIED"

        record = {
            **row,
            "phase_state": state,
            "Ag_reference_state": ag_state,
            "Cu_reference_state": cu_state,
            "ordered_analysis_eligible": eligible,
            "exclusion_reason": "" if eligible else reason,
            "production_window_verified": False,
        }

        annotated.append(record)

        if eligible:
            ordered.append(record)
        else:
            excluded.append(record)

    original_columns = list(mixing[0].keys())

    extra_columns = [
        "phase_state",
        "Ag_reference_state",
        "Cu_reference_state",
        "ordered_analysis_eligible",
        "exclusion_reason",
        "production_window_verified",
    ]

    columns = original_columns + extra_columns

    write_csv(
        "mixing_enthalpy_phase_annotated.csv",
        annotated,
        columns,
    )

    write_csv(
        "mixing_enthalpy_ordered_only.csv",
        ordered,
        columns,
    )

    write_csv(
        "mixing_enthalpy_excluded.csv",
        excluded,
        columns,
    )

    # --------------------------------------------------
    # Functional comparison.
    # Require BOTH PBE and PBEsol ordered.
    # --------------------------------------------------

    valid_keys = {
        key(row, "T_K")
        for row in ordered
    }

    matched = []
    excluded_comparisons = []

    for row in comparison:

        T = int(float(row["T_K"]))
        comp = int(float(row["Cu_at_percent"]))

        pbe_ok = ("PBE", T, comp) in valid_keys
        pbesol_ok = ("PBEsol", T, comp) in valid_keys

        record = {
            **row,
            "PBE_phase": phase[("PBE", T, comp)],
            "PBEsol_phase": phase[("PBEsol", T, comp)],
            "both_ordered": pbe_ok and pbesol_ok,
            "production_window_verified": False,
        }

        if pbe_ok and pbesol_ok:
            matched.append(record)
        else:
            excluded_comparisons.append(record)

    comparison_columns = (
        list(comparison[0].keys())
        + [
            "PBE_phase",
            "PBEsol_phase",
            "both_ordered",
            "production_window_verified",
        ]
    )

    write_csv(
        "functional_comparison_ordered_only.csv",
        matched,
        comparison_columns,
    )

    write_csv(
        "functional_comparison_excluded.csv",
        excluded_comparisons,
        comparison_columns,
    )

    # --------------------------------------------------
    # Assertions: fail loudly if filtering changes.
    # --------------------------------------------------

    if len(ordered) != 12:
        raise RuntimeError(
            f"Expected 12 ordered mixing rows; "
            f"found {len(ordered)}"
        )

    if len(excluded) != 12:
        raise RuntimeError(
            f"Expected 12 excluded mixing rows; "
            f"found {len(excluded)}"
        )

    if len(matched) != 5:
        raise RuntimeError(
            f"Expected 5 matched functional comparisons; "
            f"found {len(matched)}"
        )

    # --------------------------------------------------
    # Separate structural-validation report.
    # --------------------------------------------------

    lines = [
        "Cu-Ag ordered-crystalline structural validation",
        "=" * 60,
        "Numerically completed runs: 40/40",
        "Crystalline elemental references: 16/16",
        "Ordered alloy states: 12/24",
        "Disordered alloy states: 4/24",
        "Melted alloy states: 8/24",
        "",
        "Ordered mixing enthalpies retained: "
        f"{len(ordered)}/24",
        "Matched functional comparisons retained: "
        f"{len(matched)}/12",
        "",
        "Validated ordered-alloy coverage:",
    ]

    for functional in functionals:

        for T in temperatures:

            comps = [
                comp
                for comp in (25, 50, 75)
                if (functional, T, comp) in valid_keys
            ]

            lines.append(
                f"  {functional} T{T}: {comps}"
            )

    lines += [
        "",
        "IMPORTANT LIMITATIONS:",
        "Phase classifications come from visual inspection",
        "of final configurations.",
        "Structural state over the entire production",
        "averaging window has not been verified.",
        "Classifications alone do not establish",
        "equilibrium phase stability.",
        "A crystalline elemental state above its melting",
        "temperature may be metastable.",
        "The original numerical enthalpy and mixing",
        "files have not been modified.",
        "Do not add finite-temperature mixing enthalpy",
        "directly to harmonic vibrational free energy.",
    ]

    (outdir / "phase_validation_report.txt").write_text(
        "\n".join(lines) + "\n"
    )

    print()
    print("=" * 76)
    print("STRUCTURAL PHASE FILTER")
    print("=" * 76)
    print("Ordered mixing enthalpies: 12 / 24")
    print("Excluded alloy states:     12 / 24")
    print("  Melted:                    8")
    print("  Disordered crystalline:    4")
    print("Matched PBE/PBEsol:          5 / 12")
    print()
    print("Output:", outdir)
    print("=" * 76)

# === END CRYSTALLINE PHASE FILTER V1 ===


if __name__ == "__main__":
    main()
