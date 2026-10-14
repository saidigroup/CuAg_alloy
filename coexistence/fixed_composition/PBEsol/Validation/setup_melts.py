#!/usr/bin/env python3
"""
generate_binaries.py

Creates LAMMPS-ready directories for selected binary fcc systems at 10 at.% steps.
- Structure: 3x3x3 conventional fcc supercell (4 atoms/cell * 27 = 108 atoms)
- Occupations: random solid solution matching exact composition
- Lattice: Vegard's law between endmembers (room-T fcc lattice params)
- Outputs: data.lmp (atomic style), composition.txt; copies /REFS/melt.in and /REFS/submit_gpu

Usage:
  python generate_binaries.py
  # optional flags:
  #   --seed 1234            # random seed (default 12345)
  #   --root OUTDIR          # where to create pair folders (default = current dir)
  #   --dry-run              # show what would be created
"""

import argparse
import math
import os
import random
import shutil
from pathlib import Path

# ------------------------------
# Settings & reference data
# ------------------------------

# fcc lattice parameters at ~room temperature (Å).
# Adjust if you prefer your own references.
FCC_A = {
    "Ag": 4.0862,
    "Au": 4.0782,
    "Cu": 3.6150,
    "Pd": 3.8891,
    "Pt": 3.9231,
}

# Atomic masses (g/mol) for Masses section (not strictly needed for "atomic" style, but helpful)
AMASS = {
    "Ag": 107.8682,
    "Au": 196.96657,
    "Cu": 63.546,
    "Pd": 106.42,
    "Pt": 195.084,
}

# Atomic numbers (for consistent type ordering: type 1 = lower Z)
ZNUM = {
    "Ag": 47,
    "Au": 79,
    "Cu": 29,
    "Pd": 46,
    "Pt": 78,
}

# The 10 binary systems you asked for, as (A, B)
PAIRS = [
    ("Cu", "Ag"),
    ("Cu", "Pd"),
    ("Cu", "Pt"),
    ("Cu", "Au"),
    ("Ag", "Pd"),
    ("Ag", "Pt"),
    ("Ag", "Au"),
    ("Pd", "Pt"),
    ("Pd", "Au"),
    ("Pt", "Au"),
]

# Conventional fcc basis (fractional)
FCC_BASIS = [
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
]

REPS = (3, 3, 3)  # 3x3x3 supercell
N_ATOMS = 4 * REPS[0] * REPS[1] * REPS[2]  # 108

# ------------------------------
# Helpers
# ------------------------------

def vegard_lattice(aA: float, aB: float, fracB: float) -> float:
    """Vegard's law: a(x) = (1-x)*aA + x*aB, where x = frac of B in A-B alloy."""
    return (1.0 - fracB) * aA + fracB * aB

def build_fcc_positions(a: float) -> list:
    """Return Cartesian positions for a 3x3x3 conventional fcc supercell at lattice a (Å)."""
    positions = []
    nx, ny, nz = REPS
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for fx, fy, fz in FCC_BASIS:
                    x = (i + fx) * a
                    y = (j + fy) * a
                    z = (k + fz) * a
                    positions.append((x, y, z))
    return positions

def choose_types(A: str, B: str, percentB: int, rng: random.Random) -> list:
    """Return a list of 108 LAMMPS types (1/2), matching the requested overall composition.
       Convention: type1 = lower-Z element, type2 = higher-Z element.
    """
    # Map type->element
    if ZNUM[A] < ZNUM[B]:
        type1, type2 = A, B
    else:
        type1, type2 = B, A

    # How many of the 108 should be type2 (the "higher-Z element")?
    # percentB is at.% of element B (as named in the pair (A,B)),
    # but "type2" might not equal B if B happens to have lower Z.
    # So we compute target counts by element name first.
    nB = round(N_ATOMS * (percentB / 100.0))
    nA = N_ATOMS - nB

    # Now map to type counts:
    # count for type1 = count of whichever element has lower Z
    if type1 == A:
        n_type1 = nA
        n_type2 = nB
    else:
        n_type1 = nB
        n_type2 = nA

    types = [1] * n_type1 + [2] * n_type2
    rng.shuffle(types)
    return types, (type1, type2), (n_type1, n_type2)

def write_lammps_data(path: Path, boxlen: float, types: list, positions: list,
                      type_to_element: tuple, masses: dict):
    """Write LAMMPS 'data.lmp' in atomic style."""
    (elem1, elem2) = type_to_element

    with open(path, "w") as f:
        f.write("LAMMPS data file via generate_binaries.py (fcc 3x3x3)\n\n")
        f.write(f"{len(positions)} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"0.0 {boxlen:.6f} xlo xhi\n")
        f.write(f"0.0 {boxlen:.6f} ylo yhi\n")
        f.write(f"0.0 {boxlen:.6f} zlo zhi\n\n")

        f.write("Masses\n\n")
        f.write(f"1 {masses[elem1]:.6f} # {elem1}\n")
        f.write(f"2 {masses[elem2]:.6f} # {elem2}\n\n")

        f.write("Atoms # atomic\n\n")
        for i, ((x, y, z), t) in enumerate(zip(positions, types), start=1):
            f.write(f"{i} {t} {x:.8f} {y:.8f} {z:.8f}\n")

def safe_copy(src: Path, dst_dir: Path):
    if src.exists():
        shutil.copy(src, dst_dir / src.name)
    else:
        print(f"[WARN] missing reference: {src}")

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345, help="Random seed")
    ap.add_argument("--root", type=str, default=".", help="Output root directory")
    ap.add_argument("--dry-run", action="store_true", help="Only print actions")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root).resolve()
    refs_melt = Path("/REFS/melt.in")
    refs_submit = Path("/REFS/submit_gpu")

    for A, B in PAIRS:
        # sanity: we need both elements to be in our reference tables
        for el in (A, B):
            if el not in FCC_A or el not in AMASS or el not in ZNUM:
                raise ValueError(f"Missing reference data for element: {el}")

        pair_dir = root / f"{A}{B}"
        if not args.dry_run:
            pair_dir.mkdir(parents=True, exist_ok=True)

        for pctB in range(0, 101, 10):
            pctA = 100 - pctB
            subdir = pair_dir / f"{A}{pctA}_{B}{pctB}"
            if not args.dry_run:
                subdir.mkdir(parents=True, exist_ok=True)

            # Vegard lattice
            aA = FCC_A[A]
            aB = FCC_A[B]
            fracB = pctB / 100.0
            a_alloy = vegard_lattice(aA, aB, fracB)
            boxlen = a_alloy * REPS[0]  # cubic

            # Build positions (deterministic geometry)
            positions = build_fcc_positions(a_alloy)

            # Assign types (randomized)
            types, (type1_elem, type2_elem), (n1, n2) = choose_types(A, B, pctB, rng)

            # Write data.lmp
            data_path = subdir / "data.lmp"
            if args.dry_run:
                print(f"[DRY] Would write: {data_path}")
            else:
                write_lammps_data(
                    data_path, boxlen, types, positions,
                    (type1_elem, type2_elem), AMASS
                )

                # composition record
                with open(subdir / "composition.txt", "w") as cf:
                    cf.write(f"Pair: {A}-{B}\n")
                    cf.write(f"Folder: {A}{pctA}_{B}{pctB}\n")
                    cf.write(f"Vegard a (Å): {a_alloy:.6f}  [a_{A}={aA:.6f}, a_{B}={aB:.6f}]\n")
                    cf.write(f"Type mapping: type1={type1_elem}, type2={type2_elem}\n")
                    cf.write(f"Counts: type1={n1}, type2={n2}  (total={N_ATOMS})\n")

                # copy reference scripts
                safe_copy(refs_melt, subdir)
                safe_copy(refs_submit, subdir)

    print("Done.")

if __name__ == "__main__":
    main()
