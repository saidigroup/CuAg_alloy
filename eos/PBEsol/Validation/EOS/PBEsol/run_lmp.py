#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import re

# ---------------- USER SETTINGS ----------------
ROOT = Path(".")
SYSTEMS = ["Cu_100", "Cu_50", "Cu_0", "Cu_25", "Cu_75"]
#SYSTEMS = ["Cu_25", "Cu_75", "Cu_50"]

REF_LMP_IN = ROOT / "relaxed.in"
REF_LMP_SUBMIT = ROOT / "relaxed_submit_batch_lmp"

SUBMIT_JOBS = True  # False = setup only
# ------------------------------------------------


def run_cmd(cmd, cwd):
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print(f"{cwd}: stdout: {result.stdout.strip()}")
        if result.stderr.strip():
            print(f"{cwd}: stderr: {result.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {cwd}: {' '.join(cmd)}")
        if e.stdout.strip():
            print(f"  stdout: {e.stdout.strip()}")
        if e.stderr.strip():
            print(f"  stderr: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"ERROR: command not found: {cmd[0]}")
        return False


def get_pair_coeff_line(system: str) -> str:
    """
    Return the correct DeepMD pair_coeff mapping for each system.

    IMPORTANT:
    - Cu_100 -> works if in.lmp has 1 atom type
    - Cu_0   -> works if in.lmp has 1 atom type
    - Cu_50  -> works if in.lmp has 2 atom types
    """
    if system == "Cu_100":
        return "pair_coeff      * * Cu"
    elif system == "Cu_50":
        return "pair_coeff      * * Cu Ag"
    elif system == "Cu_75":
        return "pair_coeff      * * Cu Ag"
    elif system == "Cu_25":
        return "pair_coeff      * * Cu Ag"
    elif system == "Cu_0":
        return "pair_coeff      * * Ag"
    else:
        raise ValueError(f"Unknown system: {system}")


def patch_relaxed_in(relaxed_in_path: Path, system: str):
    """
    Replace the pair_coeff line in relaxed.in with the system-specific mapping.
    """
    lines = relaxed_in_path.read_text().splitlines()
    new_pair_coeff = get_pair_coeff_line(system)

    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith("pair_coeff"):
            lines[i] = new_pair_coeff
            found = True
            break

    if not found:
        raise ValueError(f"Could not find pair_coeff line in {relaxed_in_path}")

    relaxed_in_path.write_text("\n".join(lines) + "\n")


def count_atom_types_in_lmp(infile: Path):
    """
    Read the number of atom types from a LAMMPS data file header.
    """
    text = infile.read_text(errors="ignore").splitlines()
    for line in text:
        m = re.match(r"^\s*(\d+)\s+atom types\s*$", line)
        if m:
            return int(m.group(1))
    return None


def validate_mapping_vs_in_lmp(system: str, in_lmp: Path):
    """
    Warn if the pair_coeff mapping likely mismatches the number of atom types.
    """
    ntypes = count_atom_types_in_lmp(in_lmp)
    if ntypes is None:
        print(f"{in_lmp.parent}: WARNING could not determine atom types from in.lmp")
        return

    expected_labels = {
        "Cu_100": 1,
        "Cu_50": 2,
        "Cu_75": 2,
        "Cu_25": 2,
        "Cu_0": 1,
    }[system]

    if ntypes != expected_labels:
        print(
            f"{in_lmp.parent}: WARNING in.lmp has {ntypes} atom types, "
            f"but {system} mapping expects {expected_labels}. "
            f"DNP may fail unless you adjust pair_coeff."
        )


def main():
    missing = [p for p in [REF_LMP_IN, REF_LMP_SUBMIT] if not p.exists()]
    if missing:
        print("Missing reference files:")
        for p in missing:
            print(" ", p)
        raise SystemExit(1)

    for system in SYSTEMS:
        system_dir = ROOT / system
        if not system_dir.exists():
            print(f"Skipping {system}: folder not found")
            continue

        vdirs = sorted([d for d in system_dir.iterdir() if d.is_dir() and d.name.startswith("V_")])
        if not vdirs:
            print(f"Skipping {system}: no V_* folders found")
            continue

        for vdir in vdirs:
            in_lmp = vdir / "in.lmp"
            if not in_lmp.exists():
                print(f"Skipping {vdir}: missing in.lmp")
                continue

            shutil.copy2(REF_LMP_IN, vdir / "relaxed.in")
            shutil.copy2(REF_LMP_SUBMIT, vdir / "relaxed_submit_batch_lmp")

            try:
                patch_relaxed_in(vdir / "relaxed.in", system)
                validate_mapping_vs_in_lmp(system, in_lmp)
                print(f"{vdir}: patched relaxed.in for {system}")
            except Exception as e:
                print(f"Skipping {vdir}: failed to patch relaxed.in -> {e}")
                continue

            if SUBMIT_JOBS:
                run_cmd(["qsub", "relaxed_submit_batch_lmp"], cwd=vdir)

    print("LAMMPS EOS setup/submission complete.")


if __name__ == "__main__":
    main()
