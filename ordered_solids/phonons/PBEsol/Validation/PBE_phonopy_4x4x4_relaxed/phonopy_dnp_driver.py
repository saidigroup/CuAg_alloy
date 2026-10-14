#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from ase.io import read, write
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


MODEL = (
    "/ix3/walsaidi/cma94/Cu_Ag/Cu_Ni_Ag_Pd_Pt_Au_sol/"
    "DNPs/Cu_Ag-9-x/Cu_Ag-9-0.pb"
)

# Fixed mapping required by the trained Cu-Ag potential:
# LAMMPS type 1 -> Cu
# LAMMPS type 2 -> Ag
SPECORDER = ["Cu", "Ag"]


def ase_to_phonopy(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell),
        scaled_positions=atoms.get_scaled_positions(),
    )


def phonopy_to_ase(phatoms):
    from ase import Atoms

    return Atoms(
        symbols=list(phatoms.symbols),
        cell=np.asarray(phatoms.cell),
        scaled_positions=np.asarray(phatoms.scaled_positions),
        pbc=True,
    )


def parse_force_dump(filename, natoms):
    """
    Read:
       ITEM: ATOMS id type fx fy fz

    and return forces ordered by atom ID.
    """

    forces = np.zeros((natoms, 3), dtype=float)
    inside_atoms = False
    count = 0

    with open(filename) as f:
        for line in f:

            if line.startswith("ITEM: ATOMS"):
                inside_atoms = True
                continue

            if inside_atoms:
                fields = line.split()

                if len(fields) < 5:
                    continue

                try:
                    atom_id = int(fields[0])
                    fx = float(fields[2])
                    fy = float(fields[3])
                    fz = float(fields[4])
                except ValueError:
                    continue

                forces[atom_id - 1] = [fx, fy, fz]
                count += 1

                if count == natoms:
                    break

    if count != natoms:
        raise RuntimeError(
            f"Expected {natoms} force records in {filename}, "
            f"found {count}"
        )

    return forces


def make_lammps_input(run_dir, model):
    infile = run_dir / "in.force"

    text = f"""\
units           metal
atom_style      atomic
boundary        p p p

read_data       displaced.lmp

pair_style      deepmd {model}
pair_coeff      * * Cu Ag

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

thermo          1
thermo_style    custom step atoms pe press pxx pyy pzz pxy pxz pyz

dump            FORCE all custom 1 forces.dump id type fx fy fz
dump_modify     FORCE sort id

run             0

undump          FORCE
"""

    infile.write_text(text)


def run_lammps(run_dir, lmp_command):
    logfile = run_dir / "lammps.stdout"

    with logfile.open("w") as out:
        result = subprocess.run(
            [lmp_command, "-in", "in.force"],
            cwd=run_dir,
            stdout=out,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed in {run_dir}; "
            f"see {logfile}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--structure",
        default="unitcell.lmp",
    )

    parser.add_argument(
        "--model",
        default=MODEL,
    )

    parser.add_argument(
        "--lmp",
        default=os.environ.get("LAMMPS_EXE", "lmp"),
        help="LAMMPS executable; default = lmp",
    )

    parser.add_argument(
        "--dim",
        nargs=3,
        type=int,
        default=[4, 4, 4],
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--mesh",
        nargs=3,
        type=int,
        default=[30, 30, 30],
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=1600.0,
    )

    parser.add_argument(
        "--tstep",
        type=float,
        default=10.0,
    )

    args = parser.parse_args()

    model = Path(args.model)

    if not model.is_file():
        raise FileNotFoundError(model)

    if shutil.which(args.lmp) is None:
        raise RuntimeError(
            f"LAMMPS executable '{args.lmp}' not found in PATH."
        )

    print("=" * 72)
    print("Cu-Ag PBEsol DNP phonopy")
    print("=" * 72)
    print("structure :", args.structure)
    print("model     :", model)
    print("LAMMPS    :", args.lmp)
    print("supercell :", args.dim)
    print("distance  :", args.distance, "Angstrom")
    print("mesh      :", args.mesh)
    print("=" * 72)

    # ------------------------------------------------------------
    # Read original LAMMPS cell.
    #
    # ASE uses the masses to infer Cu and Ag:
    # 63.546  -> Cu
    # 107.8682 -> Ag
    # ------------------------------------------------------------

    atoms = read(
        args.structure,
        format="lammps-data",
        atom_style="atomic",
        sort_by_id=True,
    )

    symbols = atoms.get_chemical_symbols()

    ncu = symbols.count("Cu")
    nag = symbols.count("Ag")
    nat = len(atoms)

    print()
    print(f"Reference-cell atoms = {nat}")
    print(f"Cu                   = {ncu}")
    print(f"Ag                   = {nag}")
    print(f"Cu fraction          = {100*ncu/nat:.4f} at.%")

    expected_supercell_atoms = nat * np.prod(args.dim)

    print(
        f"Expected phonon supercell = "
        f"{expected_supercell_atoms} atoms"
    )

    # ------------------------------------------------------------
    # Phonopy
    # ------------------------------------------------------------

    unitcell = ase_to_phonopy(atoms)

    phonon = Phonopy(
        unitcell,
        supercell_matrix=np.diag(args.dim),
    )

    phonon.generate_displacements(
        distance=args.distance,
        is_plusminus=True,
    )

    supercells = phonon.supercells_with_displacements

    print()
    print(
        "Symmetry-reduced displaced structures =",
        len(supercells),
    )

    if not supercells:
        raise RuntimeError("No displaced supercells generated.")

    actual_n = len(supercells[0])

    print("Atoms/displaced supercell             =", actual_n)

    # Save displacement information immediately.
    phonon.save(
        filename="phonopy_disp.yaml",
    )

    # ------------------------------------------------------------
    # Calculate DNP forces for every displacement.
    # ------------------------------------------------------------

    root = Path("displacements")
    root.mkdir(exist_ok=True)

    force_sets = []

    for i, ph_sc in enumerate(supercells, 1):

        dirname = root / f"disp-{i:04d}"
        dirname.mkdir(exist_ok=True)

        force_file = dirname / "forces.dump"

        ase_sc = phonopy_to_ase(ph_sc)

        # Explicit specorder guarantees:
        #
        # 1 = Cu
        # 2 = Ag
        #
        write(
            dirname / "displaced.lmp",
            ase_sc,
            format="lammps-data",
            atom_style="atomic",
            specorder=SPECORDER,
            masses=True,
        )

        make_lammps_input(dirname, model)

        if not force_file.is_file():

            print(
                f"[{i:4d}/{len(supercells):4d}] "
                f"running DNP forces..."
            )

            run_lammps(
                dirname,
                args.lmp,
            )

        else:
            print(
                f"[{i:4d}/{len(supercells):4d}] "
                f"existing forces found"
            )

        forces = parse_force_dump(
            force_file,
            actual_n,
        )

        force_sets.append(forces)

        maxf = np.linalg.norm(forces, axis=1).max()

        print(
            f"                      max|F| = "
            f"{maxf:.8f} eV/Ang"
        )

    # ------------------------------------------------------------
    # Force constants
    # ------------------------------------------------------------

    phonon.forces = force_sets
    phonon.produce_force_constants()

    phonon.save(
        filename="phonopy_params.yaml",
        settings={
            "force_constants": True,
        },
    )

    # ------------------------------------------------------------
    # Frequencies / instability check
    # ------------------------------------------------------------

    phonon.run_mesh(
        args.mesh,
        with_eigenvectors=False,
        is_mesh_symmetry=True,
    )

    mesh_dict = phonon.get_mesh_dict()

    freqs = np.asarray(mesh_dict["frequencies"])

    min_freq = float(np.min(freqs))
    n_imaginary = int(np.sum(freqs < -0.10))

    print()
    print("Minimum mesh frequency =", min_freq, "THz")
    print(
        "Modes below -0.10 THz  =",
        n_imaginary,
    )

    # ------------------------------------------------------------
    # Thermal properties
    # ------------------------------------------------------------

    phonon.run_thermal_properties(
        t_min=0,
        t_max=args.tmax,
        t_step=args.tstep,
    )

    tp = phonon.get_thermal_properties_dict()

    temperatures = np.asarray(tp["temperatures"])
    free_energy = np.asarray(tp["free_energy"])
    entropy = np.asarray(tp["entropy"])
    cv = np.asarray(tp["heat_capacity"])

    # Phonopy:
    # F  -> kJ/mol of reference cells
    # S  -> J/K/mol of reference cells
    # Cv -> J/K/mol of reference cells
    #
    # Convert F to eV/atom as well.
    KJMOL_TO_EV = 0.0103642696562622

    free_energy_ev_atom = (
        free_energy * KJMOL_TO_EV / nat
    )

    data = np.column_stack(
        [
            temperatures,
            free_energy,
            free_energy_ev_atom,
            entropy,
            entropy / nat,
            cv,
            cv / nat,
        ]
    )

    np.savetxt(
        "thermal_properties.dat",
        data,
        fmt="%16.8f",
        header=(
            "T_K "
            "Fvib_kJ_per_mol_cell "
            "Fvib_eV_per_atom "
            "Svib_J_per_mol_cell_K "
            "Svib_J_per_mol_atom_K "
            "Cv_J_per_mol_cell_K "
            "Cv_J_per_mol_atom_K"
        ),
    )

    metadata = {
        "reference_atoms": nat,
        "Cu_atoms": ncu,
        "Ag_atoms": nag,
        "Cu_at_percent": 100.0 * ncu / nat,
        "supercell_matrix": args.dim,
        "supercell_atoms": int(actual_n),
        "displacement_distance_A": args.distance,
        "number_of_displacements": len(supercells),
        "mesh": args.mesh,
        "minimum_frequency_THz": min_freq,
        "number_modes_below_minus_0p10_THz": n_imaginary,
        "model": str(model),
        "type_mapping": {
            "1": "Cu",
            "2": "Ag",
        },
    }

    with open("phonopy_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 72)
    print("FINISHED")
    print("=" * 72)
    print("phonopy_disp.yaml")
    print("phonopy_params.yaml")
    print("thermal_properties.dat")
    print("phonopy_metadata.json")
    print("displacements/")
    print("=" * 72)


if __name__ == "__main__":
    main()
