#!/usr/bin/env python3
"""
Set up Cu-Ag PBE/PBEsol enthalpy scans from existing structures.

Default layout created:
  ./PBE/Cu_0/T800/
  ./PBEsol/Cu_0/T800/
  ...

Each run folder receives:
  in.lmp              copied structure
  enthalpy.in          copied from ./REFS/ and patched
  submit_enthalpy      copied from ./REFS/
  enthalpy_setup.json  setup metadata

Phase assignment is based on a built-in Tm table estimated from the Cu-Ag
Tm-vs-composition plot. If T < Tm -> solid; otherwise -> liquid.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_COMPS = [0, 20, 40, 50, 60, 80, 100]
DEFAULT_TEMPS = [800, 900, 1000, 1100, 1200, 1300, 1400]

# Estimated from Cu_Ag_tm_vs_composition.tiff.
# PBEsol Cu20 uses the lower Tm point, not the anomalously high averaged point.
DEFAULT_TM = {
    "PBE": {
        0: 980.0,
        20: 870.0,
        40: 805.0,
        50: 800.0,
        60: 825.0,
        80: 960.0,
        100: 1260.0,
    },
    "PBEsol": {
        0: 1200.0,
        20: 1080.0,  # lower PBEsol Cu20 point from plot, not high/average point
        40: 980.0,
        50: 990.0,
        60: 1010.0,
        80: 1110.0,
        100: 1430.0,
    },
}

DEFAULT_MODELS = {
    "PBE": "/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au/DNPs/Cu_Ag-9-x/Cu_Ag-9-0c.pb",
    "PBEsol": "/ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au_sol/DNPs/Cu_Ag-10-x/Cu_Ag-10-1c.pb",
}


def parse_csv_numbers(s: str, as_int: bool = True) -> List:
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        val = float(tok)
        vals.append(int(val) if as_int and val.is_integer() else val)
    return vals


def parse_csv_strings(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def comp_dir(comp: int | float) -> str:
    if float(comp).is_integer():
        return f"Cu_{int(comp)}"
    return f"Cu_{comp:g}"


def temp_dir(temp: int | float) -> str:
    if float(temp).is_integer():
        return f"T{int(temp)}"
    return f"T{temp:g}"


def patch_variable(text: str, name: str, style: str, value: str) -> str:
    """Patch LAMMPS variable line while preserving trailing comments."""
    pattern = re.compile(rf"^(\s*variable\s+{re.escape(name)}\s+{re.escape(style)}\s+)\S+(.*)$", re.M)
    repl = rf"\g<1>{value}\2"
    if pattern.search(text):
        return pattern.sub(repl, text)
    return text + f"\nvariable {name} {style} {value}\n"


def patch_enthalpy_input(template: str, *, model: str, T: float, Tinit: float, seed: int, phase: str) -> str:
    text = template
    text = patch_variable(text, "model", "string", model)
    text = patch_variable(text, "T", "equal", f"{T:g}")
    text = patch_variable(text, "Tinit", "equal", f"{Tinit:g}")
    text = patch_variable(text, "seed", "equal", str(seed))

    # For solid cases, minimize before velocity initialization and ramp NPT equilibration
    # from Tinit to T. This assumes a conventional template containing the velocity line
    # and an EQ npt line similar to the REFS template.
    if phase == "solid":
        if "SOLID_MINIMIZATION_INSERTED_BY_SETUP" not in text:
            min_block = """
# --- SOLID_MINIMIZATION_INSERTED_BY_SETUP ---
# Minimize the starting solid supercell before assigning velocities/heating.
min_style       cg
minimize        1.0e-10 1.0e-10 2000 20000
reset_timestep  0
# --- END_SOLID_MINIMIZATION_INSERTED_BY_SETUP ---
"""
            text = re.sub(r"(?m)^(\s*velocity\s+all\s+create\s+.*)$", min_block + r"\1", text, count=1)

        # Heat solid during equilibration: fix EQ all npt temp ${Tinit} ${T} ...
        text = re.sub(
            r"(?m)^(\s*fix\s+EQ\s+all\s+npt\s+temp\s+)\$\{T\}\s+\$\{T\}(\s+.*)$",
            r"\1${Tinit} ${T}\2",
            text,
            count=1,
        )
    return text


def candidate_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def find_liquid_source(diffusion_root: Path, liquid_source_temp: float, comp: int | float, liquid_rep: str) -> Optional[Path]:
    cdir = diffusion_root / temp_dir(liquid_source_temp) / comp_dir(comp)
    repdir = cdir / liquid_rep
    candidates = [
        repdir / "final_diffusion.lmp",
        repdir / "final.lmp",
        cdir / "final_diffusion.lmp",
        cdir / "final.lmp",
        repdir / "in.lmp",
        cdir / "in.lmp",
    ]
    return candidate_existing(candidates)


def find_solid_source(solid_root: Path, diffusion_root: Path, liquid_source_temp: float, comp: int | float) -> Optional[Path]:
    # solid_root may be either parent containing Cu_X directories or a T#### parent containing Cu_X directories.
    cdir = solid_root / comp_dir(comp)
    diffusion_cdir = diffusion_root / temp_dir(liquid_source_temp) / comp_dir(comp)
    candidates = [
        cdir / "solid.lmp",
        cdir / "in.solid.lmp",
        cdir / "in_lattice.lmp",
        cdir / "in.lmp",
        diffusion_cdir / "solid.lmp",
        diffusion_cdir / "in.solid.lmp",
        diffusion_cdir / "in_lattice.lmp",
        diffusion_cdir / "in.lmp",
    ]
    return candidate_existing(candidates)


def decide_phase(functional: str, comp: int | float, temp: int | float, force_phase: str) -> Tuple[str, float]:
    comp_int = int(comp)
    tm = DEFAULT_TM[functional][comp_int]
    if force_phase in {"solid", "liquid"}:
        return force_phase, tm
    return ("solid" if float(temp) < tm else "liquid"), tm


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"      (dry-run) Would run: cd {cwd} && {' '.join(cmd)}")
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Set up Cu-Ag PBE/PBEsol enthalpy scan directories.")
    ap.add_argument("--root", default=".", help="Output root for PBE/ and PBEsol/ folders. Default: .")
    ap.add_argument("--refs", default="./REFS", help="Directory containing enthalpy.in and submit_enthalpy. Default: ./REFS")
    ap.add_argument("--diffusion-root", default="../Diffusion", help="Diffusion root used for liquid source structures. Default: ../Diffusion")
    ap.add_argument("--liquid-source-temp", type=float, default=1423.0, help="Diffusion temperature folder to use for liquid structures. Default: 1423")
    ap.add_argument("--liquid-rep", default="rep01", help="Preferred diffusion replicate for liquid final structure. Default: rep01")
    ap.add_argument("--solid-root", default="../Diffusion/T1423", help="Root containing solid starting structures by Cu_X. Default: ../Diffusion/T1423")
    ap.add_argument("--compositions", default=','.join(map(str, DEFAULT_COMPS)), help="Comma-separated Cu at.%% values. Default: 0,20,40,50,60,80,100")
    ap.add_argument("--temperatures", default=','.join(map(str, DEFAULT_TEMPS)), help="Comma-separated temperatures. Default: 800,900,...,1400")
    ap.add_argument("--functionals", default="PBE,PBEsol", help="Comma-separated functionals: PBE,PBEsol")
    ap.add_argument("--pbe-model", default=DEFAULT_MODELS["PBE"], help="PBE DNP model path")
    ap.add_argument("--pbesol-model", default=DEFAULT_MODELS["PBEsol"], help="PBEsol DNP model path")
    ap.add_argument("--base-seed", type=int, default=4928459)
    ap.add_argument("--solid-tinit", type=float, default=300.0, help="Initial velocity temperature for solid heating. Default: 300 K")
    ap.add_argument("--liquid-tinit-offset", type=float, default=0.0, help="Liquid Tinit = T + offset. Default: 0 K")
    ap.add_argument("--force-phase", choices=["auto", "solid", "liquid"], default="auto", help="Override automatic Tm-based phase assignment")
    ap.add_argument("--submit", action="store_true", help="Submit each job after setup")
    ap.add_argument("--submit-cmd", default="qsub", help="Scheduler command. Default: qsub; use sbatch if needed")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite enthalpy.in/submit/in.lmp in existing folders")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing or submitting")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    refs = Path(args.refs)
    diffusion_root = Path(args.diffusion_root)
    solid_root = Path(args.solid_root)
    comps = parse_csv_numbers(args.compositions, as_int=True)
    temps = parse_csv_numbers(args.temperatures, as_int=True)
    functionals = parse_csv_strings(args.functionals)
    models = {"PBE": args.pbe_model, "PBEsol": args.pbesol_model}

    enthalpy_ref = refs / "enthalpy.in"
    submit_ref = refs / "submit_enthalpy"
    if not enthalpy_ref.is_file():
        raise SystemExit(f"ERROR: Missing {enthalpy_ref}")
    if not submit_ref.is_file():
        raise SystemExit(f"ERROR: Missing {submit_ref}")

    template = enthalpy_ref.read_text()

    print("Enthalpy scan setup")
    print("-------------------")
    print(f"Output root:       {root}")
    print(f"REFS:              {refs}")
    print(f"Diffusion root:    {diffusion_root}")
    print(f"Liquid source T:   T{args.liquid_source_temp:g}")
    print(f"Solid root:        {solid_root}")
    print(f"Compositions:      {comps}")
    print(f"Temperatures:      {temps}")
    print(f"Functionals:       {functionals}")
    print(f"Force phase:       {args.force_phase}")
    print(f"Submit:            {args.submit} ({args.submit_cmd})")
    print(f"Overwrite:         {args.overwrite}")
    print(f"Dry run:           {args.dry_run}")
    print()

    n_created = n_skipped = n_missing = n_submitted = 0

    for func in functionals:
        if func not in models:
            raise SystemExit(f"ERROR: Unknown functional {func}; valid: PBE,PBEsol")
        model = models[func]
        for comp in comps:
            for temp in temps:
                phase, tm = decide_phase(func, comp, temp, args.force_phase)
                outdir = root / func / comp_dir(comp) / temp_dir(temp)

                if outdir.exists() and not args.overwrite and (outdir / "enthalpy.in").exists():
                    print(f">> Skipping existing {outdir} (use --overwrite to replace setup files)")
                    n_skipped += 1
                    continue

                if phase == "liquid":
                    source = find_liquid_source(diffusion_root, args.liquid_source_temp, comp, args.liquid_rep)
                    tinit = float(temp) + args.liquid_tinit_offset
                else:
                    source = find_solid_source(solid_root, diffusion_root, args.liquid_source_temp, comp)
                    tinit = args.solid_tinit

                if source is None:
                    print(f">> MISSING {func} {comp_dir(comp)} {temp_dir(temp)} phase={phase}: no source structure found")
                    n_missing += 1
                    continue

                seed = args.base_seed + int(temp) + int(comp) * 10 + (0 if func == "PBE" else 100000)
                patched = patch_enthalpy_input(
                    template,
                    model=model,
                    T=float(temp),
                    Tinit=float(tinit),
                    seed=seed,
                    phase=phase,
                )

                print(f">> {outdir} phase={phase} Tm~{tm:g} K source={source}")
                if not args.dry_run:
                    outdir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, outdir / "in.lmp")
                    shutil.copy2(submit_ref, outdir / "submit_enthalpy")
                    (outdir / "enthalpy.in").write_text(patched)
                    meta = {
                        "functional": func,
                        "composition_Cu_atpct": comp,
                        "temperature_K": temp,
                        "phase": phase,
                        "Tm_threshold_K": tm,
                        "model": model,
                        "source_structure": str(source),
                        "Tinit_K": tinit,
                        "seed": seed,
                    }
                    (outdir / "enthalpy_setup.json").write_text(json.dumps(meta, indent=2))
                    n_created += 1
                    if args.submit:
                        run_cmd([args.submit_cmd, "submit_enthalpy"], outdir, dry_run=False)
                        n_submitted += 1
                else:
                    n_created += 1
                    if args.submit:
                        print(f"      (dry-run) Would submit: cd {outdir} && {args.submit_cmd} submit_enthalpy")

    print()
    print("Summary")
    print("-------")
    print(f"Setup entries prepared/planned: {n_created}")
    print(f"Existing skipped:               {n_skipped}")
    print(f"Missing source structures:      {n_missing}")
    print(f"Submitted jobs:                 {n_submitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
