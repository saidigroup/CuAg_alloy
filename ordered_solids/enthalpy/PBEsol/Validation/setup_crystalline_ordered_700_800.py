#!/usr/bin/env python3
"""Set up Cu-Ag ordered-crystalline and endpoint enthalpy runs from phonon unit cells.

Run from Validation: python3 setup_crystalline_ordered_v2.py
Creates 6x6x6 6912-atom cells for PBE and PBEsol, Cu 0/25/50/75/100,
T 900/1000/1200/1300 K, preserving existing run directories.
Requires ASE in the running Python environment. Does NOT submit jobs.
"""
import argparse
import json
import re
import shlex
from pathlib import Path
from ase.io import read, write

COMPS = (0, 25, 50, 75, 100)
DEFAULT_TEMPS = (700, 800)
MODEL_BASENAMES = {'PBE': 'Cu_Ag-9-0.pb', 'PBEsol': 'Cu_Ag-10-1.pb'}


def replace_var(text, name, value):
    pat = re.compile(r'^(\s*variable\s+' + re.escape(name) +
                     r'\s+(?:string|equal)\s+)\S+([^\n]*)$', re.M)
    new, count = pat.subn(lambda m: m.group(1) + str(value) + m.group(2), text)
    if count != 1:
        raise RuntimeError(f'Expected exactly one LAMMPS variable {name}; found {count}')
    return new


def ref_path(root, functional, comp):
    name = f'Cu_{comp}'
    if functional == 'PBE':
        campaign = ('PBE_fcc_endpoint_test' if comp in (0, 100)
                    else 'PBE_phonopy_4x4x4_relaxed')
    else:
        campaign = 'PBEsol_phonopy_6x6x6'
    return root / campaign / name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd(),
                    help='Validation directory (default: current directory)')
    ap.add_argument('--template-dir', type=Path, default=None,
                    help='Template directory containing enthalpy.in and submit_enthalpy')
    ap.add_argument('--temps', type=int, nargs='+', default=list(DEFAULT_TEMPS))
    args = ap.parse_args()
    root = args.root.resolve()
    template_dir = (args.template_dir.resolve() if args.template_dir is not None
                    else root / 'Enthalpy/PBEsol/Cu_100/T1000')
    enthalpy_template = template_dir / 'enthalpy.in'
    submit_template = root / 'Enthalpy/crystalline_ordered/PBE/Cu_25/T1000/submit_enthalpy'
    for p in (enthalpy_template, submit_template):
        if not p.is_file():
            raise SystemExit(f'ERROR: missing template {p}. Use --template-dir DIR.')
    enthalpy_text = enthalpy_template.read_text()
    submit_text = submit_template.read_text()

    if not re.search(
        r"(?m)^#SBATCH\s+--cluster=gpu\s*$",
        submit_text,
    ):
        raise SystemExit("ERROR: Expected --cluster=gpu")

    if not re.search(
        r"(?m)^#SBATCH\s+--partition=a100\s*$",
        submit_text,
    ):
        raise SystemExit("ERROR: Expected --partition=a100")

    if not re.search(
        r"(?m)^#SBATCH\s+--gres=gpu:1(?:\s|$)",
        submit_text,
    ):
        raise SystemExit("ERROR: Expected one GPU")

    if 'enthalpy_avg.dat' not in enthalpy_text or 'write_data' not in enthalpy_text:
        raise SystemExit('ERROR: enthalpy.in is not the expected NPT/averaging template.')
    if 'enthalpy.in' not in submit_text or 'lmp ' not in submit_text:
        raise SystemExit('ERROR: submit_enthalpy does not launch the expected workflow.')
    if 'pair_coeff' not in enthalpy_text or 'Cu Ag' not in enthalpy_text:
        raise SystemExit('ERROR: check LAMMPS pair_coeff type mapping; expected Cu Ag.')
    temps = args.temps
    if not temps or len(set(temps)) != len(temps) or any(t <= 0 for t in temps):
        raise SystemExit('ERROR: temperatures must be unique positive integers.')

    destination = root / 'Enthalpy/crystalline_ordered'
    planned = []
    skipped = []
    # Validate ALL structures and models before creating directories.
    for functional in ('PBE', 'PBEsol'):
        for comp in COMPS:
            parent = ref_path(root, functional, comp)
            structure = parent / 'unitcell.lmp'
            metadata = parent / 'phonopy_metadata.json'
            for p in (structure, metadata):
                if not p.is_file():
                    raise SystemExit(f'ERROR: missing phonon reference {p}')
            md = json.loads(metadata.read_text())
            model = Path(md['model'])
            if model.name != MODEL_BASENAMES[functional] or not model.is_file():
                raise SystemExit(f'ERROR: wrong or missing {functional} DNP: {model}')
            if md['reference_atoms'] != 32 or abs(float(md['Cu_at_percent']) - comp) > 0.01:
                raise SystemExit(f'ERROR: inconsistent reference phonon metadata: {metadata}')
            atoms = read(str(structure), format='lammps-data', atom_style='atomic', sort_by_id=True)
            syms = atoms.get_chemical_symbols()
            ncu = syms.count('Cu')
            nag = syms.count('Ag')
            if len(atoms) != 32 or (ncu, nag) != (32 * comp // 100, 32 * (100 - comp) // 100):
                raise SystemExit(f'ERROR: incorrect source composition: {structure} Cu={ncu} Ag={nag}')
            for t in temps:
                run = destination / functional / f'Cu_{comp}' / f'T{t}'
                if run.exists():
                    # A pre-existing directory may contain running or completed jobs.
                    # Deliberately skip; never overwrite or resubmit it.
                    skipped.append(run)
                else:
                    planned.append((functional, comp, t, structure, metadata, model, atoms, run))

    print(f'Validated 10 reference structures; {len(planned)} new jobs, '
          f'{len(skipped)} existing directories will be SKIPPED.')
    created = []
    for functional, comp, temp, structure, metadata, model, atoms, run in planned:
        run.mkdir(parents=True, exist_ok=False)
        cell = atoms.repeat((6, 6, 6))
        assert len(cell) == 6912
        target = run / 'in.lmp'
        write(str(target), cell, format='lammps-data', atom_style='atomic',
              specorder=['Cu', 'Ag'], masses=True)
        reread = read(str(target), format='lammps-data', atom_style='atomic', sort_by_id=True)
        types = reread.get_chemical_symbols()
        ncu, nag = types.count('Cu'), types.count('Ag')
        if (ncu, nag) != (6912 * comp // 100, 6912 * (100 - comp) // 100):
            raise RuntimeError(f'ERROR: generated composition mismatch in {target}: Cu={ncu} Ag={nag}')
        inp = enthalpy_text
        seed = 5030459 + (0 if functional == 'PBE' else 1000000) + 100 * comp + temp
        replacements = {'datafile': 'in.lmp', 'model': str(model), 'T': temp,
                        'Tinit': 300, 'Ptarget': 0.0, 'seed': seed}
        for key, val in replacements.items():
            inp = replace_var(inp, key, val)
        (run / 'enthalpy.in').write_text(inp)
        sub = run / 'submit_enthalpy'
        sub.write_text(submit_text)
        sub.chmod(0o755)
        provenance = {
            'functional': functional, 'Cu_at_percent': comp, 'T_K': temp,
            'reference_structure': str(structure), 'phonopy_metadata': str(metadata),
            'model': str(model), 'reference_atoms': 32, 'replication': [6, 6, 6],
            'supercell_atoms': 6912, 'Cu_atoms': ncu, 'Ag_atoms': nag,
            'enthalpy_template': str(enthalpy_template), 'submission_template': str(submit_template),
            'note': 'For finite-T delta Hmix, use the matching functional and T endpoint runs '
                    'from this campaign and identical production averaging. Confirm crystal/order persists.'
        }
        (run / 'setup_metadata.json').write_text(json.dumps(provenance, indent=2) + '\n')
        created.append(run)
        print(f'CREATED {functional:6} Cu_{comp:<3} T{temp:<4} Cu={ncu:4} Ag={nag:4}')
    submit = destination / 'submit_new.sh'
    destination.mkdir(parents=True, exist_ok=True)
    with submit.open('w') as f:
        f.write('#!/usr/bin/env bash\nset -euo pipefail\n')
        for run in created:
            f.write(f'(cd {shlex.quote(str(run))} && sbatch submit_enthalpy)\n')
    submit.chmod(0o755)
    print(f'\nNew jobs: {len(created)}; existing (not modified): {len(skipped)}')
    for run in skipped:
        print(f'SKIPPED existing: {run.relative_to(root)}')
    print(f'Submission script (new jobs ONLY): {submit}')
    print('No jobs submitted automatically. Inspect generated inputs before running submit_new.sh.')


if __name__ == '__main__':
    main()
