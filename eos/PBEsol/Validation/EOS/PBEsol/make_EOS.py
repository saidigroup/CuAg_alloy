#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess

# ---------------- USER SETTINGS ----------------
ROOT = Path('.')
#SYSTEMS = ['Cu_100', 'Cu_50', 'Cu_0']
SYSTEMS = ['Cu_25', 'Cu_75']
VFACTORS = [0.94, 0.955, 0.97, 0.985, 1.00, 1.015, 1.03, 1.045, 1.06]
REF_MPI_JOB = ROOT / 'mpi_job'
REF_INCAR = ROOT / 'INCAR.EOS.ref'
POTCAR_CU = ROOT / 'POTCAR.Cu'
POTCAR_AG = ROOT / 'POTCAR.Ag'

SUBMIT_JOBS = True   # set to False to only generate folders/files
# ------------------------------------------------


def read_poscar(path: Path):
    lines = path.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f'{path} is too short to be a POSCAR/CONTCAR')

    comment = lines[0]
    scale = float(lines[1].split()[0])
    lattice = [list(map(float, lines[i].split()[:3])) for i in range(2, 5)]

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]

    coord_line_idx = 7
    selective = False
    if lines[7].strip().lower().startswith('s'):
        selective = True
        coord_line_idx = 8

    coord_mode = lines[coord_line_idx].strip()
    first_coord_idx = coord_line_idx + 1
    natoms = sum(counts)
    coords = lines[first_coord_idx:first_coord_idx + natoms]

    return {
        'comment': comment,
        'scale': scale,
        'lattice': lattice,
        'species': species,
        'counts': counts,
        'selective': selective,
        'coord_mode': coord_mode,
        'coords': coords,
    }


def write_poscar(data, path: Path, vfactor: float):
    sf = vfactor ** (1.0 / 3.0)
    lat = data['lattice']
    scaled_lat = [[x * sf for x in vec] for vec in lat]

    comment = f"{data['comment']} | EOS scaled volume factor {vfactor:.3f}"
    out = [comment, '1.0']
    out += ['{:18.10f} {:18.10f} {:18.10f}'.format(*vec) for vec in scaled_lat]
    out.append(' '.join(data['species']))
    out.append(' '.join(str(x) for x in data['counts']))
    if data['selective']:
        out.append('Selective dynamics')
    out.append(data['coord_mode'])
    out.extend(data['coords'])
    path.write_text('\n'.join(out) + '\n')


def build_potcar(system: str, dest: Path):
    if system == 'Cu_100':
        shutil.copy2(POTCAR_CU, dest / 'POTCAR')
    elif system == 'Cu_0':
        shutil.copy2(POTCAR_AG, dest / 'POTCAR')
    elif system == 'Cu_25':
        with open(dest / 'POTCAR', 'wb') as w:
            w.write(POTCAR_CU.read_bytes())
            w.write(POTCAR_AG.read_bytes())
    elif system == 'Cu_75':
        with open(dest / 'POTCAR', 'wb') as w:
            w.write(POTCAR_CU.read_bytes())
            w.write(POTCAR_AG.read_bytes())
    else:
        raise ValueError(f'Unknown system: {system}')


def submit_job(dest: Path):
    try:
        result = subprocess.run(
            ['qsub', 'mpi_job'],
            cwd=dest,
            check=True,
            capture_output=True,
            text=True
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stdout:
            print(f'Submitted in {dest}: {stdout}')
        else:
            print(f'Submitted in {dest}')
        if stderr:
            print(f'  qsub stderr: {stderr}')
    except subprocess.CalledProcessError as e:
        print(f'ERROR submitting job in {dest}')
        if e.stdout.strip():
            print(f'  stdout: {e.stdout.strip()}')
        if e.stderr.strip():
            print(f'  stderr: {e.stderr.strip()}')
    except FileNotFoundError:
        print('ERROR: qsub command not found. Are you on the submit node / is PBS loaded?')


def main():
    missing = [p for p in [REF_MPI_JOB, REF_INCAR, POTCAR_CU, POTCAR_AG] if not p.exists()]
    if missing:
        print('Missing reference files:')
        for p in missing:
            print('  ', p)
        print('\nCreate or copy those files into the working directory and rerun.')
        raise SystemExit(1)

    for system in SYSTEMS:
        contcar_src = ROOT / system / 'CONTCAR'
        poscar_src = ROOT / system / 'POSCAR'

        if contcar_src.exists():
            source_file = contcar_src
        elif poscar_src.exists():
            source_file = poscar_src
            print(f'Warning: {contcar_src} not found for {system}; using {poscar_src} instead.')
        else:
            print(f'Skipping {system}: missing both {contcar_src} and {poscar_src}')
            continue

        data = read_poscar(source_file)

        for vf in VFACTORS:
            sub = ROOT / system / ('V_' + ('{:.3f}'.format(vf).rstrip('0').rstrip('.')))
            sub.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source_file, sub / 'POSCAR')
            write_poscar(data, sub / 'POSCAR', vf)
            shutil.copy2(REF_MPI_JOB, sub / 'mpi_job')
            shutil.copy2(REF_INCAR, sub / 'INCAR')
            build_potcar(system, sub)

            if SUBMIT_JOBS:
                submit_job(sub)

    print('EOS folders generated successfully.')


if __name__ == '__main__':
    main()
