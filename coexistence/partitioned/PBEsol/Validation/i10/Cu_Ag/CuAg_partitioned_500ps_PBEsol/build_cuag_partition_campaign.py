#!/usr/bin/env python3
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

SOURCE = Path('/mnt/data/cool1(1)')
SUBMIT_TEMPLATE = Path('/mnt/data/submit_melt')
OUTROOT = Path('/mnt/data/CuAg_partitioned_500ps_PBEsol')

# total Cu %, solid(top) Cu %, liquid(bottom) Cu %, Tm (K)
CASES = [
    (40, 30, 50, 979.6),
    (40, 20, 60, 979.6),
    (40, 10, 70, 979.6),
    (50, 40, 60, 974.8),
    (50, 30, 70, 974.8),
    (50, 20, 80, 974.8),
    (60, 70, 50, 1000.1),
    (60, 80, 40, 1000.1),
    (60, 90, 30, 1000.1),
]


def parse_lammps_data(path: Path):
    lines = path.read_text().splitlines()
    natoms = None
    zlo = zhi = None
    for line in lines:
        f = line.split()
        if len(f) == 2 and f[1] == 'atoms':
            natoms = int(f[0])
        elif len(f) >= 4 and f[-2:] == ['zlo', 'zhi']:
            zlo, zhi = float(f[0]), float(f[1])
    if natoms is None or zlo is None or zhi is None:
        raise RuntimeError('Could not parse atom count or z bounds')

    atoms_header = next(i for i, line in enumerate(lines) if line.startswith('Atoms'))
    i = atoms_header + 1
    while i < len(lines) and not lines[i].strip():
        i += 1

    records = {}
    indices = {}
    while i < len(lines) and len(records) < natoms:
        if not lines[i].strip():
            i += 1
            continue
        f = lines[i].split()
        if len(f) < 5:
            raise RuntimeError(f'Malformed atom line {i+1}: {lines[i]}')
        atom_id = int(f[0])
        atom_type = int(f[1])
        if atom_type not in (1, 2):
            raise RuntimeError(f'Unexpected atom type {atom_type}')
        records[atom_id] = f
        indices[atom_id] = i
        i += 1

    if len(records) != natoms:
        raise RuntimeError(f'Expected {natoms} atoms, parsed {len(records)}')
    return lines, records, indices, natoms, zlo, zhi


def best_counts(nsolid: int, nliquid: int, total_pct: float, solid_pct: float, liquid_pct: float):
    total_cu = round((total_pct / 100.0) * (nsolid + nliquid))
    lo = max(0, total_cu - nliquid)
    hi = min(nsolid, total_cu)
    best = None
    for ns_cu in range(lo, hi + 1):
        nl_cu = total_cu - ns_cu
        score = ((100.0 * ns_cu / nsolid - solid_pct) ** 2 +
                 (100.0 * nl_cu / nliquid - liquid_pct) ** 2)
        candidate = (score, ns_cu, nl_cu)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return total_cu, best[1], best[2]


def make_nph_input(tm: float, velocity_seed: int) -> str:
    return f'''# Cu-Ag two-phase NPH coexistence run\n# Starts directly from a prebuilt partitioned cool snapshot.\n\nclear\nunits           metal\nboundary        p p p\natom_style      atomic\nbox             tilt large\n\nread_data       data.lmp\ntimestep        0.002\nnewton          on\n\npair_style      deepmd /ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au_sol/DNPs/Cu_Ag-10-x/Cu_Ag-10-1c.pb\npair_coeff      * *\n\nneighbor        3.0 bin\nneigh_modify    delay 0 every 1 check yes\n\nvariable        Tm       equal {tm:.1f}\nvariable        Press    equal 1.01325\nvariable        VelSeed  equal {velocity_seed}\n\n# Current physical phase assignment from the supplied snapshot:\n# lower half = liquid; upper half = solid.\nvariable        z_cen equal 0.5*(zlo+zhi)\nregion          liquid_region block INF INF INF INF INF ${{z_cen}} units box\ngroup           liquid region liquid_region\ngroup           solid subtract all liquid\n\ngroup           Cu type 1\ngroup           Ag type 2\ngroup           solidCu intersect solid Cu\ngroup           solidAg intersect solid Ag\ngroup           liquidCu intersect liquid Cu\ngroup           liquidAg intersect liquid Ag\n\nvariable        nTot equal atoms\nvariable        nCu equal count(Cu)\nvariable        nAg equal count(Ag)\nvariable        nSolid equal count(solid)\nvariable        nLiquid equal count(liquid)\nvariable        nSolidCu equal count(solidCu)\nvariable        nSolidAg equal count(solidAg)\nvariable        nLiquidCu equal count(liquidCu)\nvariable        nLiquidAg equal count(liquidAg)\n\nprint "Tm_K ${{Tm}}" file initial_phase_counts.dat screen no\nprint "velocity_seed ${{VelSeed}}" file initial_phase_counts.dat screen no append yes\nprint "total_atoms ${{nTot}}" file initial_phase_counts.dat screen no append yes\nprint "total_Cu ${{nCu}}" file initial_phase_counts.dat screen no append yes\nprint "total_Ag ${{nAg}}" file initial_phase_counts.dat screen no append yes\nprint "solid_atoms ${{nSolid}}" file initial_phase_counts.dat screen no append yes\nprint "solid_Cu ${{nSolidCu}}" file initial_phase_counts.dat screen no append yes\nprint "solid_Ag ${{nSolidAg}}" file initial_phase_counts.dat screen no append yes\nprint "liquid_atoms ${{nLiquid}}" file initial_phase_counts.dat screen no append yes\nprint "liquid_Cu ${{nLiquidCu}}" file initial_phase_counts.dat screen no append yes\nprint "liquid_Ag ${{nLiquidAg}}" file initial_phase_counts.dat screen no append yes\n\ncompute         LiquidTemp liquid temp\ncompute         SolidTemp solid temp\ncompute         peatom all pe/atom\ncompute         PEs solid reduce sum c_peatom\ncompute         PEl liquid reduce sum c_peatom\nvariable        pe_solid equal c_PEs/count(solid)\nvariable        pe_liquid equal c_PEl/count(liquid)\n\n# Initialize the random-composition two-phase cell at the composition-specific Tm.\nreset_timestep  0\nvelocity        all create ${{Tm}} ${{VelSeed}} mom yes rot no dist gaussian\nrun             0 post no\n\n# Structural and composition profiles along z.\ncompute         cs all centro/atom 12\nvariable        cs_cut equal 2.5\nvariable        solidflag atom "c_cs < v_cs_cut"\nvariable        cuflag atom "type == 1"\ncompute         ssum all reduce sum v_solidflag\nvariable        f_solid equal c_ssum/atoms\ncompute         zbin all chunk/atom bin/1d z lower 0.02 units reduced\nfix             prof all ave/chunk 50 50 2500 zbin v_solidflag v_cuflag file nph_structure_composition_profile.dat\n\nvariable        T equal temp\nvariable        P equal press\nvariable        Vol equal vol\nvariable        PE equal pe\nvariable        KE equal ke\nfix             nphlog all ave/time 50 50 2500 v_f_solid c_LiquidTemp c_SolidTemp v_T v_P v_Vol v_PE v_KE v_pe_solid v_pe_liquid file nph_timeseries.dat\n\ncompute         msdL liquid msd\nfix             msdout liquid ave/time 100 50 5000 c_msdL[4] file nph_msd_liquid.dat\n\nthermo_style    custom step time temp press vol etotal pe ke c_LiquidTemp c_SolidTemp v_f_solid v_pe_solid v_pe_liquid\nthermo          2500\nthermo_modify   flush yes\necho            both\n\n# 500 ps at 2 fs = 250,000 steps. NPH is intentionally unthermostatted.\nfix             nph all nph iso ${{Press}} ${{Press}} $(1000*dt)\ndump            dnph all custom 1000 Melt_nph.* id type x y z ix iy iz\n\nprint "== NPH: 500 ps production at initial Tm = ${{Tm}} K =="\nrun             250000\nprint "== NPH: DONE =="\n\nwrite_data      two_phase_500ps.lmp\nwrite_restart   two_phase_500ps.restart\n\nunfix           prof\nunfix           nphlog\nunfix           msdout\nunfix           nph\n'''


def make_submit(job_name: str) -> str:
    text = SUBMIT_TEMPLATE.read_text()
    text = text.replace('#SBATCH --job-name=Cu_Ag_50', f'#SBATCH --job-name={job_name}')
    text = text.replace('#SBATCH --time=16:00:00', '#SBATCH --time=48:00:00')
    text = text.replace('srun --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores lmp -sf opt -i melt.in > out_melt',
                        'srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores lmp -sf opt -i nph.in > out_nph')
    return text


def main():
    if OUTROOT.exists():
        shutil.rmtree(OUTROOT)
    OUTROOT.mkdir(parents=True)

    lines, records, indices, natoms, zlo, zhi = parse_lammps_data(SOURCE)
    zmid = 0.5 * (zlo + zhi)
    solid_ids = sorted(atom_id for atom_id, f in records.items() if float(f[4]) >= zmid)
    liquid_ids = sorted(atom_id for atom_id, f in records.items() if float(f[4]) < zmid)

    audits = []
    for total_pct, solid_pct, liquid_pct, tm in CASES:
        total_cu, solid_cu, liquid_cu = best_counts(
            len(solid_ids), len(liquid_ids), total_pct, solid_pct, liquid_pct
        )
        seed = int(f'{total_pct}{solid_pct:02d}{liquid_pct:02d}')
        rng = random.Random(seed)
        selected_solid_cu = set(rng.sample(solid_ids, solid_cu))
        selected_liquid_cu = set(rng.sample(liquid_ids, liquid_cu))

        out_lines = list(lines)
        for atom_id, f in records.items():
            newf = list(f)
            if atom_id in selected_solid_cu or atom_id in selected_liquid_cu:
                newf[1] = '1'
            else:
                newf[1] = '2'
            out_lines[indices[atom_id]] = ' '.join(newf)

        case_dir = OUTROOT / f'Cu_{total_pct}_total' / f'Cu_{solid_pct}_{liquid_pct}'
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'data.lmp').write_text('\n'.join(out_lines) + '\n')
        velocity_seed = 7000000 + seed
        (case_dir / 'nph.in').write_text(make_nph_input(tm, velocity_seed))
        (case_dir / 'submit_melt').write_text(make_submit(f'Cu{total_pct}_{solid_pct}_{liquid_pct}'))
        (case_dir / 'submit_melt').chmod(0o755)

        audits.append({
            'folder': str(case_dir.relative_to(OUTROOT)),
            'Tm_K': f'{tm:.1f}',
            'total_target_Cu_pct': total_pct,
            'solid_target_Cu_pct': solid_pct,
            'liquid_target_Cu_pct': liquid_pct,
            'total_atoms': natoms,
            'solid_atoms': len(solid_ids),
            'liquid_atoms': len(liquid_ids),
            'total_Cu_atoms': total_cu,
            'solid_Cu_atoms': solid_cu,
            'liquid_Cu_atoms': liquid_cu,
            'total_actual_Cu_pct': f'{100*total_cu/natoms:.6f}',
            'solid_actual_Cu_pct': f'{100*solid_cu/len(solid_ids):.6f}',
            'liquid_actual_Cu_pct': f'{100*liquid_cu/len(liquid_ids):.6f}',
            'composition_seed': seed,
            'velocity_seed': velocity_seed,
            'timestep_fs': '2.0',
            'nph_steps': 250000,
            'nph_duration_ps': '500.0',
        })

    fields = list(audits[0].keys())
    with (OUTROOT / 'composition_audit.tsv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(audits)

    submit_all = '''#!/usr/bin/env bash\nset -euo pipefail\nroot="$(cd "$(dirname "$0")" && pwd)"\nfind "$root" -mindepth 3 -maxdepth 3 -type f -name submit_melt -print0 | sort -z | while IFS= read -r -d '' script; do\n    d="$(dirname "$script")"\n    echo "Submitting $d"\n    (cd "$d" && sbatch ./submit_melt)\ndone\n'''
    (OUTROOT / 'submit_all.sh').write_text(submit_all)
    (OUTROOT / 'submit_all.sh').chmod(0o755)

    readme = f'''Cu-Ag PBEsol phase-partition perturbation campaign\n\nSource structure: {SOURCE.name}\nAtoms: {natoms}\nType mapping: type 1 = Cu, type 2 = Ag\nBox midpoint used for initial phases: z = {zmid:.12f} Angstrom\nSolid/top: z >= midpoint ({len(solid_ids)} atoms)\nLiquid/bottom: z < midpoint ({len(liquid_ids)} atoms)\n\nIncluded systems (solid Cu % / liquid Cu %):\n  Cu_40_total: 30/50, 20/60, 10/70\n  Cu_50_total: 40/60, 30/70, 20/80\n  Cu_60_total: 70/50, 80/40, 90/30\n\nExcluded as previously available baselines: 40/40, 50/50, and 60/60.\n\nPBEsol random-alloy Tm values used to initialize velocities:\n  40 at.% total Cu: 979.6 K\n  50 at.% total Cu: 974.8 K\n  60 at.% total Cu: 1000.1 K\n\nEach directory contains:\n  data.lmp    randomized Cu/Ag identities at exact integer phase counts\n  nph.in      direct 500 ps NPH coexistence run, timestep = 0.002 ps\n  submit_melt Slurm GPU submission script, updated to run nph.in\n\nThe input starts directly from the supplied two-phase cool snapshot. It does not\nminimize a single crystal, replicate the cell, melt a half, or cool it again.\nVelocities are regenerated for all atoms at the composition-specific Tm before\nthe NPH run. The NPH production length is 250000 steps = 500 ps.\n\nBecause the current top and bottom contain 6916 and 6908 atoms, respectively,\nand atom counts must be integral, requested percentages cannot always be exact.\nThe selected counts preserve the nearest possible total composition and minimize\nthe phase-composition error. See composition_audit.tsv.\n\nRun all jobs from this root with:\n  ./submit_all.sh\n'''
    (OUTROOT / 'README.txt').write_text(readme)

    # Include the exact generator for provenance/reproducibility.
    shutil.copy2(Path(__file__), OUTROOT / Path(__file__).name)


if __name__ == '__main__':
    main()
