#!/usr/bin/env python3
"""Collect PBE/PBEsol Cu-Ag phonons; compare harmonic vibrational mixing terms.
Run from Validation: python compare_PBE_PBEsol_phonopy.py
Requires numpy and completed phonopy_metadata.json / thermal_properties.dat.
Does NOT combine static DFT/DNP energies or calculate liquid free energies.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
CAMPAIGNS = {
    "PBE_4": ROOT / "PBE_phonopy_4x4x4_relaxed",
    "PBEsol_4": ROOT / "PBEsol_phonopy_4x4x4",
    "PBEsol_6": ROOT / "PBEsol_phonopy_6x6x6",
}
COMPS = (0, 25, 50, 75, 100)
SELECTED_T = (900, 970, 1000, 1100, 1200, 1400, 1500)
EXPECTED_MODEL_NAMES = {"PBE_4": "Cu_Ag-9-0.pb", "PBEsol_4": "Cu_Ag-10-1.pb", "PBEsol_6": "Cu_Ag-10-1.pb"}
OUT = ROOT / "collected_PBE_vs_PBEsol_phonopy"
OUT.mkdir(exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_csv(name, rows, columns):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def load_run(label, comp):
    folder = CAMPAIGNS[label] / f"Cu_{comp}"
    mf, tf, sf = folder / "phonopy_metadata.json", folder / "thermal_properties.dat", folder / "unitcell.lmp"
    if not mf.is_file() or not tf.is_file():
        return None, 'MISSING: ' + ', '.join(str(p.name) for p in (mf,tf) if not p.is_file())
    try:
        m = json.loads(mf.read_text())
        a = np.loadtxt(tf, comments='#', ndmin=2)
        if a.ndim != 2 or a.shape[1] != 7 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError('invalid seven-column thermal_properties.dat')
        if np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError('temperature grid not strictly increasing')
        expected_dim = 6 if label.endswith('_6') else 4
        if m['supercell_matrix'] != [expected_dim] * 3:
            raise ValueError('incorrect supercell dimensions')
        if m['supercell_atoms'] != m['reference_atoms'] * expected_dim**3:
            raise ValueError('atom count inconsistent with supercell')
        if abs(float(m['Cu_at_percent']) - comp) > 0.01:
            raise ValueError('composition mismatch')
        if m.get('type_mapping') != {'1':'Cu', '2':'Ag'}:
            raise ValueError('type mapping not explicitly Cu / Ag')
        if Path(m['model']).name != EXPECTED_MODEL_NAMES[label]:
            raise ValueError('wrong potential filename: ' + m['model'])
        if int(m['number_modes_below_minus_0p10_THz']) != 0:
            raise ValueError('unstable: frequency modes below -0.1 THz')
        if float(m['minimum_frequency_THz']) < -1e-6:
            raise ValueError('negative mesh frequency: inspect imaginary modes')
        return {'meta':m, 'data':a, 'cell_hash':sha256(sf) if sf.is_file() else '', 'folder':folder}, 'COMPLETE'
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as e:
        return None, f'INVALID: {e}'


runs, statuses = {}, []
for label in CAMPAIGNS:
    for comp in COMPS:
        run, status = load_run(label, comp)
        runs[(label, comp)] = run
        statuses.append({'campaign':label,'Cu_at_percent':comp,'status':status,
                         'supercell_atoms':run['meta']['supercell_atoms'] if run else '',
                         'min_frequency_THz':run['meta']['minimum_frequency_THz'] if run else '',
                         'model':run['meta']['model'] if run else '',
                         'unitcell_sha256':run['cell_hash'] if run else ''})
        print(f'{label:10s} Cu_{comp:<3d} {status}')
write_csv('run_status.csv', statuses, list(statuses[0]))

warnings = []
for comp in COMPS:
    r4, r6 = runs['PBEsol_4',comp], runs['PBEsol_6',comp]
    if r4 and r6:
        if r4['cell_hash'] and r6['cell_hash'] and r4['cell_hash'] != r6['cell_hash']:
            warnings.append(f'Cu_{comp}: PBEsol 4x4 and 6x6 reference cells DIFFER by SHA256')
        model4, model6 = Path(r4['meta']['model']), Path(r6['meta']['model'])
        if model4.is_file() and model6.is_file():
            if sha256(model4) != sha256(model6):
                warnings.append(f'Cu_{comp}: PBEsol 4x4 and 6x6 POTENTIAL BYTES DIFFER')
        else:
            warnings.append(f'Cu_{comp}: unable to hash at least one PBEsol potential (model paths differ; verify separately)')

# Validate common thermodynamic settings, model and reference cell consistency within each functional.
for label in CAMPAIGNS:
    active = [runs[label,c] for c in COMPS if runs[label,c]]
    if active:
        params = {(tuple(r['meta']['mesh']),r['meta']['displacement_distance_A']) for r in active}
        models = {r['meta']['model'] for r in active}
        if len(params) > 1: warnings.append(f'{label}: q mesh/displacement inconsistent within campaign')
        if len(models) > 1: warnings.append(f'{label}: more than one model path within campaign')

rows_full, selected, cross = [], [], []
EV_TO_JMOL = 96485.33212331002


def identical_grid(*arrays):
    t0 = arrays[0][:,0]
    return all(len(a) == len(t0) and np.allclose(a[:,0],t0,rtol=0,atol=1e-8) for a in arrays[1:])

mix = {}
for label in CAMPAIGNS:
    ag, cu = runs[label,0], runs[label,100]
    if not ag or not cu:
        warnings.append(f'{label}: cannot calculate mixing without both endpoints')
        continue
    for comp in (25,50,75):
        alloy = runs[label,comp]
        if not alloy: continue
        a,b,c = alloy['data'],ag['data'],cu['data']
        if not identical_grid(a,b,c):
            warnings.append(f'{label} Cu_{comp}: temperature grids differ, skipping')
            continue
        x=comp/100
        # From original driver: F in eV/atom column 2; S,Cv in J mol^-1(atom) K^-1 columns 4,6.
        df = a[:,2]-(1-x)*b[:,2]-x*c[:,2]
        ds = a[:,4]-(1-x)*b[:,4]-x*c[:,4]
        dcv = a[:,6]-(1-x)*b[:,6]-x*c[:,6]
        du_mev = df*1000 + a[:,0]*ds/EV_TO_JMOL*1000  # U=F+TS, meV/atom
        mix[label,comp] = {'t':a[:,0], 'f':df*1000, 's':ds, 'cv':dcv, 'u':du_mev}
        for i,t in enumerate(a[:,0]):
            row={'functional_supercell':label,'Cu_at_percent':comp,'T_K':t,
                 'Delta_Fvib_mix_meV_atom':df[i]*1000,
                 'Delta_Svib_mix_J_mol_atom_K':ds[i],
                 'Delta_Uvib_mix_meV_atom':du_mev[i],
                 'Delta_Cv_mix_J_mol_atom_K':dcv[i]}
            rows_full.append(row)
            if any(abs(t-target)<1e-6 for target in SELECTED_T):selected.append(row.copy())
write_csv('mixing_all_temperatures.csv',rows_full,
          ['functional_supercell','Cu_at_percent','T_K','Delta_Fvib_mix_meV_atom','Delta_Svib_mix_J_mol_atom_K','Delta_Uvib_mix_meV_atom','Delta_Cv_mix_J_mol_atom_K'])
write_csv('selected_temperatures.csv',selected,
          ['functional_supercell','Cu_at_percent','T_K','Delta_Fvib_mix_meV_atom','Delta_Svib_mix_J_mol_atom_K','Delta_Uvib_mix_meV_atom','Delta_Cv_mix_J_mol_atom_K'])

# Match PBE and PBEsol at identical composition and T. 6x6 is primary PBEsol comparator;
# also keep 4x4 vs 4x4 comparison to isolate functional from supercell effects.
for comp in (25,50,75):
    pbe = mix.get(('PBE_4',comp))
    if pbe is None:continue
    for ref in ('PBEsol_4','PBEsol_6'):
        sol=mix.get((ref,comp))
        if sol is None:continue
        if len(pbe['t'])!=len(sol['t']) or not np.allclose(pbe['t'],sol['t'],rtol=0,atol=1e-8):
            warnings.append(f'Cu_{comp}: PBE_4 vs {ref} temperature-grid mismatch; skipping')
            continue
        for i,t in enumerate(pbe['t']):
            row={'Cu_at_percent':comp,'T_K':t,'PBEsol_campaign':ref,
                 'PBE_DeltaF_mix_meV_atom':pbe['f'][i],
                 'PBEsol_DeltaF_mix_meV_atom':sol['f'][i],
                 'PBE_minus_PBEsol_DeltaF_meV_atom':pbe['f'][i]-sol['f'][i],
                 'PBE_DeltaS_mix_J_mol_atom_K':pbe['s'][i],
                 'PBEsol_DeltaS_mix_J_mol_atom_K':sol['s'][i],
                 'PBE_minus_PBEsol_DeltaS_J_mol_atom_K':pbe['s'][i]-sol['s'][i],
                 'PBE_DeltaU_mix_meV_atom':pbe['u'][i],
                 'PBEsol_DeltaU_mix_meV_atom':sol['u'][i],
                 'PBE_minus_PBEsol_DeltaU_meV_atom':pbe['u'][i]-sol['u'][i]}
            cross.append(row)
cols=['Cu_at_percent','T_K','PBEsol_campaign','PBE_DeltaF_mix_meV_atom','PBEsol_DeltaF_mix_meV_atom',
      'PBE_minus_PBEsol_DeltaF_meV_atom','PBE_DeltaS_mix_J_mol_atom_K','PBEsol_DeltaS_mix_J_mol_atom_K',
      'PBE_minus_PBEsol_DeltaS_J_mol_atom_K','PBE_DeltaU_mix_meV_atom','PBEsol_DeltaU_mix_meV_atom',
      'PBE_minus_PBEsol_DeltaU_meV_atom']
write_csv('PBE_vs_PBEsol_all_temperatures.csv',cross,cols)
write_csv('PBE_vs_PBEsol_selected_temperatures.csv',
          [r for r in cross if any(abs(r['T_K']-t)<1e-6 for t in SELECTED_T)],cols)

# Print most relevant paper temperatures; no rounding in CSV outputs.
print('\nPBE vs PBEsol @ selected temperatures (6x6 PBEsol primary):')
for r in cross:
    if r['PBEsol_campaign']=='PBEsol_6' and r['T_K'] in (970.,1000.,1400.,1500.):
        print(f"Cu{r['Cu_at_percent']:2d} {r['T_K']:4.0f} K | "
              f"PBE {r['PBE_DeltaF_mix_meV_atom']:+9.4f} | "
              f"PBEsol {r['PBEsol_DeltaF_mix_meV_atom']:+9.4f} | "
              f"PBE-PBEsol {r['PBE_minus_PBEsol_DeltaF_meV_atom']:+9.4f} meV/atom")
print('\nWarnings:')
for w in warnings:print(' -',w)
if not warnings:print(' none')
summary = {'complete_runs':sum(r is not None for r in runs.values()),'total_runs':len(runs),
           'comparison_rows':len(cross),'warnings':warnings,
           'scope':'harmonic fixed-reference crystalline vibrational mixing only; no configurational entropy, anharmonicity, or liquid thermodynamics'}
(OUT/'collection_report.json').write_text(json.dumps(summary,indent=2)+'\n')
print('\nOutput:',OUT)
if not any(runs['PBE_4',c] for c in COMPS):
    print('No PBE data found at PBE_phonopy_4x4x4_relaxed; adjust CAMPAIGNS at top if folder name differs.')
    sys.exit(2)
