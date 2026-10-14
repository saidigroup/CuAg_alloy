# Cu–Ag PBE/PBEsol Deep Potential reproducibility archive

This repository provides six Cu–Ag Deep Potential models, their DeePMD training configurations and learning curves, selected simulation/analysis scripts, and EOS structures for the accompanying Cu–Ag melting-depression study.

## Potential models

- PBE ensemble: `potentials/PBE/DNPs/Cu_Ag-9-x/Cu_Ag-9-{0,1,2}.pb`
- PBEsol ensemble: `potentials/PBEsol/DNPs/Cu_Ag-10-x/Cu_Ag-10-{0,1,2}.pb`
- Principal models described in the study: PBE `Cu_Ag-9-0.pb`; PBEsol `Cu_Ag-10-1.pb`.
- The binary `.pb` model files are tracked with Git LFS. Run `git lfs pull` after cloning if needed.
- Corresponding `out.json` (DeePMD effective training settings) and `lcurve.out` (learning curves) are under `training/PBE/` and `training/PBEsol/`, respectively.

## VASP reference calculations

- `dft/aimd/INCAR.ref.AIMD` is the supplied **reference/template** for AIMD data generation, not a claim that each historical AIMD run used identical values.
- It contains `GGA=PS` for PBEsol; use `GGA=PE` for PBE. Set `TEBEG` and `TEEND` to the intended run temperature. Check the actual run configuration before reusing.
- `dft/eos/` contains the staged fixed-volume EOS POSCAR configurations and representative calculation settings. Five Cu fractions (0, 25, 50, 75, 100 at.% Cu) and nine volume factors per functional are expected; selected base cells are also present.
- **POTCAR files are deliberately excluded** because VASP PAW datasets must be obtained from a licensed source. The PAW dataset selections must be matched to those documented in the manuscript/SI and verified against the original calculations.

## Cu–Ag dataset filtering

- `training/preprocessing/dpdata_CuAg_v6.py` is the user-supplied Python 3 Cu–Ag-only `dpdata` filtering/export utility.
- It restricts frames to elements Cu and Ag, applies electronic/force/energy filters, and generates DeepMD raw/npy datasets.
- It uses DeepMD `type_map = ["Cu", "Ag"]`. Confirm that the selected models and LAMMPS atom-type mappings use this ordering before deploying.
- Example to inspect options (requires `numpy` and `dpdata`): `python training/preprocessing/dpdata_CuAg_v6.py --help`.
- The default output directory is a relative path; **set `--base-out` explicitly** when running. The exact settings and input frames used for the published training run have not been independently recovered here.

## Other workflows

The staged collection includes representative TPC, EOS, diffusion, viscosity and associated analysis files. Some scripts may retain original HPC paths or require external starting structures; see `INPUT_DEPENDENCY_AUDIT.tsv` after running `tools/audit_inputs.py`.

## Reproducibility limits

The archive is intended as a compact set of representative inputs, trained models, and methods. It does not claim to contain every original AIMD frame, all MD trajectories, or a completely self-contained execution environment. Do not imply that an original LAMMPS input is portable until its data/restart/model paths have been checked. Original training dataset provenance and full reference AIMD settings still warrant confirmation.

## Software

The study used DeePMD-kit 2.2.10 and VASP 6.4.1 for the methods described in the manuscript/SI. Processing requires Python 3 and packages imported by each script; LAMMPS runs require a compatible DeePMD interface. Full version pinning, structure dependencies, and executable environment specifications should be added after the final audit.
