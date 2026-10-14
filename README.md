# Cu–Ag Alloy: PBE and PBEsol Computational Archive

This repository accompanies a computational study of melting and alloy thermodynamics in the Cu–Ag binary system using density-functional-theory (DFT)-trained Deep Potential models. It provides selected input files, model checkpoints, training records, and analysis scripts supporting comparisons between the PBE and PBEsol exchange–correlation functionals.

**Scope:** This is a curated reproducibility archive containing the trained potentials and representative computational workflows. It is **not** a complete copy of every production trajectory, DFT calculation, or intermediate training dataset.

## Repository contents

| Location | Contents |
| --- | --- |
| `potentials/PBE/DNPs/Cu_Ag-9-x/` | Three frozen PBE Deep Potential models (`Cu_Ag-9-0.pb` through `Cu_Ag-9-2.pb`) |
| `potentials/PBEsol/DNPs/Cu_Ag-10-x/` | Three frozen PBEsol models (`Cu_Ag-10-0.pb` through `Cu_Ag-10-2.pb`) |
| `training/PBE/`, `training/PBEsol/` | Effective training configurations (`out.json`) and learning curves (`lcurve.out`) for the six models |
| `training/preprocessing/` | Cu–Ag training-data preprocessing utility |
| `lammps/tpc/` | Representative two-phase coexistence (TPC) melting inputs for PBE and PBEsol |
| `lammps/diffusion/` | Representative diffusion inputs for PBE and PBEsol |
| `lammps/viscosity/` | Representative viscosity inputs for PBE and PBEsol |
| `lammps/eos/` | Representative LAMMPS equation-of-state input |
| `structures/TPC/`, `structures/liquid/` | Representative Cu50Ag50 LAMMPS starting structures |
| `dft/eos/`, `eos/` | Selected VASP EOS reference inputs and structures, organized by functional and composition |
| `phonons/PBEsol/Cu_50/4x4x4/` | Representative PBEsol Cu50Ag50 Phonopy example, including thermal properties and, where present, displacement force evaluations |
| `ordered_solids/phonons/` | Supporting Phonopy drivers and analysis utilities |
| `transport/`, `thermophysical/`, `coexistence/` | Selected workflow and postprocessing scripts |
| `tools/` | Repository utilities and input-audit scripts |

Some paths retain the original campaign-oriented directory organization. Consult individual files to determine the intended execution directory and input dependencies.

## Models and file storage

Six **uncompressed** frozen Deep Potential models are distributed using **Git Large File Storage (Git LFS)**. Install Git LFS before cloning or pulling model binaries:

```bash
git lfs install
git clone https://github.com/saidigroup/CuAg_alloy.git
cd CuAg_alloy
git lfs pull
```

The original production diffusion and viscosity calculations referenced compressed Deep Potential files (with names ending in `c.pb`). Those compressed model files are **not** included to limit repository storage. The example scripts instead permit selection of the corresponding archived uncompressed model files. Results from these examples should not be assumed numerically identical to the original compressed-model production trajectories without validation.

## Representative LAMMPS workflows

The archived LAMMPS inputs use a command-line variable for the model:

```lammps
pair_style deepmd ${model}
```

For a representative PBEsol diffusion calculation, from a fresh working directory with a compatible Cu–Ag `in.lmp` structure available under that filename:

```bash
lmp -var model /absolute/path/to/CuAg_alloy/potentials/PBEsol/DNPs/Cu_Ag-10-x/Cu_Ag-10-1.pb \
    -in /absolute/path/to/CuAg_alloy/lammps/diffusion/PBEsol/diffusion.in
```

The same approach can be used for the PBE model (`Cu_Ag-9-0.pb`) and the representative TPC, viscosity, and EOS inputs. Some scripts require additional runtime files (for example, `data.lmp` or `in.lmp`) and may produce substantial output. Inspect the scripts and adjust the structure, working directory, executable, simulation duration, and resources before running. Model species/type ordering must match the selected LAMMPS input and model.

These files are **representative examples**, not one-command reruns of all simulations reported in the article.

## DFT EOS reference calculations

Selected EOS VASP `POSCAR` and `INCAR` files are archived for the Cu–Ag compositions studied with PBE and PBEsol. `POTCAR` files are intentionally excluded because their redistribution is governed by the relevant pseudopotential license. Users must supply appropriately licensed VASP potentials and verify species ordering.

**Provenance note:** The archived EOS `INCAR` files are reference inputs. In any future cleanup, changing their electronic convergence setting to `EDIFF = 1E-7` is a normalization of the *repository examples*, not evidence that the original production calculations were performed with that setting. The original production settings should be established from the corresponding VASP output (`OUTCAR` or `vasprun.xml`) if exact historical reproduction is required.

## Representative Phonopy calculation

The `phonons/PBEsol/Cu_50/4x4x4/` example concerns a **4 × 4 × 4 supercell** for Cu50Ag50 treated using the PBEsol-trained potential. `thermal_properties.dat` contains temperature-dependent harmonic thermodynamic properties, including the **700 K** evaluation point. The 700 K value is a thermodynamic evaluation temperature, not the temperature of the underlying finite-displacement force calculation.

The example includes the unit cell, displacement description, driver, metadata, and available displacement/force files. The large `phonopy_params.yaml` archive (approximately 1.06 GB) is intentionally omitted. Users should inspect the driver and force/displacement completeness before assuming they can regenerate every reported Phonopy output exactly.

## Reproducibility and limitations

- Production melting, transport, and phase-equilibrium datasets may involve multiple compositions, temperatures, and independent replicas; only selected examples are included here.
- PBE and PBEsol potentials must not be interchanged when attempting to reproduce functional-specific results.
- Exact results can depend on software versions, initial configurations, random seeds, runtime parameters, and compressed versus uncompressed potential evaluation.
- Archived VASP inputs do not include pseudopotentials (`POTCAR`).
- Individual analysis scripts may assume original HPC directory layouts; review paths and dependencies before running.

## Software

The workflows use combinations of **LAMMPS**, **DeePMD-kit**, **VASP** (for reference DFT inputs), **Phonopy**, and Python analysis packages. Refer to the paper, supporting information, and individual scripts for methodology and version details. No universal environment specification is supplied here.

## Citation

If you use this archive, please cite the associated Cu–Ag PBE/PBEsol publication and link to this repository:

**Repository:** https://github.com/saidigroup/CuAg_alloy

A versioned release/archival DOI can be added once available. For reproducible referencing, record the Git commit or release tag used.

## License and acknowledgments

No software or data redistribution license is asserted by this README. The repository maintainers should add an appropriate `LICENSE` file or licensing statement before encouraging third-party reuse. VASP pseudopotentials are not redistributed.
