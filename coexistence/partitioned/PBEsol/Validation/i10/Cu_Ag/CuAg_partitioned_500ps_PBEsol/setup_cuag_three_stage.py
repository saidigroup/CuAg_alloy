#!/usr/bin/env python3
"""Install a three-stage Cu-Ag coexistence input into existing partition folders.

Stages:
  1. 30 ps solid-coupled NPT; liquid evolves under NVE.
  2. 20 ps simultaneous, independent solid/liquid NVT.
  3. 300 ps whole-cell NPH.

Expected directory layout:
  ROOT/Cu_<total>_total/Cu_<solid>_<liquid>/data.lmp

The initial lower half is treated as liquid and the upper half as solid.
Group membership is fixed by atom IDs when the input is read.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Temperatures selected from the PBEsol random-alloy Tm table using ~10 K
# phase-preservation margins. Values are K.
TEMPERATURES: dict[tuple[int, int, int], tuple[float, float]] = {
    (40, 30, 50): (985.0, 985.0),
    (40, 20, 60): (1010.0, 1010.0),
    (40, 10, 70): (1062.0, 1062.0),
    (50, 40, 60): (970.0, 1010.0),
    (50, 30, 70): (997.0, 1062.0),
    (50, 20, 80): (1056.0, 1126.0),
    (60, 70, 50): (985.0, 985.0),
    (60, 80, 40): (990.0, 990.0),
    (60, 90, 30): (1017.0, 1017.0),
}

INPUT_TEMPLATE = r'''# ============================================================
# Cu-Ag partitioned two-phase preparation and coexistence
#
# Existing structure: data.lmp
# Initial lower half: liquid
# Initial upper half: solid
#
# Stage 1: 30 ps solid-coupled NPT + liquid NVE
# Stage 2: 20 ps independent solid/liquid NVT
# Stage 3: 300 ps whole-cell NPH
#
# IMPORTANT:
# In one connected periodic cell, LAMMPS cannot barostat only the
# solid's independent volume. During Stage 1 the solid is the NPT
# integration/thermostat group, the liquid is integrated with NVE,
# and the shared box is dilated for all atoms by the global barostat.
# ============================================================

clear
units           metal
dimension       3
boundary        p p p
atom_style      atomic
box             tilt large
newton          on

read_data       data.lmp

timestep        0.002             # ps = 2 fs

# ---------- PBEsol DeepMD potential ----------
pair_style      deepmd /ix3/walsaidi/cma94/Cu_Ni_Ag_Pd_Pt_Au_sol/DNPs/Cu_Ag-10-x/Cu_Ag-10-1c.pb
pair_coeff      * *

neighbor        3.0 bin
neigh_modify    delay 0 every 1 check yes

# ---------- run-specific targets ----------
variable        Tsolid  equal @TSOLID@
variable        Tliquid equal @TLIQUID@
variable        Press   equal 1.01325       # bar
variable        SeedS   equal @SEEDS@
variable        SeedL   equal @SEEDL@

# Damping constants in ps.
variable        Tdamp   equal 0.20
variable        Pdamp   equal 2.00

# ---------- freeze the initial phase membership by atom ID ----------
variable        z_cen equal 0.5*(zlo+zhi)
region          liquid_region block INF INF INF INF INF ${z_cen} units box
group           liquid region liquid_region
group           solid subtract all liquid

group           Cu type 1
group           Ag type 2
group           solidCu intersect solid Cu
group           solidAg intersect solid Ag
group           liquidCu intersect liquid Cu
group           liquidAg intersect liquid Ag

# ---------- initial counts ----------
run             0 post no

variable        nTot equal atoms
variable        nCu equal count(Cu)
variable        nAg equal count(Ag)
variable        nSolid equal count(solid)
variable        nLiquid equal count(liquid)
variable        nSolidCu equal count(solidCu)
variable        nSolidAg equal count(solidAg)
variable        nLiquidCu equal count(liquidCu)
variable        nLiquidAg equal count(liquidAg)
variable        xSolidCu equal count(solidCu)/count(solid)
variable        xLiquidCu equal count(liquidCu)/count(liquid)

print "Tsolid_K ${Tsolid}" file initial_phase_counts.dat screen yes
print "Tliquid_K ${Tliquid}" append initial_phase_counts.dat screen yes
print "solid_seed ${SeedS}" append initial_phase_counts.dat screen yes
print "liquid_seed ${SeedL}" append initial_phase_counts.dat screen yes
print "total_atoms ${nTot}" append initial_phase_counts.dat screen yes
print "total_Cu ${nCu}" append initial_phase_counts.dat screen yes
print "total_Ag ${nAg}" append initial_phase_counts.dat screen yes
print "solid_atoms ${nSolid}" append initial_phase_counts.dat screen yes
print "solid_Cu ${nSolidCu}" append initial_phase_counts.dat screen yes
print "solid_Ag ${nSolidAg}" append initial_phase_counts.dat screen yes
print "solid_Cu_fraction ${xSolidCu}" append initial_phase_counts.dat screen yes
print "liquid_atoms ${nLiquid}" append initial_phase_counts.dat screen yes
print "liquid_Cu ${nLiquidCu}" append initial_phase_counts.dat screen yes
print "liquid_Ag ${nLiquidAg}" append initial_phase_counts.dat screen yes
print "liquid_Cu_fraction ${xLiquidCu}" append initial_phase_counts.dat screen yes

# ---------- diagnostics ----------
compute         SolidTemp solid temp
compute         LiquidTemp liquid temp
compute         AllTemp all temp
compute         AllPress all pressure AllTemp
compute         peatom all pe/atom
compute         PEs solid reduce sum c_peatom
compute         PEl liquid reduce sum c_peatom
variable        pe_solid equal c_PEs/count(solid)
variable        pe_liquid equal c_PEl/count(liquid)

# Fixed centrosymmetry threshold is a trend diagnostic only.
compute         cs all centro/atom 12
variable        cs_cut equal 2.5
variable        solidflag atom "c_cs < v_cs_cut"
variable        cuflag atom "type == 1"
compute         ssum all reduce sum v_solidflag
variable        f_solid equal c_ssum/atoms

variable        T equal temp
variable        P equal press
variable        Vol equal vol
variable        Rho equal density
variable        PE equal pe
variable        KE equal ke

thermo_style    custom step time temp press pxx pyy pzz density vol lx ly lz pe ke etotal enthalpy c_SolidTemp c_LiquidTemp v_f_solid v_pe_solid v_pe_liquid
thermo          500
thermo_modify   flush yes lost error

echo            both
reset_timestep  0

# ============================================================
# STAGE 1: 30 ps solid-coupled NPT; liquid NVE
# 15,000 steps x 0.002 ps = 30 ps
# ============================================================

print "== STAGE 1: 30 ps solid-coupled NPT =="
print "Solid thermostat target = ${Tsolid} K"
print "Liquid begins at ${Tliquid} K and evolves under NVE"

velocity        solid create ${Tsolid} ${SeedS} mom yes rot no dist gaussian
velocity        liquid create ${Tliquid} ${SeedL} mom yes rot no dist gaussian
velocity        solid zero linear
velocity        liquid zero linear
velocity        solid scale ${Tsolid}
velocity        liquid scale ${Tliquid}

# The barostat acts on the shared periodic box and dilates all atoms.
# Only the initial-solid atom group is thermostatted by fix nptSolid.
fix             nptSolid solid npt temp ${Tsolid} ${Tsolid} ${Tdamp} iso ${Press} ${Press} ${Pdamp} drag 1.0 dilate all
fix_modify      nptSolid temp SolidTemp press AllPress
fix             nveLiquid liquid nve

compute         zbin1 all chunk/atom bin/1d z lower 0.02 units reduced
fix             prof1 all ave/chunk 50 50 2500 zbin1 v_solidflag v_cuflag file stage1_npt_profile.dat
fix             log1 all ave/time 50 50 2500 c_SolidTemp c_LiquidTemp v_T v_P v_Rho v_Vol v_PE v_KE v_f_solid v_pe_solid v_pe_liquid file stage1_npt_timeseries.dat

dump            d1 all custom 500 stage1_npt.* id type x y z ix iy iz vx vy vz c_cs
dump_modify     d1 sort id

run             15000

unfix           nptSolid
unfix           nveLiquid
unfix           prof1
unfix           log1
undump          d1
uncompute       zbin1

write_data      after_solid_npt_30ps.lmp
write_restart   after_solid_npt_30ps.restart

# ============================================================
# STAGE 2: 20 ps simultaneous independent NVT
# 10,000 steps x 0.002 ps = 20 ps
# ============================================================

print "== STAGE 2: 20 ps independent solid/liquid NVT =="

# Preserve the relaxed coordinates and cell from Stage 1, but place each
# initial phase exactly at its requested phase-preserving temperature.
velocity        all zero linear
velocity        solid scale ${Tsolid}
velocity        liquid scale ${Tliquid}

fix             nvtSolid solid nvt temp ${Tsolid} ${Tsolid} ${Tdamp}
fix_modify      nvtSolid temp SolidTemp
fix             nvtLiquid liquid nvt temp ${Tliquid} ${Tliquid} ${Tdamp}
fix_modify      nvtLiquid temp LiquidTemp

compute         zbin2 all chunk/atom bin/1d z lower 0.02 units reduced
fix             prof2 all ave/chunk 50 50 2500 zbin2 v_solidflag v_cuflag file stage2_nvt_profile.dat
fix             log2 all ave/time 50 50 2500 c_SolidTemp c_LiquidTemp v_T v_P v_Rho v_Vol v_PE v_KE v_f_solid v_pe_solid v_pe_liquid file stage2_nvt_timeseries.dat

dump            d2 all custom 500 stage2_nvt.* id type x y z ix iy iz vx vy vz c_cs
dump_modify     d2 sort id

run             10000

unfix           nvtSolid
unfix           nvtLiquid
unfix           prof2
unfix           log2
undump          d2
uncompute       zbin2

write_data      after_dual_nvt_20ps.lmp
write_restart   after_dual_nvt_20ps.restart

# ============================================================
# STAGE 3: 300 ps whole-cell NPH
# 150,000 steps x 0.002 ps = 300 ps
# ============================================================

print "== STAGE 3: 300 ps whole-cell NPH =="

# Production time starts at zero. No velocity recreation or rescaling here:
# NPH begins from the two phase temperatures produced by Stage 2.
reset_timestep  0
run             0 post no

print "NPH initial total temperature = $(temp) K"
print "NPH initial solid-group temperature = $(c_SolidTemp) K"
print "NPH initial liquid-group temperature = $(c_LiquidTemp) K"
print "NPH initial pressure = $(press) bar"

fix             nphRun all nph iso ${Press} ${Press} ${Pdamp} drag 0.5

compute         zbin3 all chunk/atom bin/1d z lower 0.02 units reduced
fix             prof3 all ave/chunk 50 50 2500 zbin3 v_solidflag v_cuflag file stage3_nph_structure_composition_profile.dat
fix             log3 all ave/time 50 50 2500 c_SolidTemp c_LiquidTemp v_T v_P v_Rho v_Vol v_PE v_KE v_f_solid v_pe_solid v_pe_liquid file stage3_nph_timeseries.dat

# MSD is for atoms that started in the liquid half, not a dynamic phase group.
compute         msdL liquid msd
fix             msdout liquid ave/time 100 50 5000 c_msdL[4] file stage3_nph_msd_initial_liquid.dat

dump            d3 all custom 1000 stage3_nph.* id type x y z ix iy iz vx vy vz c_cs
dump_modify     d3 sort id

thermo          2500
run             150000

print "== THREE-STAGE RUN COMPLETE =="

write_data      two_phase_nph_300ps.lmp
write_restart   two_phase_nph_300ps.restart

unfix           nphRun
unfix           prof3
unfix           log3
unfix           msdout
undump          d3
uncompute       zbin3
'''


def parse_case(path: Path) -> tuple[int, int, int] | None:
    """Extract (total, solid, liquid) from the expected folder names."""
    m_total = re.fullmatch(r"Cu_(\d+)_total", path.parent.name)
    m_pair = re.fullmatch(r"Cu_(\d+)_(\d+)", path.name)
    if not (m_total and m_pair):
        return None
    return int(m_total.group(1)), int(m_pair.group(1)), int(m_pair.group(2))


def render_input(total: int, solid: int, liquid: int, tsolid: float, tliquid: float) -> str:
    # Deterministic but distinct eight-digit seeds under LAMMPS's seed limit.
    seed_s = 10000000 + total * 10000 + solid * 100 + liquid
    seed_l = 20000000 + total * 10000 + liquid * 100 + solid
    return (
        INPUT_TEMPLATE
        .replace("@TSOLID@", f"{tsolid:.1f}")
        .replace("@TLIQUID@", f"{tliquid:.1f}")
        .replace("@SEEDS@", str(seed_s))
        .replace("@SEEDL@", str(seed_l))
    )


def patch_submit(path: Path) -> None:
    text = path.read_text()
    original = text
    # Change any known LAMMPS input/output invocation.
    text = re.sub(r"-i\s+nph\.in\s*>\s*out_nph\b", "-i three_stage.in > out_three_stage", text)
    text = re.sub(r"-in\s+nph\.in\s*>\s*out_nph\b", "-in three_stage.in > out_three_stage", text)
    if text == original and "three_stage.in" not in text:
        text = text.rstrip() + "\n\n# Three-stage Cu-Ag run\nsrun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores lmp -sf opt -i three_stage.in > out_three_stage\n"
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."), help="Campaign root")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing three_stage.in")
    parser.add_argument("--patch-submit", action="store_true", help="Patch submit_melt to run three_stage.in")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    found: list[tuple[Path, tuple[int, int, int]]] = []
    for data_file in root.glob("Cu_*_total/Cu_*_*/data.lmp"):
        case = parse_case(data_file.parent)
        if case in TEMPERATURES:
            found.append((data_file.parent, case))

    if not found:
        print(f"ERROR: no recognized data.lmp files below {root}", file=sys.stderr)
        return 2

    found.sort(key=lambda item: item[1])
    audit_lines = ["folder\ttotal_Cu_atpct\tsolid_Cu_atpct\tliquid_Cu_atpct\tTsolid_K\tTliquid_K"]

    for folder, case in found:
        total, solid, liquid = case
        tsolid, tliquid = TEMPERATURES[case]
        target = folder / "three_stage.in"
        audit_lines.append(f"{folder.relative_to(root)}\t{total}\t{solid}\t{liquid}\t{tsolid:.1f}\t{tliquid:.1f}")

        action = "would write" if args.dry_run else "write"
        print(f"{action}: {target.relative_to(root)}  Tsolid={tsolid:.1f} K  Tliquid={tliquid:.1f} K")
        if args.dry_run:
            continue

        if target.exists() and not args.overwrite:
            print(f"  skip existing (use --overwrite): {target}")
        else:
            target.write_text(render_input(total, solid, liquid, tsolid, tliquid))

        if args.patch_submit:
            submit = folder / "submit_melt"
            if submit.exists():
                backup = folder / "submit_melt.before_three_stage"
                if not backup.exists():
                    shutil.copy2(submit, backup)
                patch_submit(submit)
                submit.chmod(submit.stat().st_mode | 0o111)
            else:
                print(f"  WARNING: missing {submit}")

    if not args.dry_run:
        (root / "three_stage_temperature_plan.tsv").write_text("\n".join(audit_lines) + "\n")
        submit_all = root / "submit_all_three_stage.sh"
        submit_all.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "root=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
            "while IFS= read -r -d '' script; do\n"
            "  d=\"$(dirname \"$script\")\"\n"
            "  echo \"Submitting $d\"\n"
            "  (cd \"$d\" && sbatch submit_melt)\n"
            "done < <(find \"$root\" -mindepth 3 -maxdepth 3 -name submit_melt -print0 | sort -z)\n"
        )
        submit_all.chmod(0o755)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
