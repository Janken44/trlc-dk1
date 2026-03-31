"""Compare simulated vs measured J2 gravity torque at calibration poses.

For each plateau in calibration_plateaus.csv:
  - Set MuJoCo qpos to the measured joint angles
  - Run mj_inverse (zero velocity/acceleration) to get gravity-only torques
  - Compare simulated J2 torque against the measured value

Outputs a comparison table and saves compare_results.csv.

Usage:
    uv run python payload_measurement/recordings/data-extraction/compare_simulation.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

CSV_IN  = HERE / "calibration_plateaus.csv"
CSV_OUT = HERE / "compare_results.csv"

URDF_PATH = REPO_ROOT / "urdf/follower/TRLC-DK1-Follower.urdf"

# ---------------------------------------------------------------------------
# Load model via GravityCompensator (handles mesh-stripping correctly)
# ---------------------------------------------------------------------------
from trlc_dk1_control.gravity_comp import GravityCompensator

gc = GravityCompensator(str(URDF_PATH))
model = gc.mj_model
data  = gc.mj_data

def simulated_torques(qpos6: np.ndarray) -> np.ndarray:
    """Return qfrc_inverse[0:6] for the given 6-DOF arm pose (gravity only)."""
    return gc.compute(qpos6)

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_IN)

pos_cols = [f"j{i}_pos_rad" for i in range(1, 7)]
torque_cols = [f"j{i}_torque_Nm" for i in range(1, 7)]

rows = []
for _, row in df.iterrows():
    qpos = row[pos_cols].values.astype(float)
    meas_torques = row[torque_cols].values.astype(float)

    sim_torques = simulated_torques(qpos)

    # J2 is the key comparison joint
    meas_j2 = meas_torques[1]
    sim_j2   = sim_torques[1]
    ratio    = sim_j2 / meas_j2 if abs(meas_j2) > 0.2 else float("nan")

    rows.append({
        "reach_mm":       row["reach_mm"],
        "j2_pos_rad":     round(qpos[1], 4),
        "j3_pos_rad":     round(qpos[2], 4),
        "meas_j2_Nm":     round(meas_j2, 3),
        "sim_j2_Nm":      round(sim_j2, 3),
        "error_Nm":       round(sim_j2 - meas_j2, 3),
        "ratio_sim_meas": round(ratio, 3) if not np.isnan(ratio) else float("nan"),
        # Include all joints for completeness
        "meas_j1_Nm": round(meas_torques[0], 3),
        "sim_j1_Nm":  round(sim_torques[0], 3),
        "meas_j3_Nm": round(meas_torques[2], 3),
        "sim_j3_Nm":  round(sim_torques[2], 3),
        "meas_j4_Nm": round(meas_torques[3], 3),
        "sim_j4_Nm":  round(sim_torques[3], 3),
    })

results = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== J2 Torque: Simulated vs Measured ===\n")
print(results[["reach_mm", "j2_pos_rad", "j3_pos_rad",
               "meas_j2_Nm", "sim_j2_Nm", "error_Nm", "ratio_sim_meas"]].to_string(index=False))

valid = results["ratio_sim_meas"].dropna()
print(f"\nMean ratio (sim/meas): {valid.mean():.3f}  ±  {valid.std():.3f}")
print(f"Median ratio:          {valid.median():.3f}")

print("\n=== All Joints: Simulated vs Measured ===\n")
for j, idx in [("J1", 0), ("J3", 2), ("J4", 3)]:
    meas_col = f"meas_j{idx+1}_Nm"
    sim_col  = f"sim_j{idx+1}_Nm"
    errs = results[sim_col] - results[meas_col]
    print(f"  {j}: mean error = {errs.mean():.3f} Nm,  std = {errs.std():.3f} Nm")

results.to_csv(CSV_OUT, index=False)
print(f"\nSaved → {CSV_OUT}")
