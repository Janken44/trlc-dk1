"""
Programmatic payload sweep — IK-planned horizontal reach waypoints.

Computes joint-space configurations for a series of horizontal reach distances
via pyroki IK, then executes the sweep autonomously: transit to each waypoint,
dwell, record. No leader arm required.

IK strategy
-----------
Fix joint1 = 0  (arm pointing along world +x)
Fix joint5 = 0, joint6 = 0  (wrist roll/yaw neutral)
Solve joints 2, 3, 4 to place the tool0 frame at
    (target_reach_m,  0,  Z_TARGET_M)
using pyroki (Levenberg-Marquardt, JAX-based) with a high-weight rest cost on
the fixed joints and a position-only pose cost on tool0.

Recording strategy
------------------
Each waypoint produces two LeRobot episodes:
  episode 2*i   — transit frames (interpolated motion to waypoint i)
  episode 2*i+1 — dwell frames (held at waypoint i)

Transit episodes carry task="transit_to_reach_Xm".
Dwell episodes carry task="dwell_reach_Xm".

This split lets analyse.py produce two separate charts:
  nominal  — steady-state torque from dwell episodes
  peak     — peak torque during transit episodes

A waypoints.json sidecar records the episode indices and IK solutions.

Usage
-----
    # Preview planned waypoints without touching hardware:
    uv run payload_measurement/program_sweep.py --dry-run

    # Baseline (no payload):
    uv run payload_measurement/program_sweep.py --mass-g 0

    # With payload:
    uv run payload_measurement/program_sweep.py --mass-g 200

    # Use impedance mode instead (for comparison):
    uv run payload_measurement/program_sweep.py --mass-g 0 --control-mode impedance

Output
------
    outputs/payload_measurement/replays/mass_<N>g/
    outputs/payload_measurement/replays/mass_<N>g/waypoints.json

Controls
--------
    RIGHT ARROW  — payload attached / confirm, begin sweep
    Ctrl-C       — abort (episodes already saved are kept)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import rerun as rr
import rerun.blueprint as rrb
import yourdfpy

from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOLLOWER_PORT       = "/dev/tty.usbmodem00000000050C1"
CURRENT_SENSOR_PORT = "/dev/tty.usbserial-BG0038JI"

_REPO_ROOT  = Path(__file__).parent.parent
URDF_PATH   = str(_REPO_ROOT / "urdf" / "follower" / "TRLC-DK1-Follower.urdf")
REPLAYS_DIR = Path("outputs/payload_measurement/replays")

# Reach sweep: horizontal distance from the joint-1 axis to tool0, in metres.
TARGET_REACHES_M = [0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.5]

# Target end-effector height for all waypoints (m).
# ~shoulder height (joint2 is at z≈0.10 m) keeps the arm near-horizontal,
# which is the worst-case pose for gravitational joint load.
Z_TARGET_M = 0.20

# Timing
DWELL_S   = 3.0   # seconds to hold and record at each waypoint
TRANSIT_S = 8.0    # seconds to interpolate between consecutive waypoints
FPS       = 60

# Safe home position (joints 1–6, radians) — all motors at calibrated zero.
# Used at start and end of the sweep.
HOME_Q = np.zeros(6)

# ---------------------------------------------------------------------------
# IK / FK planner
# ---------------------------------------------------------------------------

class SweepPlanner:
    """
    Computes joint-space waypoints for a horizontal reach sweep using pyroki IK.

    Joints 1, 5, 6 are held at zero (arm in xz-plane, wrist neutral) via a
    high-weight rest cost. Joints 2, 3, 4 are solved freely.

    Orientation constraint: the flange (link6-7) is kept level throughout the
    sweep — local-X pointing world +X, local-Z pointing world +Z (identity
    rotation). This requires j4 to compensate the combined pitch of j2+j3,
    ensuring the payload is always held horizontally regardless of reach.

    Joint limits are enforced automatically from the URDF via limit_constraint.
    Warm-starts are chained across waypoints for smooth joint-space progression.
    """

    def __init__(self, urdf_path: str) -> None:
        self._urdf = yourdfpy.URDF.load(urdf_path)
        robot = pk.Robot.from_urdf(self._urdf)
        link_names = list(robot.links.names)
        # MuJoCo merges fixed-joint children into their parent body, so the
        # original MuJoCo-based IK targeted link6-7 (the last moving link).
        # Use the same link here to keep reach values consistent.
        for name in ("link6-7", "tool0"):
            if name in link_names:
                self._tool0_idx = link_names.index(name)
                break
        else:
            raise RuntimeError("Cannot find 'link6-7' or 'tool0' link in URDF.")
        self._n = robot.joints.num_actuated_joints
        self._robot = robot  # kept for FK

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return world-frame xyz of tool0 for a 6-DOF arm config."""
        q_full = np.zeros(self._n)
        q_full[:6] = q
        poses = self._robot.forward_kinematics(jnp.array(q_full))
        return np.array(poses[self._tool0_idx, 4:])  # wxyz+xyz → xyz

    def solve_ik(
        self,
        target_reach: float,
        z_target: float,
        q_init: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Solve IK for joints 2/3/4; fix joints 1/5/6 at zero.

        q_init is used as the warm-start for the solver.
        Returns (q_solution_6dof, position_error_m).
        """
        q_full_init = np.zeros(self._n)
        q_full_init[:6] = q_init

        # Rebuild robot with the warm-start config as the solver's initial value.
        robot = pk.Robot.from_urdf(self._urdf, default_joint_cfg=jnp.array(q_full_init))

        # Per-joint weight for rest cost: high on fixed joints, zero on free joints.
        # joint_4 is intentionally free (weight=0): the orientation constraint
        # drives it to the value that keeps the flange level.
        w = np.zeros(self._n)
        w[0]  = 100.0   # joint_1 — base yaw, fixed at 0
        w[4]  = 100.0   # joint_5 — wrist pitch, fixed at 0
        w[5]  = 100.0   # joint_6 — wrist roll, fixed at 0
        w[6:] = 100.0   # gripper + any remaining joints

        joint_var = robot.joint_var_cls(0)
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(jnp.array([1.0, 0.0, 0.0, 0.0])),  # identity = flange level
                    jnp.array([target_reach, 0.0, z_target]),
                ),
                jnp.array(self._tool0_idx, dtype=jnp.int32),
                pos_weight=50.0,
                ori_weight=10.0,  # keep flange horizontal throughout sweep
            ),
            pk.costs.limit_constraint(robot, joint_var),
            pk.costs.rest_cost(joint_var, jnp.zeros(self._n), weight=jnp.array(w)),
        ]
        sol = (
            jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
            .analyze()
            .solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            )
        )

        q_sol = np.array(sol[joint_var])[:6].copy()
        pos = self.fk(q_sol)
        error = float(np.sqrt((pos[0] - target_reach) ** 2 + (pos[2] - z_target) ** 2))
        return q_sol, error

    def plan(
        self,
        reaches_m: list[float],
        z_target: float,
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """
        Solve IK for each target reach, chaining warm-starts for continuity.

        Returns list of (q_solution, actual_reach_m, tool0_xyz).
        Note: the first call triggers JAX JIT compilation (~5–10 s).
        """
        q_prev = np.array([0.0, 2.50, 0.80, -0.30, 0.0, 0.0])
        waypoints: list[tuple[np.ndarray, float, np.ndarray]] = []

        for r in reaches_m:
            q_sol, err = self.solve_ik(r, z_target, q_prev)
            pos = self.fk(q_sol)
            actual_reach = float(np.hypot(pos[0], pos[1]))
            waypoints.append((q_sol, actual_reach, pos.copy()))
            q_prev = q_sol.copy()

        return waypoints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_action(q: np.ndarray, gripper: float = 0.0) -> dict[str, float]:
    return {
        "joint_1.pos": float(q[0]),
        "joint_2.pos": float(q[1]),
        "joint_3.pos": float(q[2]),
        "joint_4.pos": float(q[3]),
        "joint_5.pos": float(q[4]),
        "joint_6.pos": float(q[5]),
        "gripper.pos": float(gripper),
    }


def _move_to(
    follower: DK1Follower,
    q_start: np.ndarray,
    q_target: np.ndarray,
    transit_s: float,
    fps: float,
    *,
    record: dict | None = None,
) -> None:
    """
    Cosine-eased joint-space interpolation from q_start to q_target.

    If `record` is provided, every frame is saved to the dataset and the
    episode is finalised at the end.  `record` must contain:
        dataset     — LeRobotDataset
        state_names — list[str]  (observation feature keys)
        action_names — list[str] (action feature keys)
        task        — str        (task label for this episode)
    """
    n = max(1, int(transit_s * fps))
    t0 = time.perf_counter()
    for step in range(n):
        alpha = (step + 1) / n
        alpha_s = 0.5 * (1.0 - np.cos(np.pi * alpha))   # cosine ease
        q = (1.0 - alpha_s) * q_start + alpha_s * q_target
        action_dict = _build_action(q)
        follower.send_action(action_dict)
        if record is not None:
            obs = follower.get_observation()
            state_vec  = np.array([obs[sn]  for sn in record["state_names"]],  dtype=np.float32)
            action_vec = np.array([action_dict[an] for an in record["action_names"]], dtype=np.float32)
            record["dataset"].add_frame({
                "observation.state": state_vec,
                "action": action_vec,
                "task": record["task"],
            })
            log_rerun_data(observation=obs, action=action_dict)
        sleep_s = t0 + (step + 1) / fps - time.perf_counter()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    if record is not None:
        record["dataset"].save_episode()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Programmatic payload sweep.")
    parser.add_argument(
        "--mass-g", type=float, default=0.0,
        help="Payload mass in grams (0 = baseline)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned waypoints and exit — no hardware required",
    )
    parser.add_argument(
        "--control-mode", default="pos_vel", choices=["pos_vel", "impedance"],
        help="Follower control mode (default: pos_vel)",
    )
    args = parser.parse_args()
    mass_g: float = args.mass_g

    # ------------------------------------------------------------------
    # Plan waypoints (offline — no hardware)
    # ------------------------------------------------------------------
    print("\n--- Sweep planner ---")
    planner = SweepPlanner(URDF_PATH)
    waypoints = planner.plan(TARGET_REACHES_M, Z_TARGET_M)

    header = f"{'Target':>9}  {'Actual':>9}  {'Height':>9}  {'j2':>8}  {'j3':>8}  {'j4':>8}  {'err':>7}"
    print(header)
    print("-" * len(header))
    any_bad = False
    for (q, actual_reach, pos), target in zip(waypoints, TARGET_REACHES_M):
        err_mm = abs(actual_reach - target) * 1000
        flag = "  *** IK err > 10 mm ***" if err_mm > 10.0 else ""
        if err_mm > 10.0:
            any_bad = True
        print(
            f"{target*1000:>8.0f}mm  {actual_reach*1000:>8.1f}mm  "
            f"{pos[2]*1000:>8.1f}mm  "
            f"{np.degrees(q[1]):>7.1f}°  {np.degrees(q[2]):>7.1f}°  "
            f"{np.degrees(q[3]):>7.1f}°  {err_mm:>6.1f}mm{flag}"
        )

    if any_bad:
        print("\nWARNING: one or more waypoints have IK error > 10 mm.")
        print("Adjust TARGET_REACHES_M or Z_TARGET_M and re-run --dry-run to verify.")

    if args.dry_run:
        print("\n(dry-run — hardware not connected)")
        return

    # ------------------------------------------------------------------
    # Dataset setup
    # ------------------------------------------------------------------
    follower_config = DK1FollowerConfig(
        port=FOLLOWER_PORT,
        control_mode=args.control_mode,
        current_sensor_port=CURRENT_SENSOR_PORT,
    )
    follower = DK1Follower(follower_config)

    obs_features    = hw_to_dataset_features(follower.observation_features, "observation")
    action_features = hw_to_dataset_features(follower.action_features, "action")

    out_dir = REPLAYS_DIR / f"mass_{mass_g:.0f}g"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    dataset = LeRobotDataset.create(
        repo_id=f"trlc/payload_mass_{mass_g:.0f}g",
        fps=FPS,
        features={**obs_features, **action_features},
        robot_type=follower.name,
        root=out_dir,
    )

    # Derive flat feature name lists in the same order as hw_to_dataset_features
    state_names  = [k for k, v in follower.observation_features.items() if v is float]
    action_names = list(follower.action_features.keys())

    _, events = init_keyboard_listener()
    init_rerun(session_name=f"sweep_mass_{mass_g:.0f}g")

    _joints = [f"joint_{i}" for i in range(1, 7)]
    rr.send_blueprint(rrb.Blueprint(
        rrb.Grid(
            rrb.TimeSeriesView(
                name="Torques (Nm)",
                contents=[f"observation.{j}.torque" for j in _joints]
                        + ["observation.gripper.torque"],
            ),
            rrb.TimeSeriesView(
                name="Temperatures (°C)",
                contents=[f"observation.{j}.temp_motor" for j in _joints]
                        + [f"observation.{j}.temp_mos" for j in _joints],
            ),
            rrb.TimeSeriesView(
                name="Supply Current (A)",
                contents=["observation.external_current_a"],
            ),
            rrb.TimeSeriesView(
                name="Positions (rad)",
                contents=[f"observation.{j}.pos" for j in _joints]
                        + ["observation.gripper.pos"],
            ),
        ),
        collapse_panels=False,
    ))

    follower.connect()
    waypoint_meta: list[dict] = []
    episode_counter = 0   # tracks dataset episode index as we save episodes

    try:
        # Move to home, then first waypoint
        obs0 = follower.get_observation()
        q_current = np.array([obs0[f"joint_{j+1}.pos"] for j in range(6)])

        print("\nMoving to home position ...")
        _move_to(follower, q_current, HOME_Q, TRANSIT_S, FPS)

        print("Moving to first waypoint ...")
        _move_to(follower, HOME_Q, waypoints[0][0], TRANSIT_S, FPS)

        # Wait for payload attachment confirmation
        events["exit_early"] = False
        label = f"{mass_g:.0f}g payload" if mass_g > 0 else "no payload (baseline)"
        print(f"\nAttach {label}.")
        print("Press RIGHT ARROW to begin sweep.")
        q_hold = waypoints[0][0]
        while not events["exit_early"]:
            follower.send_action(_build_action(q_hold))
            log_rerun_data(observation=follower.get_observation())
            time.sleep(1.0 / FPS)

        # ------------------------------------------------------------------
        # Sweep: for each waypoint, record a transit episode then a dwell
        # episode.  episode_counter tracks the dataset index.
        # ------------------------------------------------------------------
        q_prev = waypoints[0][0].copy()

        for wp_idx, (q_target, actual_reach, pos) in enumerate(waypoints):
            print(f"\n[{wp_idx + 1}/{len(waypoints)}]  "
                  f"reach {actual_reach * 1000:.0f} mm  "
                  f"(target {TARGET_REACHES_M[wp_idx] * 1000:.0f} mm)")

            # --- Transit episode ---
            transit_task = f"transit_to_reach_{actual_reach:.3f}m"
            print(f"  Transiting ({TRANSIT_S:.0f} s) → episode {episode_counter} ...")
            transit_episode_idx = episode_counter
            _move_to(
                follower, q_prev, q_target, TRANSIT_S, FPS,
                record={
                    "dataset":      dataset,
                    "state_names":  state_names,
                    "action_names": action_names,
                    "task":         transit_task,
                },
            )
            episode_counter += 1

            # --- Dwell episode ---
            dwell_task   = f"dwell_reach_{actual_reach:.3f}m"
            action_dict  = _build_action(q_target)
            action_vec   = np.array([action_dict[an] for an in action_names], dtype=np.float32)

            print(f"  Recording dwell ({DWELL_S:.0f} s) → episode {episode_counter} ...")
            dwell_episode_idx = episode_counter

            t0 = time.perf_counter()
            n_frames = int(DWELL_S * FPS)
            q_errors: list[np.ndarray] = []

            for step in range(n_frames):
                follower.send_action(action_dict)
                obs = follower.get_observation()
                state_vec = np.array([obs[sn] for sn in state_names], dtype=np.float32)
                dataset.add_frame({
                    "observation.state": state_vec,
                    "action": action_vec,
                    "task": dwell_task,
                })
                log_rerun_data(observation=obs, action=action_dict)

                # Accumulate position error for diagnostics (not stored)
                q_actual = np.array([obs[f"joint_{j+1}.pos"] for j in range(6)])
                q_errors.append(np.abs(q_actual - q_target))

                sleep_s = t0 + (step + 1) / FPS - time.perf_counter()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

            dataset.save_episode()
            episode_counter += 1

            # Print position error diagnostics
            mean_err = np.mean(q_errors, axis=0)
            max_err  = np.max(q_errors, axis=0)
            print(f"  Position error (mean/max rad): "
                  + "  ".join(
                      f"j{i+1}: {mean_err[i]:.3f}/{max_err[i]:.3f}"
                      for i in range(6)
                  ))
            fk_mean = planner.fk(q_target + mean_err)
            fk_target = planner.fk(q_target)
            print(f"  FK reach error (mean): "
                  f"{np.linalg.norm(fk_mean - fk_target) * 1000:.1f} mm")

            waypoint_meta.append({
                "episode_transit_idx": transit_episode_idx,
                "episode_dwell_idx":   dwell_episode_idx,
                "target_reach_m":      TARGET_REACHES_M[wp_idx],
                "actual_reach_m":      actual_reach,
                "tool0_xyz_m":         pos.tolist(),
                "q_rad":               q_target.tolist(),
                "q_deg":               np.degrees(q_target).tolist(),
            })
            q_prev = q_target.copy()

        # Return to home
        print("\nMoving back to home ...")
        _move_to(follower, q_prev, HOME_Q, TRANSIT_S, FPS)

        dataset.finalize()

        # Write reach metadata alongside the dataset
        (out_dir / "waypoints.json").write_text(
            json.dumps(
                {
                    "mass_g":        mass_g,
                    "z_target_m":    Z_TARGET_M,
                    "control_mode":  args.control_mode,
                    "waypoints":     waypoint_meta,
                },
                indent=2,
            )
        )

        total_episodes = episode_counter
        total_frames   = (len(waypoints) * int(DWELL_S * FPS)
                          + len(waypoints) * int(TRANSIT_S * FPS))
        print(f"\nSweep complete → {out_dir}")
        print(f"{total_episodes} episodes ({len(waypoints)} transit + "
              f"{len(waypoints)} dwell), ~{total_frames} frames.")
        print("Run analyse.py to update the payload charts.")

    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
