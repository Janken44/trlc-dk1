"""
Button-gated intervention teleop for the TRLC-DK1 — DAgger-style intervention test.

Scenario: a policy runs, detects it needs help and pauses; the operator steps in
with the leader arm as an intervention controller. The two handle buttons
(bus node ID 8, see leader_v1_MVP/sketch_handle-buttonnode) gate HOW leader
motion is applied:

  no button   FREEZE   — follower holds the last commanded pose. The leader can
                         be moved/repositioned freely without the follower moving.
  button 1    DIRECT   — leader joint targets are sent to the follower verbatim
                         (same path as examples/teleop_ee.py without IK).
  button 2    REL-EE   — relative Cartesian control: on press, the leader's EE
                         pose and the follower's current EE setpoint are latched
                         as references. While held, the leader's EE *change*
                         (relative to its latch) is applied to the follower's
                         latched EE setpoint and tracked via damped-LS IK.
                         Combined with TRANSLATION_SCALE < 1 this gives precise,
                         clutched manipulation from any leader posture.

  button 1 takes priority if both are held.

FREEZE is the safe default: it is also entered if the button node stops
responding (read failure -> -1 -> no bits set).

Run:  python examples/intervention_buttons.py
"""

import time

import numpy as np

from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig, JOINT_NAMES
from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig
from trlc_dk1_control.kinematics import DK1Kinematics

FREQ_HZ = 60

# Relative-EE tuning: leader motion is scaled by these before being applied to the
# follower setpoint. <1.0 = finer, more precise follower motion ("precision clutch").
TRANSLATION_SCALE = 1.0
ROTATION_SCALE = 1.0

# Per-IK-call trust region (rad/joint). At 60 Hz this bounds joint speed to
# ~max_step*60 rad/s and keeps the DLS solver on the local branch (no jumps).
IK_MAX_STEP_RAD = 0.1

follower_config = DK1FollowerConfig(
    port="/dev/tty.usbmodem00000000050C1",
    control_mode="impedance",
)
leader_config = DK1LeaderConfig(
    port="/dev/tty.usbmodemE072A1F88B781",
)

leader = DK1Leader(leader_config)
leader.connect()
follower = DK1Follower(follower_config)
follower.connect()

kin = DK1Kinematics(max_step_rad=IK_MAX_STEP_RAD)


def joints_from_action(action: dict) -> np.ndarray:
    return np.array([action[f"{n}.pos"] for n in JOINT_NAMES], dtype=float)


def action_from_joints(q: np.ndarray, gripper: float) -> dict:
    action = {f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, q)}
    action["gripper.pos"] = float(gripper)
    return action


def scaled_rotation_delta(R_now: np.ndarray, R_ref: np.ndarray, scale: float) -> np.ndarray:
    """World-frame rotation delta R_now @ R_ref.T, optionally scaled via its rotvec."""
    R_delta = R_now @ R_ref.T
    if scale == 1.0:
        return R_delta
    # Scale the axis-angle magnitude (uses MuJoCo-free helper from kinematics).
    from trlc_dk1_control.kinematics import _rotation_matrix_to_rotvec

    rotvec = _rotation_matrix_to_rotvec(R_delta) * scale
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.eye(3)
    axis = rotvec / angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# --- Initial state: freeze at the follower's current pose -----------------------
obs = follower.get_observation()
q_cmd = joints_from_action(obs)          # last commanded arm joints (rad)
gripper_cmd = float(obs["gripper.pos"])  # last commanded gripper (normalized)

# Relative-EE latches (set on button-2 press edge).
leader_ref_pose: np.ndarray | None = None    # leader EE 4x4 at latch
follower_ref_pose: np.ndarray | None = None  # follower EE *setpoint* 4x4 at latch

prev_mode = "freeze"
print("Intervention teleop: btn1 = direct joint teleop, btn2 = relative EE, "
      "none = freeze. Ctrl-C to stop.")

try:
    while True:
        t0 = time.perf_counter()

        leader_action = leader.get_action()
        buttons = leader.read_handle_buttons()   # -1 on node failure -> freeze
        btn1 = buttons > 0 and bool(buttons & 0b01)
        btn2 = buttons > 0 and bool(buttons & 0b10)
        mode = "direct" if btn1 else ("rel_ee" if btn2 else "freeze")

        if mode == "direct":
            # Leader joints straight through (teleop_ee.py behavior, no IK).
            q_cmd = joints_from_action(leader_action)
            gripper_cmd = float(leader_action["gripper.pos"])

        elif mode == "rel_ee":
            q_leader = joints_from_action(leader_action)
            leader_pose = kin.forward_kinematics(q_leader)

            if prev_mode != "rel_ee":
                # Press edge: latch leader pose and the follower's current *setpoint*
                # (q_cmd, not the measured joints — impedance mode sags under load,
                # and latching the setpoint avoids a jump on engage).
                leader_ref_pose = leader_pose.copy()
                follower_ref_pose = kin.forward_kinematics(q_cmd)

            # Apply the leader's EE change (scaled) to the latched follower setpoint.
            target = np.eye(4)
            target[:3, 3] = follower_ref_pose[:3, 3] + TRANSLATION_SCALE * (
                leader_pose[:3, 3] - leader_ref_pose[:3, 3]
            )
            target[:3, :3] = (
                scaled_rotation_delta(leader_pose[:3, :3], leader_ref_pose[:3, :3], ROTATION_SCALE)
                @ follower_ref_pose[:3, :3]
            )

            # IK seeded from the previous command: trust region acts as a rate limit.
            q_cmd = kin.inverse_kinematics(q_cmd, target)
            gripper_cmd = float(leader_action["gripper.pos"])

        # freeze: q_cmd/gripper_cmd unchanged — keep commanding the held pose.

        follower.send_action(action_from_joints(q_cmd, gripper_cmd))
        prev_mode = mode

        dt = time.perf_counter() - t0
        time.sleep(max(0.0, 1.0 / FREQ_HZ - dt))
except KeyboardInterrupt:
    print("\nStopping intervention teleop...")
    leader.disconnect()
    follower.disconnect()
