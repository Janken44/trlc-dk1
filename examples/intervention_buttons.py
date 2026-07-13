"""
Button-gated intervention teleop for the TRLC-DK1 — DAgger-style intervention test.

Scenario: a policy runs, detects it needs help and pauses; the operator steps in
with the leader arm as an intervention controller. The two handle buttons
(bus node ID 8, see leader_v1_MVP/sketch_handle-buttonnode) gate HOW leader
motion is applied:

  no button   FREEZE   — follower holds the last commanded pose. The leader can
                         be moved/repositioned freely without the follower moving.
  button 1    DIRECT   — joint-space teleop with OFFSET-DECAY engagement: on
                         press, the leader/follower joint mismatch is latched as
                         an offset and subtracted from the passthrough, so the
                         follower does NOT jump — you have control instantly.
                         The offset then decays to zero over OFFSET_DECAY_S,
                         smoothly melting the follower onto the leader's true
                         pose while you work (standard bilateral-teleop trick).
  button 2    REL-EE   — relative Cartesian control: on press, the leader's EE
                         pose and the follower's current EE setpoint are latched
                         as references. While held, the leader's EE *change*
                         (relative to its latch) is applied to the follower's
                         latched EE setpoint and tracked via damped-LS IK.
                         Combined with TRANSLATION_SCALE < 1 this gives precise,
                         clutched manipulation from any leader posture.
  both        HOME     — the follower ramps (rate-limited) to the pose it had at
                         script start; use to end a teleop session cleanly.
                         Release to stop mid-way (freeze).

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

# Good params for precision tasks
# TRANSLATION_SCALE = .2
# ROTATION_SCALE = .6

# Good params for intervention
TRANSLATION_SCALE = 1
ROTATION_SCALE = 1

# Per-IK-call trust region (rad/joint). At 60 Hz this bounds joint speed to
# ~max_step*60 rad/s and keeps the DLS solver on the local branch (no jumps).
IK_MAX_STEP_RAD = 0.1

# DIRECT-mode engagement: the initial leader/follower mismatch is latched and
# decays to zero over this time — instant control, no jump, no waiting.
OFFSET_DECAY_S = 0.6

# HOME mode (both buttons): follower ramps to its script-start pose at this
# bounded joint speed.
HOME_SPEED_RAD_S = 0.8
HOME_DONE_TOL_RAD = 0.02

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

# HOME target: the pose the follower had when the script started.
q_home = q_cmd.copy()
gripper_home = gripper_cmd

# Relative-EE latches (set on button-2 press edge).
leader_ref_pose: np.ndarray | None = None    # leader EE 4x4 at latch
follower_ref_pose: np.ndarray | None = None  # follower EE *setpoint* 4x4 at latch

# DIRECT-mode offset-decay state (latched on press edge).
direct_offset = np.zeros(len(JOINT_NAMES))
direct_t0 = 0.0

home_announced = False

prev_mode = "freeze"
print("Intervention teleop: btn1 = direct joints (offset-decay engage), "
      "btn2 = relative EE, both = home to start pose, none = freeze. Ctrl-C to stop.")

try:
    while True:
        t0 = time.perf_counter()

        leader_action = leader.get_action()
        buttons = leader.read_handle_buttons()   # -1 on node failure -> freeze
        btn1 = buttons > 0 and bool(buttons & 0b01)
        btn2 = buttons > 0 and bool(buttons & 0b10)
        if btn1 and btn2:
            mode = "home"
        elif btn1:
            mode = "direct"
        elif btn2:
            mode = "rel_ee"
        else:
            mode = "freeze"

        if mode == "direct":
            q_leader = joints_from_action(leader_action)

            if prev_mode != "direct":
                # Press edge: latch the mismatch. Commanding q_leader - offset keeps
                # the follower exactly where it is — control is instant, no jump.
                direct_offset = q_leader - q_cmd
                direct_t0 = time.perf_counter()

            # Decay the latched offset to zero: the follower tracks all leader
            # *motion* 1:1 immediately, while the residual melts away smoothly.
            lam = max(0.0, 1.0 - (time.perf_counter() - direct_t0) / OFFSET_DECAY_S)
            q_cmd = q_leader - direct_offset * lam
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

        elif mode == "home":
            # Ramp the follower to the script-start pose at bounded joint speed.
            if prev_mode != "home":
                home_announced = False
                print("home: moving follower to start pose (release to stop)...")
            err = q_home - q_cmd
            if np.max(np.abs(err)) <= HOME_DONE_TOL_RAD:
                q_cmd = q_home.copy()
                gripper_cmd = gripper_home
                if not home_announced:
                    home_announced = True
                    print("home: start pose reached — safe to quit (Ctrl-C).")
            else:
                step = HOME_SPEED_RAD_S / FREQ_HZ
                q_cmd = q_cmd + np.clip(err, -step, step)

        # freeze: q_cmd/gripper_cmd unchanged — keep commanding the held pose.

        follower.send_action(action_from_joints(q_cmd, gripper_cmd))
        prev_mode = mode

        dt = time.perf_counter() - t0
        time.sleep(max(0.0, 1.0 / FREQ_HZ - dt))
except KeyboardInterrupt:
    print("\nStopping intervention teleop...")
    leader.disconnect()
    follower.disconnect()
