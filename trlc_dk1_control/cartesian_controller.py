"""
CartesianController — Cartesian-space teleop layer on top of DK1Robot.

Drives the joint impedance controller from a Cartesian setpoint. Two ways to
move the setpoint:

  • Discrete shift   : shift_target_pos / shift_target_rot  (instant target jump)
  • Latched velocity : set_velocity(lin, rot, ...)          (smooth motion while
                                                              keys are held)

Background loop (control_hz):

    1. Decay/refresh the velocity command (TTL-based — held-key teleop).
    2. First-order LP filter on velocity for smooth start/stop.
    3. Integrate target_pose by the filtered velocity.
    4. Compute pose error err = target − FK(q_target_internal).
       (q_target_internal is the *commanded* config — decoupled from the measured
       state so external disturbances on the arm don't move the setpoint.)
    5. Damped LS IK on the twist vel_ff + Kp*err → joint velocity.
    6. Integrate q_target_internal += q_dot * dt, clip, send to robot.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .kinematics import DK1Kinematics
from .robot import DK1Robot

logger = logging.getLogger(__name__)


def _rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → axis-angle (axis * angle, magnitude = angle in rad)."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    if angle < 1e-8:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # 180° rotation — recover axis from the largest diagonal of (R+I)/2
        i = int(np.argmax(np.diag(R)))
        axis = np.zeros(3)
        axis[i] = np.sqrt(max(0.0, (R[i, i] + 1.0) / 2.0))
        if axis[i] > 1e-8:
            j, k = (i + 1) % 3, (i + 2) % 3
            axis[j] = (R[j, i] + R[i, j]) / (4.0 * axis[i])
            axis[k] = (R[k, i] + R[i, k]) / (4.0 * axis[i])
        return axis * angle
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def _axisangle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """axis-angle (3-vector with magnitude=angle) → 3x3 rotation matrix."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.eye(3)
    k = aa / angle
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


@dataclass
class CartesianControllerConfig:
    """Parameters for the Cartesian-space controller."""

    control_hz: float = 100.0

    # IK damping (m) — bigger = more conservative near singularities (more error)
    damping: float = 0.08

    # Closed-loop Cartesian P-gains (1/s). Set the bandwidth at which the
    # virtual EE chases the target pose. Roughly: lag ≈ vel / gain.
    pos_gain: float = 8.0
    rot_gain: float = 10.0

    # Velocity-command smoothing time constant (s). Higher = silkier but laggier;
    # lower = snappier but more abrupt on key press/release.
    vel_smooth_tau: float = 0.06

    # Default TTL for set_velocity() commands. Held-key teleop relies on this:
    # terminal auto-repeat refreshes the command every ~30 ms, well inside TTL.
    vel_ttl: float = 0.15

    # Per-step joint motion cap (rad). Hard safety speed limit.
    max_dq_per_step: float = 0.05

    # Workspace bounds for the target position (m, world frame).
    pos_min: np.ndarray = field(
        default_factory=lambda: np.array([-0.5, -0.5, 0.05])
    )
    pos_max: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 0.5, 0.8])
    )

    # Anti-windup: clamp target back if it drifts further than this from virt-EE.
    max_pos_lag: float = 0.10   # m
    max_rot_lag: float = 0.6    # rad

    # IK control point along the tool approach (X) axis from link6-7 (m).
    # 0.158 = tool0 tip (default). 0.0 = at link6-7 (wrist joint).
    # Smaller values move the rotation pivot toward the wrist.
    # Note: pos_min/pos_max bound the control point, not the gripper tip.
    control_point_offset: float = 0.158


class CartesianController:
    """
    Cartesian impedance teleop layer on top of an active DK1Robot.

    Example::

        robot = DK1Robot(DK1_DEFAULT_CONFIG("/dev/ttyACM0"))
        robot.connect()

        cart = CartesianController(robot)
        cart.start()

        # Smooth held-key motion (call repeatedly, e.g. on each keypress):
        cart.set_velocity(lin=[0.1, 0, 0], rot=[0, 0, 0])

        # Or discrete jumps:
        cart.shift_target_pos([0.01, 0, 0])

        cart.stop()
        robot.disconnect()
    """

    def __init__(
        self,
        robot: DK1Robot,
        config: CartesianControllerConfig | None = None,
    ) -> None:
        self.robot = robot
        self.cfg = config or CartesianControllerConfig()

        model_path = robot._config.mjcf_path or robot._config.urdf_path
        if not model_path:
            raise ValueError(
                "DK1Robot config has no mjcf_path/urdf_path — kinematics unavailable."
            )
        self.kin = DK1Kinematics(
            model_path,
            control_point_offset=self.cfg.control_point_offset,
        )

        self._lock = threading.Lock()
        self._target_pos = np.zeros(3)
        self._target_rot = np.eye(3)
        # Internal commanded joint config — decoupled from q_measured so a hand
        # pushing the arm doesn't drag the setpoint along with it.
        self._q_target = np.zeros(6)

        # Latched velocity command (set_velocity)
        self._cmd_v = np.zeros(3)    # m/s
        self._cmd_w = np.zeros(3)    # rad/s
        self._cmd_lin_frame = "world"
        self._cmd_rot_frame = "tool"
        self._cmd_expire = 0.0       # monotonic time when command auto-zeros

        # LP-filtered velocity (loop-local; held under _lock for thread-safe reads)
        self._v_smooth = np.zeros(3)
        self._w_smooth = np.zeros(3)

        self._running = False
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> None:
        q = self.robot.get_joint_state()["pos"]
        pos, rot = self.kin.fk(q)
        with self._lock:
            self._target_pos = pos.copy()
            self._target_rot = rot.copy()
            self._q_target = q.copy()
            self._v_smooth = np.zeros(3)
            self._w_smooth = np.zeros(3)

        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="dk1-cart"
        )
        self._thread.start()
        logger.info(
            "CartesianController started — initial tool pos=%s",
            np.round(pos, 4),
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -------------------------------------------------------------------------
    # Target interface (thread-safe)
    # -------------------------------------------------------------------------

    def set_velocity(
        self,
        lin: np.ndarray | list[float] = (0.0, 0.0, 0.0),
        rot: np.ndarray | list[float] = (0.0, 0.0, 0.0),
        lin_frame: str = "world",
        rot_frame: str = "tool",
        ttl: float | None = None,
    ) -> None:
        """
        Latch a Cartesian twist command.

        The command stays active for `ttl` seconds (default `cfg.vel_ttl`) since
        the last call to `set_velocity`. Calling repeatedly (e.g. on every
        terminal key-repeat) keeps it alive; stop calling and motion decays
        smoothly to zero via the LP filter.

        Args:
            lin:        Linear velocity 3-vector (m/s).
            rot:        Angular velocity 3-vector (rad/s).
            lin_frame:  "world" or "tool" — frame the linear vel is expressed in.
            rot_frame:  "world" or "tool" — frame the angular vel is expressed in.
            ttl:        Override the default TTL (seconds).
        """
        if lin_frame not in ("world", "tool"):
            raise ValueError(f"lin_frame must be 'world' or 'tool', got {lin_frame!r}")
        if rot_frame not in ("world", "tool"):
            raise ValueError(f"rot_frame must be 'world' or 'tool', got {rot_frame!r}")
        v = np.asarray(lin, dtype=float)
        w = np.asarray(rot, dtype=float)
        ttl_use = self.cfg.vel_ttl if ttl is None else ttl
        with self._lock:
            self._cmd_v = v
            self._cmd_w = w
            self._cmd_lin_frame = lin_frame
            self._cmd_rot_frame = rot_frame
            self._cmd_expire = time.monotonic() + ttl_use

    def stop_velocity(self) -> None:
        """Immediately zero the latched velocity command (LP filter still decays)."""
        with self._lock:
            self._cmd_v = np.zeros(3)
            self._cmd_w = np.zeros(3)
            self._cmd_expire = 0.0

    def shift_target_pos(self, dxyz: np.ndarray | list[float]) -> None:
        """Step the target tool position by `dxyz` (world frame, m)."""
        d = np.asarray(dxyz, dtype=float)
        with self._lock:
            self._target_pos = np.clip(
                self._target_pos + d, self.cfg.pos_min, self.cfg.pos_max
            )

    def shift_target_rot(
        self,
        daxis_angle: np.ndarray | list[float],
        frame: str = "tool",
    ) -> None:
        """
        Step the target tool orientation by an axis-angle delta.

        `frame="tool"` rotates about the EE's own axes (intuitive for teleop);
        `frame="world"` rotates about world axes.
        """
        aa = np.asarray(daxis_angle, dtype=float)
        R_delta = _axisangle_to_rotmat(aa)
        with self._lock:
            if frame == "tool":
                self._target_rot = self._target_rot @ R_delta
            elif frame == "world":
                self._target_rot = R_delta @ self._target_rot
            else:
                raise ValueError(f"frame must be 'world' or 'tool', got {frame!r}")

    def set_target_pose(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
    ) -> None:
        """Set the target tool pose directly (world frame).

        Args:
            pos: (3,) position in metres.
            rot: (3, 3) rotation matrix.
        """
        p = np.clip(np.asarray(pos, dtype=float), self.cfg.pos_min, self.cfg.pos_max)
        r = np.asarray(rot, dtype=float)
        with self._lock:
            self._target_pos = p
            self._target_rot = r

    def set_gripper(self, normalized_pos: float) -> None:
        self.robot.command_gripper(float(normalized_pos))

    def reset_target_to_current(self) -> None:
        """Snap target pose AND internal q_target back to the measured state."""
        q = self.robot.get_joint_state()["pos"]
        pos, rot = self.kin.fk(q)
        with self._lock:
            self._target_pos = pos.copy()
            self._target_rot = rot.copy()
            self._q_target = q.copy()
            self._v_smooth = np.zeros(3)
            self._w_smooth = np.zeros(3)
            self._cmd_expire = 0.0

    def get_target(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._target_pos.copy(), self._target_rot.copy()

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current measured tool pose (pos, rot)."""
        q = self.robot.get_joint_state()["pos"]
        return self.kin.fk(q)

    # -------------------------------------------------------------------------
    # Control loop
    # -------------------------------------------------------------------------

    def _loop(self) -> None:
        cfg = self.cfg
        period = 1.0 / cfg.control_hz
        I6 = np.eye(6)
        joint_lims = self.robot._config.joint_pos_limits
        # LP filter alpha = dt/tau (clamped to <= 1)
        alpha = min(1.0, period / max(cfg.vel_smooth_tau, period))

        loop_count = 0
        last_log = time.monotonic()
        pos_lag = 0.0
        rot_lag = 0.0

        while self._running:
            t_start = time.monotonic()

            with self._lock:
                target_pos = self._target_pos.copy()
                target_rot = self._target_rot.copy()
                q_target = self._q_target.copy()
                v_smooth = self._v_smooth.copy()
                w_smooth = self._w_smooth.copy()
                # Resolve latched velocity command (expired → zero)
                if t_start < self._cmd_expire:
                    cmd_v = self._cmd_v.copy()
                    cmd_w = self._cmd_w.copy()
                    lin_frame = self._cmd_lin_frame
                    rot_frame = self._cmd_rot_frame
                else:
                    cmd_v = np.zeros(3)
                    cmd_w = np.zeros(3)
                    lin_frame = "world"
                    rot_frame = "world"

            # ---- Express commanded velocity in world frame ----------------
            v_world = (target_rot @ cmd_v) if lin_frame == "tool" else cmd_v
            w_world = (target_rot @ cmd_w) if rot_frame == "tool" else cmd_w

            # ---- LP filter ------------------------------------------------
            v_smooth = (1.0 - alpha) * v_smooth + alpha * v_world
            w_smooth = (1.0 - alpha) * w_smooth + alpha * w_world

            # ---- Integrate target pose ------------------------------------
            target_pos = np.clip(
                target_pos + v_smooth * period, cfg.pos_min, cfg.pos_max
            )
            rot_step = w_smooth * period
            if np.linalg.norm(rot_step) > 1e-9:
                target_rot = _axisangle_to_rotmat(rot_step) @ target_rot

            # ---- Closed-loop error against the *internal* virtual EE ------
            J, virt_pos, virt_rot = self.kin.jacobian(q_target)
            e_pos = target_pos - virt_pos
            R_err = target_rot @ virt_rot.T
            e_rot = _rotmat_to_axisangle(R_err)

            pos_lag = float(np.linalg.norm(e_pos))
            if pos_lag > cfg.max_pos_lag:
                e_pos = e_pos * (cfg.max_pos_lag / pos_lag)
                target_pos = virt_pos + e_pos

            rot_lag = float(np.linalg.norm(e_rot))
            if rot_lag > cfg.max_rot_lag:
                e_rot = e_rot * (cfg.max_rot_lag / rot_lag)
                target_rot = _axisangle_to_rotmat(e_rot) @ virt_rot

            # ---- Twist = velocity feedforward + P correction --------------
            v_des = v_smooth + cfg.pos_gain * e_pos
            w_des = w_smooth + cfg.rot_gain * e_rot
            twist = np.concatenate([v_des, w_des])

            # Damped LS: q_dot = J^T (J J^T + λ² I)^-1 twist  (rad/s)
            JJt = J @ J.T
            try:
                rhs = np.linalg.solve(JJt + (cfg.damping ** 2) * I6, twist)
            except np.linalg.LinAlgError:
                rhs = np.linalg.pinv(JJt + (cfg.damping ** 2) * I6) @ twist
            q_dot = J.T @ rhs

            # Integrate and clip
            dq = np.clip(
                q_dot * period, -cfg.max_dq_per_step, cfg.max_dq_per_step
            )
            q_target = np.clip(
                q_target + dq, joint_lims[:, 0], joint_lims[:, 1]
            )

            # ---- Publish state back under lock ----------------------------
            with self._lock:
                self._target_pos = target_pos
                self._target_rot = target_rot
                self._q_target = q_target
                self._v_smooth = v_smooth
                self._w_smooth = w_smooth

            self.robot.command_joint_pos(q_target)

            # ---- Maintain period ------------------------------------------
            elapsed = time.monotonic() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            loop_count += 1
            now = time.monotonic()
            if now - last_log >= 5.0:
                hz = loop_count / (now - last_log)
                print(
                    f"[cart]   {hz:6.1f} Hz  (target {cfg.control_hz:.0f} Hz)  "
                    f"loop={elapsed*1e3:.2f} ms  pos_err={pos_lag*1e3:5.1f} mm  "
                    f"rot_err={rot_lag*180/np.pi:4.1f}°"
                )
                loop_count = 0
                last_log = now
