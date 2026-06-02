"""
Meta Quest → DK1 Cartesian end-effector teleoperation.

Uses Quest controllers via WebXR (Vuer) to drive the DK1 end-effector in
Cartesian space. Hold the right trigger to track controller pose; release to
hold position.

ADB setup (one-time, Quest connected via USB):
    # Enable developer mode on Quest (Meta Horizon app → device → Developer Mode)
    adb reverse tcp:8012 tcp:8012

    # Open in Quest Meta Browser — use explicit http://, not https://:
    http://localhost:8012

    # Tap "Enter VR" or "Pass Through", then pick up both controllers

Run:
    uv run python examples/quest_teleop.py --port /dev/tty.usbmodemXXXX
    uv run python examples/quest_teleop.py --dry-run   # no robot, print targets only

Install Vuer if not present:
    uv pip install vuer

Controls:
    Right trigger (hold) — track controller pose → EE follows delta from latch point
    Right trigger (release) — EE holds last position
    Left trigger (rising edge) — toggle gripper open/closed
    Ctrl-C — quit

Frame convention:
    Quest world: Y-up, Z toward viewer (WebXR standard)
    Robot world: Z-up, X forward  (MuJoCo standard)
    Transform: quest_to_robot = grd_yup2grd_zup from TeleVision/teleop/constants_vuer.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

LOG_FILE = Path(__file__).parent.parent / "quest_teleop.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="w")],
)
log = logging.getLogger("quest")

# ---------------------------------------------------------------------------
# Vuer import — graceful error
# ---------------------------------------------------------------------------
try:
    from vuer import Vuer
    from vuer.schemas import DefaultScene, MotionControllers
except ImportError:
    print("Vuer not installed. Run: uv pip install vuer", file=sys.stderr)
    sys.exit(1)

from trlc_dk1_control import (
    CartesianController,
    CartesianControllerConfig,
    DK1_DEFAULT_CONFIG,
    DK1Robot,
)

# ---------------------------------------------------------------------------
# Frame transform: Quest Y-up → robot Z-up world frame
# From TeleVision/teleop/constants_vuer.py  (grd_yup2grd_zup)
#   Quest +X (right)         → robot -Y
#   Quest +Y (up)            → robot +Z
#   Quest +Z (toward viewer) → robot -X
# ---------------------------------------------------------------------------
POS_SCALE = 1.0       # Quest metres → robot metres
GRIPPER_CURVE = 0.1   # <1 = fast start, slow end; 1 = linear; >1 = slow start, fast end

# Quest world Y is always gravity-up = robot Z.
# Horizontal axes are derived per-grab from the controller's pointing direction
# so the mapping stays consistent regardless of Quest world orientation.
_QUEST_UP = np.array([0.0, 1.0, 0.0])


def _build_latch_frame(rmat: np.ndarray) -> np.ndarray:
    """
    Build a 3×3 Quest-world → robot-world rotation from the controller pose at latch.

    The controller's horizontal forward direction (Z axis projected to horizontal
    plane) becomes robot +X. Quest Y (gravity up) → robot +Z.
    Falls back to the fixed default if the controller is pointing nearly vertical.
    """
    ctrl_z = -rmat[:3, 2]                                     # pointing direction = -Z in WebXR
    fwd = ctrl_z - np.dot(ctrl_z, _QUEST_UP) * _QUEST_UP     # project to horizontal plane
    norm = float(np.linalg.norm(fwd))
    if norm < 0.1:
        # Controller nearly vertical — use fixed fallback
        return np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    fwd /= norm
    right = np.cross(fwd, _QUEST_UP)        # right-hand horizontal perpendicular
    right /= np.linalg.norm(right)
    # Rows: robot X = fwd, robot Y = -right (robot Y is left), robot Z = up
    return np.array([fwd, -right, _QUEST_UP], dtype=float)


# ---------------------------------------------------------------------------
# Module-level cart reference and per-session state
# Set by main() before app.start()
# ---------------------------------------------------------------------------
_cart: CartesianController | None = None
_dry_run: bool = False

_ref_rmat:      np.ndarray | None = None
_ref_pos:       np.ndarray | None = None
_ref_rot:       np.ndarray | None = None
_latch_Q2R:     np.ndarray = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
_was_trigger_r: bool = False
_event_count:   int = 0

# ---------------------------------------------------------------------------
# Vuer application — queue_len=1 drops stale events immediately
# ---------------------------------------------------------------------------
app = Vuer(host="0.0.0.0", queries=dict(grid=False), queue_len=1)


@app.add_handler("CONTROLLER_MOVE")
async def on_controller_move(event, session) -> None:
    global _ref_rmat, _ref_pos, _ref_rot, _latch_Q2R
    global _was_trigger_r, _event_count
    try:
        v = event.value

        r = v.get("right")
        if not (isinstance(r, (list, tuple)) and len(r) == 16):
            return
        rmat = np.array(r, dtype=float).reshape(4, 4, order="F")

        rs = v.get("rightState")
        squeeze  = bool(rs.get("squeeze", False))      if isinstance(rs, dict) else False
        trig_val = float(rs.get("triggerValue", 0.0))  if isinstance(rs, dict) else 0.0

        _event_count += 1

        # Front trigger (analog) → proportional gripper with curve, no sticking
        if _cart is not None:
            _cart.set_gripper(trig_val ** GRIPPER_CURVE)

        # Side trigger (squeeze) → grab and move EE
        if squeeze:
            if not _was_trigger_r:
                _ref_rmat  = rmat.copy()
                _latch_Q2R = _build_latch_frame(rmat)
                if _cart is None:
                    _ref_pos = np.array([0.4, 0.0, 0.3])
                    _ref_rot = np.eye(3)
                else:
                    _ref_pos, _ref_rot = _cart.get_ee_pose()
                log.info(f"Latch. ref_pos={np.round(_ref_pos, 3)}")
            elif _ref_rmat is not None:
                delta_pos_q = rmat[:3, 3] - _ref_rmat[:3, 3]
                delta_pos_r = _latch_Q2R @ delta_pos_q * POS_SCALE

                R_delta_q = rmat[:3, :3] @ _ref_rmat[:3, :3].T
                R_delta_r = _latch_Q2R @ R_delta_q @ _latch_Q2R.T

                target_pos = _ref_pos + delta_pos_r
                target_rot = R_delta_r @ _ref_rot

                log.info(
                    f"delta_q={np.round(delta_pos_q*100, 1)} cm  "
                    f"delta_r={np.round(delta_pos_r*100, 1)} cm  "
                    f"target={np.round(target_pos, 3)}"
                )
                if _cart is not None:
                    _cart.set_target_pose(pos=target_pos, rot=target_rot)
        else:
            if _was_trigger_r:
                _ref_rmat = None
        _was_trigger_r = squeeze

    except Exception as e:
        log.warning(f"handler error: {e}")


@app.spawn(start=False)
async def _scene(session, fps: int = 60) -> None:
    log.info("Session connected")
    session.upsert @ DefaultScene()
    session.upsert @ MotionControllers(
        key="controllers", stream=True, left=True, right=True, fps=90
    )
    log.info("MotionControllers upserted")

    prev_count = 0
    prev_t = time.monotonic()

    while True:
        await asyncio.sleep(1.0)
        cnt = _event_count
        now = time.monotonic()
        hz = (cnt - prev_count) / max(now - prev_t, 1e-3)
        prev_count, prev_t = cnt, now
        log.info(f"events={cnt}  rate={hz:.1f}Hz")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _cart, _dry_run

    parser = argparse.ArgumentParser(description="Quest → DK1 Cartesian teleop")
    parser.add_argument("--port", help="DK1 follower serial port (omit with --dry-run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No robot — print computed targets only")
    parser.add_argument("--control-hz", type=float, default=200.0)
    parser.add_argument("--wireless", metavar="QUEST_IP",
                        help="Quest IP for wireless ADB, e.g. 192.168.1.42  "
                             "(requires prior: adb tcpip 5555 while USB-connected)")
    args = parser.parse_args()

    if not args.dry_run and not args.port:
        parser.error("--port is required unless --dry-run is set")

    _dry_run = args.dry_run

    def _adb(cmd: list[str], label: str) -> bool:
        try:
            r = subprocess.run(["adb"] + cmd, capture_output=True, text=True, timeout=8)
            if r.returncode == 0:
                log.info(f"adb {label} OK")
                return True
            log.warning(f"adb {label}: {r.stderr.strip() or 'non-zero exit'}")
        except FileNotFoundError:
            log.warning("adb not found in PATH")
        except subprocess.TimeoutExpired:
            log.warning(f"adb {label} timed out")
        return False

    if args.wireless:
        if _adb(["connect", f"{args.wireless}:5555"], f"connect {args.wireless}"):
            import time as _time; _time.sleep(1.5)  # wait for connection to settle
    _adb(["reverse", "tcp:8012", "tcp:8012"], "reverse")

    robot = None

    if not args.dry_run:
        cfg   = DK1_DEFAULT_CONFIG(args.port)
        robot = DK1Robot(cfg)
        log.info("Connecting robot...")
        robot.connect()
        _cart = CartesianController(robot, CartesianControllerConfig(
            control_hz=args.control_hz,
            control_point_offset=0.02,
            max_dq_per_step=1,
            pos_gain=50.0,
            rot_gain=50.0,
        ))
        _cart.start()
    else:
        log.info("DRY RUN — no robot")

    print(f"Logging to {LOG_FILE}")
    print("Quest browser: http://localhost:8012   (explicit http://)")
    print("Ctrl-C to quit")

    import signal
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    try:
        app.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        log.info("Stopping")
        if _cart  is not None: _cart.stop()
        if robot  is not None: robot.disconnect()
        log.info("Done")


if __name__ == "__main__":
    main()
