"""Real-time 3D arm visualization for Rerun.

Computes forward kinematics via MuJoCo and logs the arm skeleton
as a line strip + joint markers to Rerun's 3D spatial view.

Usage::

    from trlc_dk1_control.rerun_viz import ArmViz

    viz = ArmViz()                       # loads MuJoCo model from default URDF
    viz.log(joint_positions_6dof)        # call each frame
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None  # type: ignore[assignment]

try:
    import rerun as rr
except ImportError:
    rr = None  # type: ignore[assignment]

from .gravity_comp import GravityCompensator

# Offset from link6-7 frame origin to tool0 (URDF fixed joint)
_TOOL0_LOCAL = np.array([0.158, 0.0, 0.0])

# Body indices for the arm skeleton (world → link1-2 → … → link6-7 → tool0)
_SKELETON_BODIES = [
    "link1-2",
    "link2-3",
    "link3-4",
    "link4-5",
    "link5-6",
    "link6-7",
]


class ArmViz:
    """Lightweight FK → Rerun logger for the DK1 arm skeleton."""

    def __init__(self, urdf_path: str | None = None) -> None:
        from .config import _DEFAULT_URDF

        urdf_path = urdf_path or _DEFAULT_URDF
        self._gc = GravityCompensator(urdf_path)
        m = self._gc.mj_model
        self._body_ids = [m.body(name).id for name in _SKELETON_BODIES]
        self._link67_id = m.body("link6-7").id

    def _compute_skeleton(self, qpos_6: np.ndarray) -> np.ndarray:
        """Return (N, 3) array of skeleton vertices from base to tool0."""
        d = self._gc.mj_data
        d.qpos[:6] = qpos_6[:6]
        mujoco.mj_kinematics(self._gc.mj_model, d)

        pts = [np.array([0.0, 0.0, 0.0])]  # world origin (base)
        for bid in self._body_ids:
            pts.append(d.xpos[bid].copy())

        # tool0 = link6-7 frame + rotated local offset
        rot = d.xmat[self._link67_id].reshape(3, 3)
        tool0 = d.xpos[self._link67_id] + rot @ _TOOL0_LOCAL
        pts.append(tool0)

        return np.array(pts)

    def log(self, qpos_6: np.ndarray, *, entity_root: str = "arm") -> None:
        """Compute FK and log skeleton + tool0 coordinate readout to Rerun."""
        pts = self._compute_skeleton(qpos_6)
        tool0 = pts[-1]

        # 3D skeleton
        rr.log(f"{entity_root}/skeleton", rr.LineStrips3D(
            [pts],
            colors=[[60, 60, 60]],
            radii=[0.004],
        ))
        rr.log(f"{entity_root}/joints", rr.Points3D(
            pts[1:-1],  # skip base origin and tool0
            colors=[[40, 120, 200]],
            radii=[0.010],
        ))
        rr.log(f"{entity_root}/tool0", rr.Points3D(
            [pts[-1]],
            colors=[[220, 50, 50]],
            radii=[0.014],
        ))

        # Tool0 coordinates as time series (mm)
        rr.log("tool0.x_mm", rr.Scalars(tool0[0] * 1000))
        rr.log("tool0.y_mm", rr.Scalars(tool0[1] * 1000))
        rr.log("tool0.z_mm", rr.Scalars(tool0[2] * 1000))
        reach_mm = np.sqrt(tool0[0]**2 + tool0[1]**2) * 1000
        rr.log("tool0.reach_mm", rr.Scalars(reach_mm))
