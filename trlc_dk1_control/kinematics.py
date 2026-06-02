"""
DK1Kinematics — MuJoCo-backed forward kinematics + analytical Jacobian.

Loads the same URDF used for gravity compensation and provides:
  - fk(q):        end-effector position + rotation matrix
  - jacobian(q):  6 x num_dofs body Jacobian at the tool tip
"""

from __future__ import annotations

import logging

import numpy as np

from .gravity_comp import _urdf_strip_meshes

logger = logging.getLogger(__name__)


# Candidate end-effector body names, in priority order. MuJoCo's URDF parser
# sometimes folds fixed-joint children into their parent, so we fall back.
_EE_BODY_CANDIDATES = ("tool0", "link6-7")
# Fallback tool tip offset (in link6-7 local frame) — matches URDF gripper_tool0 joint.
_LINK6_TIP_OFFSET = np.array([0.158, 0.0, 0.0])


class DK1Kinematics:
    """
    Forward kinematics and Jacobian for the 6-DoF arm.

    The gripper and camera links are ignored. The tool frame is the URDF's
    `tool0` site (centre of the gripper, +X along the approach direction).

    Args:
        model_path: Path to a URDF (.urdf) or MuJoCo XML (.xml) file.
        num_dofs:   Arm DoFs (default 6, excludes gripper).
    """

    def __init__(self, model_path: str, num_dofs: int = 6,
                 control_point_offset: float | None = None) -> None:
        try:
            import mujoco
        except ImportError as e:
            raise ImportError(
                "mujoco is required for kinematics. Install with: pip install mujoco"
            ) from e

        self._mujoco = mujoco
        self.num_dofs = num_dofs

        if model_path.endswith(".urdf"):
            xml_str = _urdf_strip_meshes(model_path)
            self.model = mujoco.MjModel.from_xml_string(xml_str)
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)

        self.data = mujoco.MjData(self.model)

        # Resolve end-effector body
        self.ee_body_id = -1
        self.tool_offset_local = np.zeros(3)
        for name in _EE_BODY_CANDIDATES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.ee_body_id = bid
                self.ee_body_name = name
                if name == "link6-7":
                    self.tool_offset_local = _LINK6_TIP_OFFSET.copy()
                break

        # Allow caller to override the offset along the approach (X) axis.
        if control_point_offset is not None:
            self.tool_offset_local = np.array([control_point_offset, 0.0, 0.0])

        if self.ee_body_id < 0:
            raise RuntimeError(
                f"None of the candidate EE bodies {_EE_BODY_CANDIDATES} were found "
                "in the MuJoCo model."
            )

        if self.model.nv < num_dofs:
            raise ValueError(
                f"MuJoCo model has nv={self.model.nv}, expected >= {num_dofs}"
            )

        logger.info(
            "DK1Kinematics: EE body '%s' (id=%d, local offset=%s)",
            self.ee_body_name, self.ee_body_id, self.tool_offset_local,
        )

    def _forward(self, q: np.ndarray) -> None:
        self.data.qpos[: self.num_dofs] = q[: self.num_dofs]
        self.data.qvel[:] = 0.0
        self._mujoco.mj_kinematics(self.model, self.data)
        self._mujoco.mj_comPos(self.model, self.data)  # required for mj_jac

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics of the tool frame.

        Returns:
            pos: (3,) tool position in world frame.
            rot: (3, 3) tool rotation matrix in world frame.
        """
        self._forward(q)
        body_pos = self.data.xpos[self.ee_body_id].copy()
        body_mat = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        tip_pos = body_pos + body_mat @ self.tool_offset_local
        return tip_pos, body_mat

    def jacobian(
        self, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        6 x num_dofs spatial Jacobian at the tool tip.

        Returns:
            J:   (6, num_dofs) — rows 0..2 linear, 3..5 angular (world frame).
            pos: (3,) tool position.
            rot: (3, 3) tool rotation.
        """
        self._forward(q)
        body_pos = self.data.xpos[self.ee_body_id]
        body_mat = self.data.xmat[self.ee_body_id].reshape(3, 3)
        tip_pos = body_pos + body_mat @ self.tool_offset_local

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self._mujoco.mj_jac(
            self.model, self.data, jacp, jacr, tip_pos, self.ee_body_id
        )
        J = np.vstack([jacp[:, : self.num_dofs], jacr[:, : self.num_dofs]])
        return J, tip_pos.copy(), body_mat.copy()
