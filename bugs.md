# Bug Reports

## [wip/impedance-controller] Gripper observation stuck at 1.0 (normalized)

**Affects:** `wip/impedance-controller`, `upstream/wip/impedance-controller`
**Not affected:** `main`, `upstream/main`

### Symptom
`observation.gripper.pos` is always ~1.0 (fully closed) regardless of actual gripper position.

### Root cause
Three-layer bug:

**Layer 1 (primary) — `robot.py` (`get_gripper_state`): wrong `np.interp` direction**
`np.interp` requires `xp` to be monotonically increasing. `gripper_open_pos ≈ 0.0` and `gripper_closed_pos = -4.7`, so calling `np.interp(pos, [0.0, -4.7], [0.0, 1.0])` gives decreasing `xp`. Numpy's undefined behaviour for this case returns `fp[-1] = 1.0` for every input except the exact closed position. This is the direct cause of observation.gripper.pos = 1.0 for all frames.

**Layer 2 — `robot.py` (`get_gripper_state`): using config default instead of calibrated value**
Used `cfg.gripper_open_pos` (always 0.0) instead of `self._motor_chain.gripper_open_pos` (the actual post-calibration value). With layer 1 present this made no observable difference, but is still wrong.

**Layer 3 — `motor_chain.py` (`_calibrate_gripper`): stale position readback**
`gripper_open_pos = gripper.getPosition()` was called before any `refresh_motor_status` after `set_zero_position()`. Returns the pre-zeroing cached position instead of ~0.0. Also masked by layer 1 while it existed.

**Files:** `trlc_dk1_control/robot.py` (`get_gripper_state`), `trlc_dk1_control/motor_chain.py` (`_calibrate_gripper`)

### Fix
All three layers fixed on `feat/helper-scripts`:

**`robot.py`** — flip `xp`/`fp` so `xp` is increasing (closed < open):
```python
normalized = np.interp(
    pos[6],
    [cfg.gripper_closed_pos, self._motor_chain.gripper_open_pos],  # [-4.7, ~0] increasing
    [1.0, 0.0],
)
```

**`motor_chain.py`** — move `gripper_open_pos` assignment to after the `refresh_motor_status` that follows `switchControlMode`, so the value reflects the post-zero encoder reading (~0.0):
```python
self._control.switchControlMode(gripper, Control_Type.Torque_Pos)
self._control.refresh_motor_status(gripper)
self._pos[6] = gripper.getPosition()
...
self.gripper_open_pos = gripper.getPosition()  # post-zero = ~0.0
```

### Why main is unaffected
`upstream/main` reads `motor.getPosition()` directly (no normalisation via `np.interp`), so the direction issue doesn't apply.

### Why main is unaffected
The old stack (`upstream/main`) zeros the encoder with `set_zero_position()` and reads `motor.getPosition()` directly in `get_observation()`. Since the encoder is zeroed at open, `getPosition()` returns ~0.0 at open, matching the hardcoded `gripper_open_pos = 0.0`. No mismatch.
