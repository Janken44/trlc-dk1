<p align="center">
    <img src="media/xray.jpg">
</p>
<p align="center">
    <a href="https://docs.robot-learning.co/">
        <img src="https://img.shields.io/badge/Documentation-📕-blue" alt="Chat on Discord"></a>
    <a href="https://discord.gg/PTZ3CN5WkJ">
        <img src="https://img.shields.io/discord/1409155673572249672?color=7289DA&label=Discord&logo=discord&logoColor=white"></a>
    <a href="https://x.com/JannikGrothusen">
        <img src="https://img.shields.io/twitter/follow/Jannik?style=social"></a>
    <a href="https://www.robot-learning.co/">
        <img src=https://img.shields.io/badge/Order%20a%20kit-8A2BE2></a>
</p>

<h1 align="center">An Open Source Dev Kit for AI-native Robotics</h1>
<p align="center">by The Robot Learning Company</p>

## Demo

<p align="center">
    <img src="media/demo.gif">
</p>

## CAD

<table align="center">
<tr>
<td width="50%">
<a href="https://github.com/robot-learning-co/trlc-dk1/blob/main/hardware/TRLC-DK1-Follower_v0.3.0.step" target="_blank">
TRLC-DK1 v0.3.0 Follower CAD<br>
<img src="media/follower_cad.png" width="100%">
</a>
</td>
<td width="50%">
<a href="https://a360.co/481PSQH" target="_blank">
TRLC-DK1 v0.2.0 Leader CAD<br>
<img src="media/leader_cad.png" width="100%">
</a>
</td>
</tr>
</table>
Copyright 2025-2026 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.

## Installation

```
git clone https://github.com/robot-learning-co/trlc-dk1.git
uv venv
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
GIT_LFS_SKIP_SMUDGE=1 is needed to pull LeRobot as a dependency.


This repo uses [LeRobot's plugin conventions](https://huggingface.co/docs/lerobot/integrate_hardware#using-your-own-lerobot-devices-) to be automatically detected by a LeRobot installation in the same Python environment.


## Examples

Use [LeRobot's CLI](https://huggingface.co/docs/lerobot/il_robots) to identify your teleop, robot, and camera ports:

```
uv run lerobot-find-port
uv run lerobot-find-cameras
```

<details>
<summary>Example I: Single Arm Teleoperation
</summary>

```bash
uv run lerobot-teleoperate \
    --robot.type=dk1_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.joint_velocity_scaling=0.2 \
    --teleop.type=dk1_leader \
    --teleop.port=/dev/ttyACM1 \
    --robot.cameras="{ 
        context: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 60, fourcc: "MJPG"}, 
        wrist: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 60, rotation: 180, fourcc: "MJPG"}
      }" \
    --display_data=true
```
</details>

<details>
<summary>Example II: Bimanual Recording
</summary>

```bash
lerobot-record \
    --robot.type=bi_dk1_follower \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.joint_velocity_scaling=1.0 \
    --teleop.type=bi_dk1_leader \
    --teleop.right_arm_port=/dev/ttyACM2 \
    --teleop.left_arm_port=/dev/ttyACM3 \
    --robot.cameras="{ 
        head: {type: opencv, index_or_path: /dev/video0, width: 960, height: 540, fps: 60, fourcc: "MJPG"},
        right_wrist: {type: opencv, index_or_path: /dev/video2, width: 960, height: 540, fps: 60, rotation: 180, fourcc: "MJPG"},
        left_wrist: {type: opencv, index_or_path: /dev/video4, width: 960, height: 540, fps: 60, rotation: 180, fourcc: "MJPG"},
      }" \
    --dataset.repo_id=$USER/my_test_dataset \
    --dataset.push_to_hub=false \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=20 \
    --dataset.single_task="Test the LeRobot recording pipeling."
```
</details>

<details>
<summary>Example III: Quest VR Cartesian Teleoperation
</summary>

Control the arm end-effector directly in Cartesian space using a Meta Quest controller. The arm follows your hand movements in 6DoF — position and orientation — mapped from the Quest world frame to the robot world frame.

**Requirements**
- Meta Quest 2/3/Pro with developer mode enabled (Meta Horizon app → device → Developer Mode)
- USB cable (Quest to Mac)
- `adb` (Android Debug Bridge) installed as a system tool — macOS: `brew install android-platform-tools`
- `vuer` is included as a package dependency and installed automatically with `uv pip install -e .`

**Run**
```bash
# USB (~20ms latency)
uv run python examples/quest_teleop.py --port YOUR_PORT

# Dry run (no robot — verify tracking and frame alignment before connecting hardware)
uv run python examples/quest_teleop.py --dry-run
```

**Logging**

The script logs to `quest_teleop.log` in the repo root (not the terminal). The file is overwritten on each run.

**Connect from headset**

Open Meta Browser on Quest, navigate to:
```
http://localhost:8012
```
Tap **Enter VR**

**Controls**

| Input | Action |
|-------|--------|
| Side trigger (squeeze, hold) | Grab EE setpoint — arm follows controller delta |
| Side trigger (release) | EE holds last position |
| Front trigger (analog) | Proportional gripper close |

The horizontal forward axis is derived from the controller's pointing direction at latch time.

**Key parameters** (`CartesianControllerConfig`)

| Parameter | Quest teleop default | Effect |
|-----------|---------------------|--------|
| `control_hz` | 200 | IK loop rate (Hz) |
| `control_point_offset` | 0.02 | Control pivot along tool axis (m); 0 = joint 4, 0.158 = gripper tip |
| `pos_gain` | 50.0 | Position tracking bandwidth (1/s); higher = faster tracking |
| `rot_gain` | 50.0 | Rotation tracking bandwidth (1/s) |
| `max_dq_per_step` | 1.0 | Max joint step per cycle (rad); hardware limits apply beyond this |
| `pos_min` / `pos_max` | `[-0.5,-0.5,0.05]` / `[0.7,0.5,0.8]` | Workspace bounds for EE setpoint (m) |

**Discarded: Wireless control**
**Setup (one-time per Quest boot)**
```bash
# With Quest plugged in via USB, accept "Allow USB debugging" on the headset:
adb tcpip 5555
# Unplug — wireless ADB now available until next reboot

# Wireless ADB (requires setup above; higher jitter, not recommended for precise control)
uv run python examples/quest_teleop.py --port /dev/ttyACM0 --wireless <quest-ip>
```

</details>

## URDF

<p align="center">
    <img src="media/follower_urdf.png">
</p>

The follower arm URDF with visual (GLB) and collision (STL) meshes is available in [`urdf/follower/`](urdf/follower/).

## Acknowledgements

- [GELLO](https://wuphilipp.github.io/gello_site/) by Philipp Wu et al.
- [Low-Cost Robot Arm](https://github.com/AlexanderKoch-Koch/low_cost_robot) by Alexander Koch
- [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace, Inc.
- [SO-100](https://github.com/TheRobotStudio/SO-ARM100) by TheRobotStudio
- [OpenArm](https://openarm.dev/) by Enactic, Inc.
