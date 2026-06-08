import argparse
import builtins
import json
import os
import shutil
import time
import threading
from collections import deque
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SVT_LOG", "0")
os.environ.setdefault("SVT_LOG_LEVEL", "0")

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig
from lerobot.common.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("task", help="Task name (snake_case) — used as dataset name and task description")
parser.add_argument("--user", default="jvg")
parser.add_argument("--episodes", type=int, default=50, help="Target total episodes across all runs")
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--episode-time", type=int, default=90, help="Max episode duration in seconds (press → to end early)")
parser.add_argument("--reset-time", type=int, default=20, help="Reset phase duration in seconds")
parser.add_argument("--robot-port", default="/dev/tty.usbmodem00000000050C1")
parser.add_argument("--leader-port", default="/dev/tty.usbmodem59700732181")
parser.add_argument("--dataset-root", type=Path, default=Path("~/.cache/lerobot/datasets").expanduser())
parser.add_argument("--restart", action="store_true", help="Delete existing dataset and start fresh")
parser.add_argument("--dry-run", action="store_true", help="Test UI and flow without hardware or real dataset")
args = parser.parse_args()

REPO_ID = f"{args.user}/{args.task}"
TASK_DESCRIPTION = args.task.replace("_", " ")
dataset_path = args.dataset_root / REPO_ID

# ── Terminal UI ───────────────────────────────────────────────────────────────
console = Console()
_log_lines: deque[tuple[str, str]] = deque(maxlen=5)
_ui_state = {"phase": "STARTING", "episode": 0, "total": args.episodes, "phase_start": 0.0, "phase_max": 0}
_ui_lock = threading.Lock()

STATE_STYLES = {
    "STARTING":  "bold white",
    "RECORDING": "bold green",
    "RESETTING": "bold yellow",
    "SAVING":    "bold cyan",
    "DONE":      "bold blue",
}

def ui_log(msg: str, style: str = "white"):
    with _ui_lock:
        _log_lines.append((msg, style))

def _say(msg: str, style: str = "white", blocking: bool = False):
    ui_log(msg, style)
    if not args.dry_run:
        log_say(msg, blocking=blocking)

def _set_phase(phase: str, max_s: int = 0):
    with _ui_lock:
        _ui_state["phase"] = phase
        _ui_state["phase_start"] = time.monotonic()
        _ui_state["phase_max"] = max_s

class LiveDisplay:
    """Recomputed on every auto-refresh tick — timer counts live."""
    def __rich_console__(self, console, options):
        with _ui_lock:
            state = _ui_state.copy()
            lines = list(_log_lines)

        elapsed = time.monotonic() - state["phase_start"]
        max_s = state["phase_max"]
        time_str = f"{elapsed:.0f}s / {max_s}s" if max_s else f"{elapsed:.0f}s"

        style = STATE_STYLES.get(state["phase"], "white")
        status_grid = Table.grid(padding=(0, 2))
        status_grid.add_column(style="dim")
        status_grid.add_column()
        status_grid.add_row("Task",    TASK_DESCRIPTION + ("  [dim](dry run)[/dim]" if args.dry_run else ""))
        status_grid.add_row("Phase",   Text(state["phase"], style=style))
        status_grid.add_row("Episode", f"{state['episode']} / {state['total']}")
        status_grid.add_row("Time",    time_str)
        status_grid.add_row("Keys",    "→ end early   ← discard   ESC stop")

        log_text = Text()
        for msg, sty in lines:
            log_text.append(f"  {msg}\n", style=sty)

        yield Group(
            Panel(status_grid, title="[bold]Recording Status[/bold]"),
            Panel(log_text, title="Log"),
        )

# ── Stderr pipe: route C-level writes (SVT, objc) into the UI log ─────────────
_pipe_r_fd, _pipe_w_fd = os.pipe()
_stderr_orig_fd = os.dup(2)
os.dup2(_pipe_w_fd, 2)
os.close(_pipe_w_fd)

def _stderr_reader():
    with os.fdopen(_pipe_r_fd, "rb") as pipe:
        buf = b""
        while True:
            chunk = pipe.read(256)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    ui_log(text, "dim red")

_stderr_thread = threading.Thread(target=_stderr_reader, daemon=True)
_stderr_thread.start()

# ── Hardware / dataset setup ──────────────────────────────────────────────────
if not args.dry_run:
    robot_config = DK1FollowerConfig(
        port=args.robot_port,
        cameras={"front": OpenCVCameraConfig(
            index_or_path=0, width=960, height=540, fps=args.fps, rotation=180, fourcc="MJPG"
        )},
    )
    teleop_config = DK1LeaderConfig(port=args.leader_port)
    robot = DK1Follower(robot_config)
    teleop = DK1Leader(teleop_config)
else:
    robot = teleop = None

if args.restart and dataset_path.exists():
    ui_log(f"Deleting existing dataset at {dataset_path}", "yellow")
    shutil.rmtree(dataset_path)

if not args.dry_run:
    if dataset_path.exists():
        data_dir = dataset_path / "data"
        has_episodes = data_dir.exists() and any(data_dir.rglob("*.parquet"))
        if has_episodes:
            saved_episodes = json.loads((dataset_path / "meta/info.json").read_text()).get("total_episodes", 0)
            ui_log(f"Resuming dataset ({saved_episodes} episodes already recorded)", "cyan")
            dataset = LeRobotDataset.resume(repo_id=REPO_ID, root=str(dataset_path), image_writer_threads=4)
        else:
            ui_log("Existing dataset has no saved episodes — re-creating", "yellow")
            shutil.rmtree(dataset_path)
            dataset = None
    else:
        dataset = None

    if dataset is None:
        ui_log(f"Creating new dataset at {dataset_path}", "cyan")
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID, root=str(dataset_path), fps=args.fps,
            features={**action_features, **obs_features},
            robot_type=robot.name, use_videos=True, image_writer_threads=4,
        )
    already_recorded = dataset.num_episodes
else:
    # Dry run — mock dataset
    class _MockDataset:
        def __init__(self): self.num_episodes = 0
        def save_episode(self): time.sleep(0.4)
        def clear_episode_buffer(self): pass
    dataset = _MockDataset()
    already_recorded = 0
    ui_log("Dry run — no hardware, no real dataset", "yellow")

remaining = args.episodes - already_recorded
ui_log(f"Episodes so far: {already_recorded} / {args.episodes}  ({remaining} remaining)", "dim white")

if remaining <= 0:
    console.print("Target already reached. Pass --episodes N to record more.")
    exit(0)

# ── Recording loop ────────────────────────────────────────────────────────────
_, events = init_keyboard_listener()

def _dry_record_loop(control_time_s, events):
    deadline = time.monotonic() + control_time_s
    while time.monotonic() < deadline:
        if events["exit_early"] or events["stop_recording"]:
            events["exit_early"] = False
            break
        time.sleep(0.05)

_real_print = builtins.print

def _patched_print(*a, sep=" ", end="\n", **kw):
    ui_log(sep.join(str(x) for x in a), "dim white")

with Live(LiveDisplay(), console=console, refresh_per_second=4, screen=False) as live:
    builtins.print = _patched_print

    if not args.dry_run:
        init_rerun(session_name="recording")
        robot.connect()
        teleop.connect()
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    else:
        teleop_action_processor = robot_action_processor = robot_observation_processor = None

    episode_idx = 0
    while episode_idx < remaining and not events["stop_recording"]:
        total_idx = already_recorded + episode_idx + 1
        _ui_state["episode"] = total_idx
        events["exit_early"] = False
        _set_phase("RECORDING", args.episode_time)
        _say(f"Recording episode {total_idx} of {args.episodes}", "green")

        if args.dry_run:
            _dry_record_loop(args.episode_time, events)
        else:
            record_loop(
                robot=robot, events=events, fps=args.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop, dataset=dataset,
                control_time_s=args.episode_time,
                single_task=TASK_DESCRIPTION, display_data=False,
            )

        if events["rerecord_episode"]:
            _say("Discarding — re-recording episode", "yellow")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        episode_idx += 1

        if not events["stop_recording"] and episode_idx < remaining:
            _set_phase("RESETTING", args.reset_time)
            save_thread = threading.Thread(target=dataset.save_episode, daemon=True)
            save_thread.start()
            ui_log(f"Saving episode {total_idx} in background...", "cyan")
            events["exit_early"] = False
            _say("Reset the environment", "yellow", blocking=True)

            if args.dry_run:
                _dry_record_loop(args.reset_time, events)
            else:
                record_loop(
                    robot=robot, events=events, fps=args.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop, control_time_s=args.reset_time,
                    single_task=TASK_DESCRIPTION, display_data=False,
                )
            save_thread.join()
            ui_log(f"Saved episode {total_idx}  ({already_recorded + episode_idx} total)", "cyan")
        else:
            _set_phase("SAVING")
            dataset.save_episode()
            ui_log(f"Saved episode {total_idx}  ({already_recorded + episode_idx} total)", "cyan")

    _set_phase("DONE")
    _say("Recording complete", "bold green")

# ── Teardown ──────────────────────────────────────────────────────────────────
builtins.print = _real_print
os.dup2(_stderr_orig_fd, 2)   # restore real stderr
os.close(_stderr_orig_fd)
# closing fd 2 write end causes _stderr_thread to drain and exit naturally

console.print(f"Session done. Total episodes recorded: {already_recorded + episode_idx} / {args.episodes}")
if not args.dry_run:
    robot.disconnect()
    teleop.disconnect()
