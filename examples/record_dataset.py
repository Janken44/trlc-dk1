import argparse
import builtins
import json
import os
import shutil
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
_ui_state = {"phase": "STARTING", "episode": 0, "total": args.episodes, "phase_start": 0.0, "phase_max": 0, "saving": False}
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
        if state["saving"]:
            status_grid.add_row("", Text("saving previous episode…", style="dim cyan"))
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

# ── Stderr pipe: set up fds now, activate redirect inside Live ────────────────
_pipe_r_fd, _pipe_w_fd = os.pipe()
_stderr_orig_fd = os.dup(2)
# NOTE: os.dup2(_pipe_w_fd, 2) is called AFTER Live starts so pre-Live
# exceptions print to the real terminal instead of being silently swallowed.

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
                if text and not text.startswith("Map:"):
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

def _repair_episodes_meta(dataset_path: Path) -> None:
    """Reconstruct meta/episodes/ from data/*.parquet when finalize() was never called.

    LeRobotDataset.resume() requires meta/episodes/ to exist. With default
    metadata_buffer_size=10, sessions shorter than 10 episodes never flush
    the buffer to disk. This function reconstructs the parquet from existing data.
    """
    episodes_dir = dataset_path / "meta" / "episodes"
    if episodes_dir.exists():
        existing = list(episodes_dir.rglob("*.parquet"))
        if existing:
            try:
                for f in existing:
                    pq.read_schema(str(f))  # raises ArrowInvalid if corrupted
                return  # all valid
            except Exception:
                pass  # fall through to repair
        shutil.rmtree(episodes_dir)

    data_files = sorted((dataset_path / "data").rglob("*.parquet"))
    if not data_files:
        return

    info = json.loads((dataset_path / "meta/info.json").read_text())
    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    df = pq.read_table(data_files[0]).to_pandas()
    for f in data_files[1:]:
        df = pd.concat([df, pq.read_table(f).to_pandas()], ignore_index=True)

    tasks_df = pd.read_parquet(dataset_path / "meta/tasks.parquet")
    task_name_map = {int(row["task_index"]): name for name, row in tasks_df.iterrows()}

    ep_groups = df.groupby("episode_index")["index"].agg(["min", "max", "count"]).reset_index()
    ep_task_groups = df.groupby("episode_index")["task_index"].unique()

    records = []
    cumulative_s = 0.0
    for _, row in ep_groups.iterrows():
        ep_idx = int(row["episode_index"])
        length = int(row["count"])
        duration_s = length / fps

        episode_tasks = [task_name_map[int(ti)] for ti in sorted(ep_task_groups[ep_idx])]

        rec = {
            "episode_index": ep_idx,
            "tasks": episode_tasks,
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": int(row["min"]),
            "dataset_to_index": int(row["max"]) + 1,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for vk in video_keys:
            rec[f"videos/{vk}/chunk_index"] = 0
            rec[f"videos/{vk}/file_index"] = 0
            rec[f"videos/{vk}/from_timestamp"] = cumulative_s
            rec[f"videos/{vk}/to_timestamp"] = cumulative_s + duration_s
        cumulative_s += duration_s
        records.append(rec)

    out_path = episodes_dir / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), out_path, compression="snappy")


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
            _repair_episodes_meta(dataset_path)
            dataset = LeRobotDataset.resume(repo_id=REPO_ID, root=str(dataset_path), image_writer_threads=4)
            dataset.meta._metadata_buffer_size = 1
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
            metadata_buffer_size=1,
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

with Live(LiveDisplay(), console=console, refresh_per_second=4, screen=True) as live:
    builtins.print = _patched_print
    os.dup2(_pipe_w_fd, 2)   # activate stderr capture now that Live owns the screen
    os.close(_pipe_w_fd)

    if not args.dry_run:
        ui_log("Connecting to hardware…", "dim white")
        init_rerun(session_name="recording")
        robot.connect()
        teleop.connect()
        ui_log("Hardware ready", "dim white")
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    else:
        teleop_action_processor = robot_action_processor = robot_observation_processor = None

    episode_idx = 0
    try:
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
                    single_task=TASK_DESCRIPTION, display_data=True,
                )

            if events["rerecord_episode"]:
                _say("Discarding — re-recording episode", "yellow")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # ESC during recording → discard current episode and stop
            if events["stop_recording"]:
                _say("Discarding current episode", "yellow")
                dataset.clear_episode_buffer()
                break

            episode_idx += 1

            if episode_idx < remaining:
                # ── Reset phase: save synchronously, then wait for → ──────────
                _set_phase("RESETTING")
                with _ui_lock:
                    _ui_state["saving"] = True
                save_thread = threading.Thread(target=dataset.save_episode, daemon=True)
                save_thread.start()
                events["exit_early"] = False
                _say("Reset the environment", "yellow", blocking=True)

                # Suppress → presses until save finishes, then notify.
                # Robot stays live but Rerun is off (display_data=False) to
                # avoid streaming stale data into the viewer during save.
                def _hold_until_saved(st=save_thread):
                    while st.is_alive():
                        events["exit_early"] = False
                        time.sleep(0.05)
                    with _ui_lock:
                        _ui_state["saving"] = False
                    ui_log(f"Saved ep {total_idx} — press → for next episode", "cyan")

                hold_thread = threading.Thread(target=_hold_until_saved, daemon=True)
                hold_thread.start()

                if args.dry_run:
                    hold_thread.join()
                    _dry_record_loop(99999, events)
                else:
                    record_loop(
                        robot=robot, events=events, fps=args.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop, control_time_s=99999,
                        single_task=TASK_DESCRIPTION, display_data=False,
                    )
                save_thread.join()
                with _ui_lock:
                    _ui_state["saving"] = False
            else:
                # Last episode — save synchronously
                _set_phase("SAVING")
                dataset.save_episode()
                ui_log(f"Saved episode {total_idx}  ({already_recorded + episode_idx} total)", "cyan")

        _set_phase("DONE")
        _say("Recording complete", "bold green")
    finally:
        # Always flush metadata buffer — protects against crashes mid-session.
        # If the process dies without finalize(), meta/episodes/ is left incomplete
        # or corrupted, which breaks the next resume().
        if not args.dry_run:
            try:
                dataset.meta.finalize()
            except Exception:
                pass

# ── Teardown ──────────────────────────────────────────────────────────────────
builtins.print = _real_print
os.dup2(_stderr_orig_fd, 2)   # restore real stderr
os.close(_stderr_orig_fd)

if not args.dry_run:
    dataset.finalize()
    robot.disconnect()
    teleop.disconnect()

console.print(f"Session done. Total episodes recorded: {already_recorded + episode_idx} / {args.episodes}")
