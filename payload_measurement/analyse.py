"""
Analyse payload sweep replay datasets and produce a torque vs payload chart.

Scans outputs/payload_measurement/replays/ for all mass_<N>g/ folders,
extracts peak absolute torque per joint and peak supply current, and saves
a chart to outputs/payload_measurement/payload_chart.png.

Usage:
    uv run examples/payload_measurement/analyse.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPLAYS_DIR = Path("outputs/payload_measurement/replays")
OUTPUT_CHART = Path("outputs/payload_measurement/payload_chart.png")

JOINT_LABELS = [f"Joint {i}" for i in range(1, 7)]
JOINT_TORQUE_KEYS = [f"joint_{i}.torque" for i in range(1, 7)]

# Colours for each joint line
JOINT_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def load_replay(replay_dir: Path) -> tuple[float, np.ndarray, dict]:
    """Load a replay dataset and return (mass_g, states_array, feature_names)."""
    with open(replay_dir / "meta" / "info.json") as f:
        info = json.load(f)

    parquet_paths = sorted((replay_dir / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {replay_dir}")

    dfs = [pd.read_parquet(p) for p in parquet_paths]
    df  = pd.concat(dfs, ignore_index=True)

    states = np.stack(df["observation.state"].values)   # (N, 15)
    names  = info["features"]["observation.state"]["names"]

    # Parse mass from directory name: "mass_100g" → 100.0
    mass_g = float(replay_dir.name.removeprefix("mass_").removesuffix("g"))
    return mass_g, states, names


def extract_metrics(states: np.ndarray, names: list[str]) -> dict:
    """Return peak absolute torque per joint and peak supply current."""
    name_idx = {n: i for i, n in enumerate(names)}
    torques = {
        key: np.abs(states[:, name_idx[key]]).max()
        for key in JOINT_TORQUE_KEYS
    }
    current_idx = name_idx.get("external_current_a")
    peak_current = float(states[:, current_idx].max()) if current_idx is not None else None
    return {"torques": torques, "peak_current_a": peak_current}


def main() -> None:
    replay_dirs = sorted(REPLAYS_DIR.glob("mass_*g"))
    if not replay_dirs:
        print(f"No replay datasets found in {REPLAYS_DIR}")
        return

    rows = []
    for d in replay_dirs:
        try:
            mass_g, states, names = load_replay(d)
            metrics = extract_metrics(states, names)
            row = {"mass_g": mass_g, **metrics["torques"]}
            if metrics["peak_current_a"] is not None:
                row["peak_current_a"] = metrics["peak_current_a"]
            rows.append(row)
            print(f"  {d.name}: {states.shape[0]} frames, "
                  f"max |τ| = {max(metrics['torques'].values()):.2f} Nm")
        except Exception as e:
            print(f"  Skipping {d.name}: {e}")

    if not rows:
        print("Nothing to plot.")
        return

    df = pd.DataFrame(rows).sort_values("mass_g").reset_index(drop=True)

    has_current = "peak_current_a" in df.columns
    n_plots = 2 if has_current else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5),
                             squeeze=False, constrained_layout=True)
    ax_torque  = axes[0, 0]
    ax_current = axes[0, 1] if has_current else None

    # --- Torque plot ---
    for key, label, colour in zip(JOINT_TORQUE_KEYS, JOINT_LABELS, JOINT_COLOURS):
        ax_torque.plot(df["mass_g"], df[key], marker="o", label=label, color=colour)

    ax_torque.set_xlabel("Payload mass (g)")
    ax_torque.set_ylabel("Peak absolute torque (Nm)")
    ax_torque.set_title("Peak Joint Torque vs Payload Mass")
    ax_torque.legend(loc="upper left", fontsize=8)
    ax_torque.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_torque.grid(True, linestyle="--", alpha=0.5)
    ax_torque.set_xlim(left=0)
    ax_torque.set_ylim(bottom=0)

    # --- Current plot ---
    if ax_current is not None:
        ax_current.plot(df["mass_g"], df["peak_current_a"],
                        marker="o", color="#333333")
        ax_current.set_xlabel("Payload mass (g)")
        ax_current.set_ylabel("Peak supply current (A)")
        ax_current.set_title("Peak Supply Current vs Payload Mass")
        ax_current.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_current.grid(True, linestyle="--", alpha=0.5)
        ax_current.set_xlim(left=0)
        ax_current.set_ylim(bottom=0)

    OUTPUT_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_CHART, dpi=150)
    print(f"\nChart saved → {OUTPUT_CHART}")

    # Print summary table
    print("\nSummary (peak |torque| per joint, Nm):")
    col_w = 10
    header = f"{'mass_g':>8}  " + "  ".join(f"{l:>{col_w}}" for l in JOINT_LABELS)
    print(header)
    for _, row in df.iterrows():
        vals = "  ".join(f"{row[k]:>{col_w}.3f}" for k in JOINT_TORQUE_KEYS)
        print(f"{row['mass_g']:>8.0f}  {vals}")


if __name__ == "__main__":
    main()
