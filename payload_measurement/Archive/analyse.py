"""
Analyse payload sweep datasets and produce payload/reach charts.

Scans outputs/payload_measurement/replays/ for all mass_<N>g/ folders.

When a waypoints.json sidecar is present (produced by program_sweep.py), the
dataset is split into dwell and transit episodes and two additional charts are
generated:

  payload_chart_nominal.png — peak torque from dwell frames (steady-state load)
  payload_chart_peak.png    — peak torque from transit frames (dynamic load)

A combined chart using all frames is always written for backwards compatibility:

  payload_chart.png

Usage:
    uv run payload_measurement/analyse.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPLAYS_DIR    = Path("outputs/payload_measurement/replays")
OUTPUT_DIR     = Path("outputs/payload_measurement")
CHART_ALL      = OUTPUT_DIR / "payload_chart.png"
CHART_NOMINAL  = OUTPUT_DIR / "payload_chart_nominal.png"
CHART_PEAK     = OUTPUT_DIR / "payload_chart_peak.png"

JOINT_LABELS       = [f"Joint {i}" for i in range(1, 7)]
JOINT_TORQUE_KEYS  = [f"joint_{i}.torque" for i in range(1, 7)]
JOINT_COLOURS      = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_parquets(replay_dir: Path) -> pd.DataFrame:
    """Load and concatenate all parquet files for a replay directory."""
    parquet_paths = sorted((replay_dir / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {replay_dir}")
    return pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)


def load_replay(replay_dir: Path) -> tuple[float, np.ndarray, list[str]]:
    """Load a replay dataset and return (mass_g, states_array, feature_names).

    For backwards-compatible callers that do not use the episode split.
    """
    with open(replay_dir / "meta" / "info.json") as f:
        info = json.load(f)
    df     = _load_parquets(replay_dir)
    states = np.stack(df["observation.state"].values)
    names  = info["features"]["observation.state"]["names"]
    mass_g = float(replay_dir.name.removeprefix("mass_").removesuffix("g"))
    return mass_g, states, names


def load_replay_split(
    replay_dir: Path,
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[dict]] | None:
    """Load replay split into dwell and transit frames using waypoints.json.

    Returns (mass_g, dwell_states, transit_states, feature_names, waypoints)
    or None if waypoints.json is absent.
    """
    wp_path = replay_dir / "waypoints.json"
    if not wp_path.exists():
        return None

    with open(replay_dir / "meta" / "info.json") as f:
        info = json.load(f)
    with open(wp_path) as f:
        wp_data = json.load(f)

    df    = _load_parquets(replay_dir)
    names = info["features"]["observation.state"]["names"]
    mass_g = float(replay_dir.name.removeprefix("mass_").removesuffix("g"))

    dwell_idxs   = [wp["episode_dwell_idx"]   for wp in wp_data["waypoints"]]
    transit_idxs = [wp["episode_transit_idx"] for wp in wp_data["waypoints"]]

    dwell_mask   = df["episode_index"].isin(dwell_idxs)
    transit_mask = df["episode_index"].isin(transit_idxs)

    dwell_states   = np.stack(df.loc[dwell_mask,   "observation.state"].values)
    transit_states = np.stack(df.loc[transit_mask, "observation.state"].values)

    return mass_g, dwell_states, transit_states, names, wp_data["waypoints"]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(states: np.ndarray, names: list[str]) -> dict:
    """Return peak absolute torque per joint and peak supply current."""
    name_idx = {n: i for i, n in enumerate(names)}
    torques = {
        key: float(np.abs(states[:, name_idx[key]]).max())
        for key in JOINT_TORQUE_KEYS
        if key in name_idx
    }
    current_idx = name_idx.get("external_current_a")
    peak_current = float(states[:, current_idx].max()) if current_idx is not None else None
    return {"torques": torques, "peak_current_a": peak_current}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_torque_figure(
    df: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """Create a torque-vs-payload-mass figure from a DataFrame with joint torque columns."""
    has_current = "peak_current_a" in df.columns
    n_plots = 2 if has_current else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5),
                             squeeze=False, constrained_layout=True)
    ax_torque  = axes[0, 0]
    ax_current = axes[0, 1] if has_current else None

    for key, label, colour in zip(JOINT_TORQUE_KEYS, JOINT_LABELS, JOINT_COLOURS):
        if key in df.columns:
            ax_torque.plot(df["mass_g"], df[key], marker="o", label=label, color=colour)

    ax_torque.set_xlabel("Payload mass (g)")
    ax_torque.set_ylabel("Peak absolute torque (Nm)")
    ax_torque.set_title(title)
    ax_torque.legend(loc="upper left", fontsize=8)
    ax_torque.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_torque.grid(True, linestyle="--", alpha=0.5)
    ax_torque.set_xlim(left=0)
    ax_torque.set_ylim(bottom=0)

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

    return fig


def _print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\nSummary — {label} (peak |torque| per joint, Nm):")
    col_w = 10
    header = f"{'mass_g':>8}  " + "  ".join(f"{l:>{col_w}}" for l in JOINT_LABELS)
    print(header)
    for _, row in df.iterrows():
        vals = "  ".join(
            f"{row[k]:>{col_w}.3f}" for k in JOINT_TORQUE_KEYS if k in row
        )
        print(f"{row['mass_g']:>8.0f}  {vals}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    replay_dirs = sorted(REPLAYS_DIR.glob("mass_*g"))
    if not replay_dirs:
        print(f"No replay datasets found in {REPLAYS_DIR}")
        return

    rows_all     = []
    rows_dwell   = []
    rows_transit = []
    has_split    = False

    for d in replay_dirs:
        try:
            # Try split first (waypoints.json present)
            split = load_replay_split(d)
            if split is not None:
                has_split = True
                mass_g, dwell_states, transit_states, names, _ = split

                m_dwell   = extract_metrics(dwell_states,   names)
                m_transit = extract_metrics(transit_states, names)
                # All-frames baseline (combine dwell + transit)
                all_states = np.concatenate([dwell_states, transit_states], axis=0)
                m_all = extract_metrics(all_states, names)

                row_dwell   = {"mass_g": mass_g, **m_dwell["torques"]}
                row_transit = {"mass_g": mass_g, **m_transit["torques"]}
                row_all     = {"mass_g": mass_g, **m_all["torques"]}

                if m_dwell["peak_current_a"] is not None:
                    row_dwell["peak_current_a"]   = m_dwell["peak_current_a"]
                    row_transit["peak_current_a"] = m_transit["peak_current_a"]
                    row_all["peak_current_a"]     = m_all["peak_current_a"]

                rows_dwell.append(row_dwell)
                rows_transit.append(row_transit)
                rows_all.append(row_all)

                print(f"  {d.name}: {len(dwell_states)} dwell frames, "
                      f"{len(transit_states)} transit frames, "
                      f"dwell max |τ| = {max(m_dwell['torques'].values()):.2f} Nm, "
                      f"transit max |τ| = {max(m_transit['torques'].values()):.2f} Nm")
            else:
                # Legacy: no waypoints.json — load all frames
                mass_g, states, names = load_replay(d)
                metrics = extract_metrics(states, names)
                row = {"mass_g": mass_g, **metrics["torques"]}
                if metrics["peak_current_a"] is not None:
                    row["peak_current_a"] = metrics["peak_current_a"]
                rows_all.append(row)
                print(f"  {d.name}: {states.shape[0]} frames, "
                      f"max |τ| = {max(metrics['torques'].values()):.2f} Nm")
        except Exception as e:
            print(f"  Skipping {d.name}: {e}")

    if not rows_all:
        print("Nothing to plot.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Always write the combined chart
    df_all = pd.DataFrame(rows_all).sort_values("mass_g").reset_index(drop=True)
    fig_all = _make_torque_figure(df_all, "Peak Joint Torque vs Payload Mass")
    fig_all.savefig(CHART_ALL, dpi=150)
    plt.close(fig_all)
    print(f"\nChart saved → {CHART_ALL}")
    _print_summary(df_all, "all frames")

    if has_split and rows_dwell and rows_transit:
        df_dwell   = pd.DataFrame(rows_dwell).sort_values("mass_g").reset_index(drop=True)
        df_transit = pd.DataFrame(rows_transit).sort_values("mass_g").reset_index(drop=True)

        fig_nom = _make_torque_figure(
            df_dwell, "Nominal Torque vs Payload Mass\n(steady-state dwell)"
        )
        fig_nom.savefig(CHART_NOMINAL, dpi=150)
        plt.close(fig_nom)
        print(f"Chart saved → {CHART_NOMINAL}")
        _print_summary(df_dwell, "nominal (dwell)")

        fig_peak = _make_torque_figure(
            df_transit, "Peak Torque vs Payload Mass\n(transit / dynamic)"
        )
        fig_peak.savefig(CHART_PEAK, dpi=150)
        plt.close(fig_peak)
        print(f"Chart saved → {CHART_PEAK}")
        _print_summary(df_transit, "peak (transit)")


if __name__ == "__main__":
    main()
