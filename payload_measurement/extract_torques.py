"""Extract reach-torque calibration plateaus from a Rerun .rrd recording.

Loads the recording, reads tool0.reach_mm and observation.joint_N.torque,
detects steady-state plateaus (where reach is ~constant), groups nearby
plateaus into distinct hold positions, and outputs a CSV of
(reach_mm, j1_torque, ..., j6_torque) per unique hold.

Usage:
    uv run python payload_measurement/data-extraction/extract_calibration.py <path_to_rrd>

Output CSV is saved next to the input file with '_plateaus.csv' suffix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import rerun as rr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RRD_PATH: Path | None = None   # set from CLI argument
OUTPUT_CSV: Path | None = None  # derived from RRD_PATH

# Plateau detection: contiguous segments where reach std < threshold
SEGMENT_MIN_S = 2.0          # minimum hold duration to count as a plateau
REACH_STD_THRESH_MM = 10.0   # max std-dev of reach within a segment

# De-duplication: plateaus within REACH_CLUSTER_MM of each other → one group
# Pick the longest/most-stable hold per group
REACH_CLUSTER_MM = 5.0

# Torque sampling: average over the middle fraction of each plateau segment
TORQUE_SAMPLE_FRACTION = 0.5   # use central 50% of segment to avoid transients

# Minimum reach filter: ignore plateaus below this (capped by joint 4 anyway)
MIN_REACH_MM = 350.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_timeseries(rec, entity_path: str) -> pd.DataFrame:
    """Return a DataFrame with columns ['time_s', 'value'] for a scalar entity."""
    view = rec.view(index="log_time", contents=entity_path)
    table = view.select().read_all()

    val_col = [c for c in table.column_names if c not in ("log_time", "log_tick")][0]
    times_ns = table["log_time"].cast(pa.int64()).to_pylist()
    raw_vals = table[val_col].to_pylist()

    values, times = [], []
    for t, v in zip(times_ns, raw_vals):
        if v is None or t is None:
            continue
        val = v[0] if isinstance(v, (list, tuple)) else v
        if val is None:
            continue
        values.append(float(val))
        times.append(float(t))

    times_s = np.array(times) / 1e9
    times_s -= times_s[0]
    return pd.DataFrame({"time_s": times_s, "value": np.array(values)})


def _find_stable_segments(reach_df: pd.DataFrame) -> list[dict]:
    """
    Scan reach timeseries and return a list of stable segments.
    Each segment: {t_start, t_end, duration_s, mean_reach_mm, std_reach_mm}
    """
    times = reach_df["time_s"].values
    values = reach_df["value"].values

    segments = []
    seg_start = 0

    for i in range(1, len(times)):
        window = values[seg_start:i+1]
        if np.std(window) > REACH_STD_THRESH_MM:
            # Current point broke stability — close segment at i-1
            seg_vals = values[seg_start:i]
            duration = times[i-1] - times[seg_start]
            if duration >= SEGMENT_MIN_S and len(seg_vals) >= 5:
                segments.append({
                    "t_start": times[seg_start],
                    "t_end": times[i-1],
                    "duration_s": duration,
                    "mean_reach_mm": float(np.mean(seg_vals)),
                    "std_reach_mm": float(np.std(seg_vals)),
                })
            seg_start = i

    # Close final segment
    seg_vals = values[seg_start:]
    duration = times[-1] - times[seg_start]
    if duration >= SEGMENT_MIN_S and len(seg_vals) >= 5:
        segments.append({
            "t_start": times[seg_start],
            "t_end": times[-1],
            "duration_s": duration,
            "mean_reach_mm": float(np.mean(seg_vals)),
            "std_reach_mm": float(np.std(seg_vals)),
        })

    return segments


def _cluster_segments(segments: list[dict]) -> list[list[dict]]:
    """Group segments by similar reach (within REACH_CLUSTER_MM)."""
    if not segments:
        return []

    # Sort by reach for greedy clustering
    sorted_segs = sorted(segments, key=lambda s: s["mean_reach_mm"])
    clusters: list[list[dict]] = [[sorted_segs[0]]]

    for seg in sorted_segs[1:]:
        if abs(seg["mean_reach_mm"] - clusters[-1][0]["mean_reach_mm"]) <= REACH_CLUSTER_MM:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    return clusters


def _best_segment(cluster: list[dict]) -> dict:
    """Pick the longest segment from a cluster (most data = most stable average)."""
    return max(cluster, key=lambda s: s["duration_s"])


def _average_cluster(
    cluster: list[dict],
    torque_dfs: dict[str, pd.DataFrame],
    pos_dfs: dict[str, pd.DataFrame],
) -> dict:
    """Average torque/position across all segments in a cluster (repeated holds)."""
    # Weighted average by segment duration (longer holds = more weight)
    weights = np.array([s["duration_s"] for s in cluster])
    total_w = weights.sum()

    reach_avg = np.average([s["mean_reach_mm"] for s in cluster], weights=weights)

    row = {
        "reach_mm": round(reach_avg, 1),
        "duration_s": round(total_w, 1),
        "n_holds": len(cluster),
    }

    for jname, tdf in torque_dfs.items():
        vals = np.array([_sample_torque(tdf, seg) for seg in cluster])
        row[f"{jname}_torque_Nm"] = round(np.average(vals, weights=weights), 3)

    for jname, pdf in pos_dfs.items():
        vals = np.array([_sample_torque(pdf, seg) for seg in cluster])
        row[f"{jname}_pos_rad"] = round(np.average(vals, weights=weights), 4)

    return row


def _sample_torque(torque_df: pd.DataFrame, seg: dict) -> float:
    """Average torque over the central TORQUE_SAMPLE_FRACTION of the segment."""
    t_start, t_end = seg["t_start"], seg["t_end"]
    margin = (t_end - t_start) * (1 - TORQUE_SAMPLE_FRACTION) / 2
    t_lo, t_hi = t_start + margin, t_end - margin
    mask = (torque_df["time_s"] >= t_lo) & (torque_df["time_s"] <= t_hi)
    subset = torque_df.loc[mask, "value"]
    return float(subset.mean()) if len(subset) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(rrd_path: Path, output_csv: Path) -> None:
    print(f"Loading: {rrd_path}")
    rec = rr.dataframe.load_recording(str(rrd_path))

    print("Extracting tool0.reach_mm ...")
    reach_df = _extract_timeseries(rec, "/tool0.reach_mm")
    print(f"  {len(reach_df)} samples over {reach_df['time_s'].iloc[-1]:.1f} s")

    torque_dfs: dict[str, pd.DataFrame] = {}
    pos_dfs: dict[str, pd.DataFrame] = {}
    for j in range(1, 7):
        print(f"Extracting joint_{j}.torque + .pos ...")
        torque_dfs[f"j{j}"] = _extract_timeseries(rec, f"/observation.joint_{j}.torque")
        pos_dfs[f"j{j}"] = _extract_timeseries(rec, f"/observation.joint_{j}.pos")

    # --- Find stable segments ---
    segments = _find_stable_segments(reach_df)
    print(f"\nFound {len(segments)} stable segments")

    # --- Filter by minimum reach ---
    segments = [s for s in segments if s["mean_reach_mm"] >= MIN_REACH_MM]
    print(f"After filtering reach >= {MIN_REACH_MM:.0f} mm: {len(segments)} segments")

    # --- Cluster by reach → one row per distinct hold position ---
    clusters = _cluster_segments(segments)
    print(f"Clustered into {len(clusters)} distinct reach positions\n")

    rows = []
    for cluster in sorted(clusters, key=lambda c: np.mean([s["mean_reach_mm"] for s in c])):
        row = _average_cluster(cluster, torque_dfs, pos_dfs)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    df.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_rrd> [output_csv]")
        sys.exit(1)

    rrd = Path(sys.argv[1]).resolve()
    if len(sys.argv) >= 3:
        out = Path(sys.argv[2]).resolve()
    else:
        out = rrd.with_name(rrd.stem + "_plateaus.csv")

    main(rrd, out)
