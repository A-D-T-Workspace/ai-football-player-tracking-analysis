from pathlib import Path

import pandas as pd
import numpy as np

INPUT_PATH = Path(
    "outputs/buffer_60_tracking_observations.csv"
)

OUTPUT_PATH = Path(
    "outputs/buffer_60_track_summary.csv"
)

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Tracking observations not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = [
        "frame", "track_id", "x1", "y1", "x2", "y2"
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if data.empty:
        raise ValueError("Tracking observations file is empty.")

    data["center_x"] = (data["x1"] + data["x2"]) / 2
    data["center_y"] = (data["y1"] + data["y2"]) / 2

    data["box_height"] = (
        data["y2"] - data["y1"]
    ).replace(0, np.nan)

    data = data.sort_values(
        ["track_id", "frame"]
    ).copy()

    grouped = data.groupby("track_id")

    data["frame_gap"] = grouped["frame"].diff()

    dx = grouped["center_x"].diff()
    dy = grouped["center_y"].diff()

    data["center_jump"] = np.sqrt(dx**2 + dy**2)

    data["normalized_jump"] = (
        data["center_jump"] / data["box_height"]
    )

    tracks = []

    for track_id, group in data.groupby("track_id"):
        frames = sorted(group["frame"].unique())

        gaps = [
            later - earlier - 1
            for earlier, later in zip(frames, frames[1:])
            if later - earlier > 1
        ]

        span = frames[-1] - frames[0] + 1

        tracks.append({
            "track_id": int(track_id),
            "observed_frames": len(frames),
            "span_frames": span,
            "coverage_pct": (
                100 * len(frames) / span
                if span > 0 else 0
            ),
            "gap_count": len(gaps),
            "largest_gap": max(gaps, default=0),
        })

    track_summary = pd.DataFrame(tracks)

    short_gaps = data[
        data["frame_gap"].between(1, 4)
    ]

    print(f"Observations: {len(data)}")
    print(f"Unique track IDs: {data['track_id'].nunique()}")
    print(f"Frames represented: {data['frame'].nunique()}")

    print(
        "Tracks with gaps: "
        f"{(track_summary['gap_count'] > 0).sum()}"
    )

    print(
        f"Total internal gaps: "
        f"{track_summary['gap_count'].sum()}"
    )

    print(
        "Median normalized movement: "
        f"{short_gaps['normalized_jump'].median():.3f}"
    )

    print(
        "95th percentile normalized movement: "
        f"{short_gaps['normalized_jump'].quantile(0.95):.3f}"
    )

    print("\nLowest-coverage tracks:")

    print(
        track_summary.sort_values("coverage_pct")
        .head(10)
        .round(2)
        .to_string(index=False)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    track_summary.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nTrack summary saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
