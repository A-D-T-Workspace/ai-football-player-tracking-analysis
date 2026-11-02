import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path(
    "outputs/buffer_60_tracking_observations.csv"
)

OUTPUT_PATH = Path(
    "outputs/player_movement_metrics.csv"
)

USABLE_OUTPUT_PATH = Path(
    "outputs/player_movement_metrics_usable.csv"
)

FPS = 30
MAX_FRAME_GAP = 4
MIN_OBSERVATIONS = 15
MIN_DURATION_SECONDS = 1.0

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Tracking observations not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "frame", "track_id", "x1", "y1",
        "x2", "y2"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Tracking observations file is empty.")

    df["foot_x"] = (df["x1"] + df["x2"]) / 2
    df["foot_y"] = df["y2"]

    df["time_seconds"] = df["frame"] / FPS

    df = df.sort_values(
        ["track_id", "frame"]
    ).copy()

    df["dx"] = df.groupby("track_id")["foot_x"].diff()
    df["dy"] = df.groupby("track_id")["foot_y"].diff()

    df["displacement_pixels"] = np.sqrt(
        df["dx"] ** 2 + df["dy"] ** 2
    )

    df["elapsed_seconds"] = (
        df.groupby("track_id")["time_seconds"].diff()
    )

    df["frame_gap"] = (
        df.groupby("track_id")["frame"].diff()
    )

    print("\nObservation gap distribution:")
    print(df["frame_gap"].value_counts().sort_index())

    df["speed_pixels_per_second"] = (
        df["displacement_pixels"] / df["elapsed_seconds"]
    )

    invalid_gaps = df["frame_gap"] > MAX_FRAME_GAP

    df.loc[
        invalid_gaps,
        ["displacement_pixels", "speed_pixels_per_second"]
    ] = np.nan

    summary = df.groupby("track_id").agg(
        observations=("frame", "count"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        total_movement_pixels=("displacement_pixels", "sum"),
        average_speed_pixels_per_second=(
            "speed_pixels_per_second", "mean"
        ),
        maximum_speed_pixels_per_second=(
            "speed_pixels_per_second", "max"
        ),
    ).reset_index()

    summary["track_duration_seconds"] = (
        summary["last_frame"] - summary["first_frame"]
    ) / FPS

    summary["track_quality"] = np.where(
        (summary["observations"] >= MIN_OBSERVATIONS)
        & (
            summary["track_duration_seconds"]
            >= MIN_DURATION_SECONDS
        ),
        "usable",
        "short_or_fragmented"
    )

    print("\nTrack quality distribution:")
    print(summary["track_quality"].value_counts())

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(OUTPUT_PATH, index=False)

    usable_tracks = summary[
        summary["track_quality"] == "usable"
    ].copy()

    usable_tracks.to_csv(
        USABLE_OUTPUT_PATH,
        index=False
    )

    print(f"\nUsable tracks saved: {USABLE_OUTPUT_PATH}")
    print(f"Usable track count: {len(usable_tracks)}")

    print("\nMovement analysis completed.")
    print(f"Players analysed: {len(summary)}")
    print(f"Saved: {OUTPUT_PATH}")

    print("\nFirst 10 player summaries:")
    print(summary.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
