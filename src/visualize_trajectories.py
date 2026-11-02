import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OBSERVATIONS_PATH = Path(
    "outputs/buffer_60_tracking_observations.csv"
)

METRICS_PATH = Path(
    "outputs/player_movement_metrics_usable.csv"
)

OUTPUT_PATH = Path(
    "outputs/player_movement_trajectories.png"
)

MAX_FRAME_GAP = 4

def main():
    for path in [OBSERVATIONS_PATH, METRICS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    observations = pd.read_csv(OBSERVATIONS_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    required_observation_columns = [
        "frame", "track_id", "x1", "y1", "x2", "y2"
    ]

    for column in required_observation_columns:
        if column not in observations.columns:
            raise ValueError(
                f"Missing observation column: {column}"
            )

    if "track_id" not in metrics.columns:
        raise ValueError(
            "Movement metrics must contain track_id."
        )

    if metrics.empty:
        raise ValueError("No usable tracks available.")

    usable_ids = metrics["track_id"].tolist()

    df = observations[
        observations["track_id"].isin(usable_ids)
    ].copy()

    if df.empty:
        raise ValueError(
            "No observations found for usable tracks."
        )

    df["foot_x"] = (df["x1"] + df["x2"]) / 2
    df["foot_y"] = df["y2"]

    df = df.sort_values(["track_id", "frame"])

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.get_cmap("tab20")

    for i, (track_id, track) in enumerate(
        df.groupby("track_id")
    ):
        color = colors(i % 20)

        track = track.copy()
        track["frame_gap"] = track["frame"].diff()

        track["segment"] = (
            track["frame_gap"] > MAX_FRAME_GAP
        ).cumsum()

        first_segment = True

        for _, segment in track.groupby("segment"):
            ax.plot(
                segment["foot_x"],
                segment["foot_y"],
                marker=".",
                markersize=3,
                linewidth=1.5,
                label=(
                    f"Track {track_id}"
                    if first_segment else None
                ),
                color=color,
            )

            first_segment = False

        ax.scatter(
            track["foot_x"].iloc[0],
            track["foot_y"].iloc[0],
            marker="o",
            s=35,
            color=color,
        )

    ax.set_xlim(0, 640)
    ax.set_ylim(360, 0)

    ax.set_xlabel("Horizontal image position (pixels)")
    ax.set_ylabel("Vertical image position (pixels)")
    ax.set_title("Player Movement Trajectories")

    ax.legend(
        title="Track IDs",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print("Trajectory visualization completed.")
    print(f"Tracks visualized: {df['track_id'].nunique()}")
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
