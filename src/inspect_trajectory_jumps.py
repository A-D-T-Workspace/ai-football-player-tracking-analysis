import pandas as pd
import numpy as np

INPUT_PATH = "outputs/buffer_60_tracking_observations.csv"

df = pd.read_csv(INPUT_PATH)

df["foot_x"] = (df["x1"] + df["x2"]) / 2
df["foot_y"] = df["y2"]

df = df.sort_values(["track_id", "frame"])

df["frame_gap"] = df.groupby("track_id")["frame"].diff()

df["dx"] = df.groupby("track_id")["foot_x"].diff()
df["dy"] = df.groupby("track_id")["foot_y"].diff()

df["displacement_pixels"] = np.sqrt(
    df["dx"] ** 2 + df["dy"] ** 2
)

df["elapsed_seconds"] = df["frame_gap"] / 30

df["speed_pixels_per_second"] = (
    df["displacement_pixels"] / df["elapsed_seconds"]
)

jumps = df[
    (df["frame_gap"] <= 4)
    & (df["frame_gap"] > 0)
].sort_values(
    "displacement_pixels",
    ascending=False
)

print("\nTop 15 largest consecutive movements:\n")

print(
    jumps[
        [
            "track_id",
            "frame",
            "frame_gap",
            "displacement_pixels",
            "speed_pixels_per_second"
        ]
    ].head(15).to_string(index=False)
)

track_25 = df[df["track_id"] == 25].copy()

print("\nTrack 25 movement transitions:\n")

print(
    track_25[
        [
            "frame",
            "frame_gap",
            "foot_x",
            "foot_y",
            "displacement_pixels",
            "speed_pixels_per_second"
        ]
    ].to_string(index=False)
)