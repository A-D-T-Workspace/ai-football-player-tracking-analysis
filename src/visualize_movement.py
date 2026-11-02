import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("outputs/player_movement_metrics_usable.csv")
OUTPUT_PATH = Path("outputs/player_movement_speed.png")

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Movement metrics not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        raise ValueError("No usable tracks available to visualize.")

    required_columns = [
        "track_id",
        "average_speed_pixels_per_second",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values(
        "average_speed_pixels_per_second",
        ascending=False,
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        df["track_id"].astype(str),
        df["average_speed_pixels_per_second"],
    )

    plt.xlabel("Track ID")
    plt.ylabel("Average image-plane speed (pixels/second)")
    plt.title("Average Movement Speed by Usable Track")

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print("Movement visualization created.")
    print(f"Tracks visualized: {len(df)}")
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
