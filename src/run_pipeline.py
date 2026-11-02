from pathlib import Path

from analyze_tracking import main as analyze_tracking
from movement_analysis import main as analyze_movement
from visualize_movement import main as visualize_movement
from visualize_trajectories import main as visualize_trajectories

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "buffer_60_tracking_observations.csv"
)

def main():
    print("\n" + "=" * 60)
    print("AI FOOTBALL PLAYER TRACKING & PERFORMANCE ANALYSIS")
    print("=" * 60)

    if not REQUIRED_INPUT.exists():
        raise FileNotFoundError(
            f"Required tracking data not found: {REQUIRED_INPUT}"
        )

    print("\n[1/4] Running tracking quality analysis...")
    analyze_tracking()

    print("\n[2/4] Running player movement analysis...")
    analyze_movement()

    print("\n[3/4] Creating movement speed visualization...")
    visualize_movement()

    print("\n[4/4] Creating player trajectory visualization...")
    visualize_trajectories()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nFinal outputs:")

    outputs = [
        "buffer_60_track_summary.csv",
        "player_movement_metrics.csv",
        "player_movement_metrics_usable.csv",
        "player_movement_speed.png",
        "player_movement_trajectories.png",
    ]

    for filename in outputs:
        path = PROJECT_ROOT / "outputs" / filename

        status = "FOUND" if path.exists() else "MISSING"

        print(f"{filename}: {status}")

if __name__ == "__main__":
    main()
