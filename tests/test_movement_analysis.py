import numpy as np
import pandas as pd


def create_sample_observations():
    """Create controlled tracking observations for testing."""
    return pd.DataFrame({
        "frame": [0, 2, 4, 0, 2, 10],
        "track_id": [1, 1, 1, 2, 2, 2],
        "x1": [10, 12, 14, 30, 32, 40],
        "y1": [10, 10, 10, 20, 20, 20],
        "x2": [20, 22, 24, 40, 42, 50],
        "y2": [30, 30, 30, 40, 40, 40],
        "confidence": [0.9, 0.9, 0.9, 0.8, 0.8, 0.8],
    })


def test_foot_position_calculation():
    data = create_sample_observations()

    data["foot_x"] = (data["x1"] + data["x2"]) / 2
    data["foot_y"] = data["y2"]

    assert data.loc[0, "foot_x"] == 15
    assert data.loc[0, "foot_y"] == 30


def test_movement_displacement_calculation():
    data = create_sample_observations()
    data = data.sort_values(["track_id", "frame"]).copy()

    data["foot_x"] = (data["x1"] + data["x2"]) / 2
    data["foot_y"] = data["y2"]

    dx = data.groupby("track_id")["foot_x"].diff()
    dy = data.groupby("track_id")["foot_y"].diff()

    displacement = np.sqrt(dx**2 + dy**2)

    assert displacement.iloc[1] == 2
    assert displacement.iloc[2] == 2


def test_large_observation_gaps_are_excluded():
    data = create_sample_observations()
    data = data.sort_values(["track_id", "frame"]).copy()

    data["frame_gap"] = data.groupby("track_id")["frame"].diff()

    max_frame_gap = 4
    invalid_gaps = data["frame_gap"] > max_frame_gap

    assert invalid_gaps.sum() == 1
    assert data.loc[invalid_gaps, "frame_gap"].iloc[0] == 8


def test_elapsed_time_uses_original_video_fps():
    data = create_sample_observations()
    data = data.sort_values(["track_id", "frame"]).copy()

    data["time_seconds"] = data["frame"] / 30
    elapsed = data.groupby("track_id")["time_seconds"].diff()

    assert np.isclose(elapsed.iloc[1], 2 / 30)
    assert np.isclose(elapsed.iloc[2], 2 / 30)


def test_track_quality_thresholds():
    summary = pd.DataFrame({
        "observations": [15, 10, 20],
        "track_duration_seconds": [1.0, 2.0, 0.5],
    })

    summary["track_quality"] = np.where(
        (summary["observations"] >= 15)
        & (summary["track_duration_seconds"] >= 1.0),
        "usable",
        "short_or_fragmented",
    )

    assert summary.loc[0, "track_quality"] == "usable"
    assert summary.loc[1, "track_quality"] == "short_or_fragmented"
    assert summary.loc[2, "track_quality"] == "short_or_fragmented"
