AI Football Player Tracking & Performance Analysis

A computer vision pipeline for detecting and tracking football players in broadcast footage, analysing their movement, and generating structured performance outputs.

Project Overview

This project explores how computer vision can support football performance analysis by extracting player tracking observations from match footage and converting them into movement metrics and visualisations.

The current MVP uses YOLO for player detection and ByteTrack for multi-object tracking. It processes tracking observations to assess track quality, calculate image-plane movement metrics, and generate trajectory and speed visualisations.

Objectives

- Detect players in football broadcast footage.
- Track detected players across sampled video frames.
- Investigate tracking consistency and track fragmentation.
- Calculate movement metrics from player bounding boxes.
- Filter short or fragmented tracks.
- Visualise movement speed and player trajectories.
- Provide a reproducible analysis pipeline.

Technology Stack

- Python 3.12
- Ultralytics YOLO
- ByteTrack
- OpenCV
- NumPy
- Pandas
- Matplotlib
- SciPy
- Pytest

Pipeline Architecture

The analysis pipeline follows these stages:

1. Load existing player tracking observations.
2. Validate tracking data and analyse track quality.
3. Calculate movement metrics.
4. Filter usable tracks.
5. Generate movement-speed visualisations.
6. Generate player trajectory visualisations.
7. Save the resulting analysis files.

Project Structure

ai-football-player-tracking-analysis/
│
├── data/
│   └── football_test.mp4
│
├── models/
│
├── outputs/
│   ├── buffer_60_tracking_observations.csv
│   ├── buffer_60_track_summary.csv
│   ├── player_movement_metrics.csv
│   ├── player_movement_metrics_usable.csv
│   ├── player_movement_speed.png
│   └── player_movement_trajectories.png
│
├── src/
│   ├── analyze_tracking.py
│   ├── movement_analysis.py
│   ├── visualize_movement.py
│   ├── visualize_trajectories.py
│   ├── run_pipeline.py
│   ├── bytetrack_custom.yaml
│   └── other diagnostic scripts
│
├── tests/
│   └── test_movement_analysis.py
│
├── requirements.txt
├── README.md
└── .gitignore

Setup - Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ai-football-player-tracking-analysis


Create and activate a virtual environment:

in powershell:
python -m venv .venv
.venv\Scripts\Activate.ps1


Install dependencies:

in powershell:
pip install -r requirements.txt

Place the required input video and YOLO model in their expected locations:

- Video: `data/football_test.mp4`
- Model: `yolo11n.pt`

The tracking observations required by the analysis pipeline must also be available in the `outputs/` directory.

Running the Pipeline

Run:

python src/run_pipeline.py

The pipeline reuses existing tracking observations. It does not rerun YOLO inference.

Generated Outputs

| Output | Description |
| `buffer_60_track_summary.csv` | Track-level quality and continuity diagnostics |
| `player_movement_metrics.csv` | Movement metrics for all analysed tracks |
| `player_movement_metrics_usable.csv` | Metrics for tracks passing quality thresholds |
| `player_movement_speed.png` | Average image-plane speed visualisation |
| `player_movement_trajectories.png` | Visual representation of tracked player trajectories |

Movement Metrics:

The movement analysis calculates:

- Number of observations per track
- First and last observed frame
- Track duration
- Total image-plane displacement
- Average image-plane speed
- Maximum image-plane speed
- Track quality classification

Player foot positions are approximated using the bottom-centre point of each detection bounding box.

Large observation gaps are excluded from individual movement calculations to reduce misleading displacement and speed estimates.

Tracking Experiment:

The project evaluated different detection and tracking configurations.

The selected experimental configuration uses:

- YOLO11n
- ByteTrack
- Detection confidence threshold: 0.10
- Track buffer: 60
- Sampling rate: 15 frames per second

In the selected 10-second experiment, the system produced 1,170 tracking observations across 150 sampled frames and 36 unique track IDs.

These results are experimental diagnostics, not ground-truth tracking accuracy measurements.

Testing - Run the test suite:

in powershell:
pytest -v

The current test suite covers:

- Foot-position calculation
- Movement displacement calculation
- Exclusion of large observation gaps
- Time calculation using the original video FPS
- Track-quality thresholds

Current Limitations

This MVP has important limitations:

1. Identity consistency: Track IDs can change when players overlap, become occluded, or move through difficult camera views. Identity-switch accuracy has not been verified against ground truth.
2. Pitch calibration: Reliable pitch-to-image calibration has not yet been completed.
3. Physical distance and speed: Movement measurements are currently expressed in pixels and pixels per second. They are not real-world metres or metres per second.
4. Camera movement: Broadcast camera motion can affect image-plane trajectory and speed measurements.
5. Track quality: Filtering reduces the influence of fragmented tracks but does not guarantee that every retained track represents one player throughout its duration.

uture Improvements

- Robust pitch landmark detection and camera calibration.
- Conversion from image coordinates to pitch coordinates.
- More reliable player identity association.
- Improved handling of occlusion and camera transitions.
- Validation against manually annotated ground-truth tracking data.
- Additional football performance metrics.

Project Status

The current MVP implements a reproducible tracking-analysis pipeline with movement metrics, track-quality diagnostics, visualisations, and automated tests.

The system is a research and engineering prototype. Its current image-plane measurements should not be interpreted as validated professional football performance statistics.
