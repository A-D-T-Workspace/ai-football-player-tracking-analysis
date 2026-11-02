import cv2
import pandas as pd
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

VIDEO_PATH = Path("data/football_test.mp4")
MODEL_PATH = "yolo11n.pt"

EXPERIMENT_ID = "buffer_60"
CONFIDENCE_THRESHOLD = 0.1
TRACKER = "src/bytetrack_custom.yaml"
VIDEO_SECONDS = 10
SAMPLING_FPS = 15

OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / f"{EXPERIMENT_ID}_tracking_preview.mp4"
OBSERVATIONS_PATH = OUTPUT_DIR / f"{EXPERIMENT_ID}_tracking_observations.csv"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps / 2,
    (width, height),
)

frames_to_process = int(fps * 10)
sampled_frames = 0
all_track_ids = set()
previous_ids = set()
visible_id_set_changes = 0
new_ids_per_frame = []
track_lifetimes = Counter()
track_observations = []

for frame_index in range(frames_to_process):
    ret, frame = cap.read()

    if frame_index < 5:
        print(f"Frame {frame_index}: ret={ret}")

    if not ret:
        break

    if frame_index % 2 != 0:
        continue

    result = model.track(
        frame,
        persist=True,
        tracker=TRACKER,
        classes=[0],
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]
    
    current_ids = set()

    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for box, track_id, confidence in zip(
            boxes, track_ids, confidences
        ):
            track_id = int(track_id)
            current_ids.add(track_id)

            track_observations.append({
                "frame": frame_index,
                "track_id": track_id,
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "confidence": confidence,
            })

    all_track_ids.update(current_ids)
    track_lifetimes.update(current_ids)

    new_ids = current_ids - previous_ids
    new_ids_per_frame.append(len(new_ids))

    if sampled_frames > 0 and current_ids != previous_ids:
        visible_id_set_changes += 1

    previous_ids = current_ids

    writer.write(result.plot())
    sampled_frames += 1

cap.release()
writer.release()

observations_path = OBSERVATIONS_PATH
pd.DataFrame(track_observations).to_csv(
    observations_path,
    index=False,
)

print(f"Tracking observations saved to: {observations_path}")
print(f"Sampled frames processed: {sampled_frames}")
print(f"Unique track IDs observed: {len(all_track_ids)}")
print(f"Visible ID-set changes: {visible_id_set_changes}")
print(f"Tracking preview saved to: {OUTPUT_PATH}")

if new_ids_per_frame:
    print(f"Total new-ID appearances: {sum(new_ids_per_frame)}")
    print(
        "Frames with at least one new ID: "
        f"{sum(1 for count in new_ids_per_frame if count > 0)}"
    )
    print(f"Maximum new IDs in one frame: {max(new_ids_per_frame)}")

if track_lifetimes:
    lifetimes = list(track_lifetimes.values())
    print(f"Tracks observed: {len(lifetimes)}")
    print(
        "Tracks lasting 1-3 sampled frames: "
        f"{sum(1 for n in lifetimes if n <= 3)}"
    )
    print(
        "Tracks lasting 10+ sampled frames: "
        f"{sum(1 for n in lifetimes if n >= 10)}"
    )
    print(f"Longest track: {max(lifetimes)} sampled frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

observations_count = len(track_observations)
short_tracks = sum(1 for n in track_lifetimes.values() if 1 <= n <= 3)
long_tracks = sum(1 for n in track_lifetimes.values() if n >= 10)
longest_track = max(track_lifetimes.values(), default=0)

log_row = {
    "experiment_id": EXPERIMENT_ID,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "tracker": TRACKER,
    "video_seconds": VIDEO_SECONDS,
    "sampling_fps": SAMPLING_FPS,
    "sampled_frames": sampled_frames,
    "observations": observations_count,
    "unique_track_ids": len(all_track_ids),
    "visible_id_set_changes": visible_id_set_changes,
    "new_id_appearances": sum(new_ids_per_frame),
    "short_tracks_1_3": short_tracks,
    "long_tracks_10_plus": long_tracks,
    "longest_track_frames": longest_track,
    "model": MODEL_PATH,
    "sampling_interval": 2,
}

log_path = OUTPUT_DIR / "experiment_log.csv"
log_df = pd.DataFrame([log_row])

if log_path.exists():
    existing_log = pd.read_csv(log_path)
    existing_log = existing_log[
        existing_log["experiment_id"] != EXPERIMENT_ID
    ]
    log_df = pd.concat([existing_log, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False)
print(f"Experiment metrics logged to: {log_path}")
