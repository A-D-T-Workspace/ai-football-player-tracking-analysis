import cv2
import numpy as np
from pathlib import Path

video_path = "data/football_test.mp4"
output_path = "outputs/football_transition_comparison.jpg"

times = [14, 16, 18, 20, 22, 26, 28, 30, 32, 36, 38, 40, 42]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

tiles = []

for seconds in times:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.resize(frame, (320, 180))

    cv2.putText(
        frame,
        f"{seconds}s",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    tiles.append(frame)

cap.release()

blank = np.zeros((180, 320, 3), dtype=np.uint8)

while len(tiles) % 4 != 0:
    tiles.append(blank.copy())

rows = []

for i in range(0, len(tiles), 4):
    row = np.hstack(tiles[i:i + 4])
    rows.append(row)

sheet = np.vstack(rows)

Path("outputs").mkdir(exist_ok=True)
cv2.imwrite(output_path, sheet)

print(f"Comparison sheet saved to: {output_path}")
