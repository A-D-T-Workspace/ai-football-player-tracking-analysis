import cv2
import numpy as np
import csv
from pathlib import Path

VIDEO_PATH = "data/football_test.mp4"
OUTPUT_PATH = Path("outputs/camera_diagnostic.csv")

SAMPLE_INTERVAL_SECONDS = 2
INSPECT_SECONDS = 60

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

orb = cv2.ORB_create(nfeatures=1500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

previous_descriptors = None
previous_keypoints = None

results = []

for seconds in range(
    0, INSPECT_SECONDS + 1, SAMPLE_INTERVAL_SECONDS
):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    success, frame = cap.read()

    if not success:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    matches_count = 0
    inliers = 0
    inlier_ratio = 0.0

    if previous_descriptors is not None and descriptors is not None:
        pairs = matcher.knnMatch(
            previous_descriptors, descriptors, k=2
        )

        good = [
            first for first, second in pairs
            if first.distance < 0.75 * second.distance
        ]

        matches_count = len(good)

        if matches_count >= 8:
            source = np.float32([
                previous_keypoints[m.queryIdx].pt for m in good
            ]).reshape(-1, 1, 2)

            target = np.float32([
                keypoints[m.trainIdx].pt for m in good
            ]).reshape(-1, 1, 2)

            _, mask = cv2.findHomography(
                source, target, cv2.RANSAC, 5.0
            )

            if mask is not None:
                inliers = int(mask.sum())
                inlier_ratio = inliers / matches_count

    results.append({
        "timestamp_seconds": seconds,
        "matches": matches_count,
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 3),
    })

    print(
        f"{seconds}s | Matches: {matches_count} | "
        f"Inliers: {inliers} | Ratio: {inlier_ratio:.2f}"
    )

    previous_keypoints = keypoints
    previous_descriptors = descriptors

cap.release()

with open(OUTPUT_PATH, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "timestamp_seconds",
            "matches",
            "inliers",
            "inlier_ratio",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nDiagnostic saved to: {OUTPUT_PATH}")
