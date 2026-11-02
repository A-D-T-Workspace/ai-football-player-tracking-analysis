import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = "data/football_test.mp4"
OUTPUT_PATH = "outputs/pitch_intersections.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

cap.set(cv2.CAP_PROP_POS_MSEC, 30000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read the 30-second frame.")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=35,
    minLineLength=40,
    maxLineGap=20,
)

def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]

def intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    p = np.array([x1, y1], dtype=float)
    r = np.array([x2 - x1, y2 - y1], dtype=float)

    q = np.array([x3, y3], dtype=float)
    s = np.array([x4 - x3, y4 - y3], dtype=float)

    denominator = cross(r, s)

    if abs(denominator) < 1e-6:
        return None

    t = cross(q - p, s) / denominator
    u = cross(q - p, r) / denominator

    if -0.10 <= t <= 1.10 and -0.10 <= u <= 1.10:
        point = p + t * r
        return point

    return None

long_lines = []

if lines is not None:
    for line in lines:
        coords = line.reshape(-1)
        x1, y1, x2, y2 = coords

        length = np.hypot(x2 - x1, y2 - y1)

        if length >= 80:
            long_lines.append(coords)

candidates = []

for i in range(len(long_lines)):
    for j in range(i + 1, len(long_lines)):
        point = intersection(long_lines[i], long_lines[j])

        if point is not None:
            x, y = point

            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                candidates.append(point)

unique_points = []

for point in candidates:
    if not any(np.linalg.norm(point - existing) < 12
               for existing in unique_points):
        unique_points.append(point)

for line in long_lines:
    x1, y1, x2, y2 = line
    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

for point in unique_points:
    x, y = np.round(point).astype(int)
    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

Path("outputs").mkdir(exist_ok=True)
cv2.imwrite(OUTPUT_PATH, frame)

print(f"Long line segments: {len(long_lines)}")
print(f"Raw intersection candidates: {len(candidates)}")
print(f"Unique candidate intersections: {len(unique_points)}")

if unique_points:
    points_array = np.array(unique_points)

    print(f"Average X coordinate: {points_array[:, 0].mean():.1f}")
    print(f"Average Y coordinate: {points_array[:, 1].mean():.1f}")

    print(
        "Points inside central image region:",
        sum(
            0.2 * frame.shape[1] < x < 0.8 * frame.shape[1]
            and 0.2 * frame.shape[0] < y < 0.8 * frame.shape[0]
            for x, y in unique_points
        )
    )

    print(
    "Points inside central image region:",
    sum(
        0.2 * frame.shape[1] < x < 0.8 * frame.shape[1]
        and 0.2 * frame.shape[0] < y < 0.8 * frame.shape[0]
        for x, y in unique_points
        )
    )

    if unique_points:
        xs = [point[0] for point in unique_points]
        ys = [point[1] for point in unique_points]

        print(f"Minimum X: {min(xs):.1f}")
        print(f"Maximum X: {max(xs):.1f}")
        print(f"Minimum Y: {min(ys):.1f}")
        print(f"Maximum Y: {max(ys):.1f}")

print(f"Saved: {OUTPUT_PATH}")
