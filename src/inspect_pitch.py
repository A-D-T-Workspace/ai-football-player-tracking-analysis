import cv2
import numpy as np

VIDEO_PATH = "data/football_test.mp4"
OUTPUT_PATH = "outputs/pitch_lines.jpg"
FRAME_OUTPUT_PATH = "outputs/calibration_frame_30s.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, 30000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read calibration frame.")

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

line_count = 0

long_lines = 0
short_lines = 0

long_horizontal = 0
long_vertical = 0

if lines is not None:
    line_count = len(lines)

    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

horizontal = 0
vertical = 0
diagonal = 0

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if length >= 80:
            long_lines += 1

            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if angle < 15 or angle > 165:
                long_horizontal += 1
            elif 75 < angle < 105:
                long_vertical += 1
        else:
            short_lines += 1

        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if angle < 15 or angle > 165:
            horizontal += 1
        elif 75 < angle < 105:
            vertical += 1
        else:
            diagonal += 1

print(f"Long line segments (80+ pixels): {long_lines}")
print(f"Short line segments: {short_lines}")

print(f"Long horizontal segments: {long_horizontal}")
print(f"Long vertical segments: {long_vertical}")

print(f"Horizontal segments: {horizontal}")
print(f"Vertical segments: {vertical}")
print(f"Diagonal segments: {diagonal}")

cv2.imwrite(FRAME_OUTPUT_PATH, frame)
cv2.imwrite(OUTPUT_PATH, frame)

print("Pitch line detection complete.")
print(f"Detected line segments: {line_count}")
print(f"Saved: {OUTPUT_PATH}")
print(f"Saved: {FRAME_OUTPUT_PATH}")
