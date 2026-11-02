import cv2
from pathlib import Path

image_path = Path("outputs/calibration_frame_30s.jpg")

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Could not load calibration frame.")
else:
    height, width = image.shape[:2]

    print("Calibration frame loaded successfully.")
    print(f"Width: {width}")
    print(f"Height: {height}")