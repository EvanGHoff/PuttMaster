import cv2

cap = cv2.VideoCapture(1)  # Change 0 to your camera index

# Get max resolution (some cameras report limits, others don't)
max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Max Resolution: {max_width}x{max_height}")

cap.release()