import cv2
import numpy as np

def detect_green_rectangle():
    """
    Detects a specific green rectangular shape, applies perspective transformation,
    and extracts the corrected shape region.
    """
    cap = cv2.VideoCapture(0)

    shape_corners = None  # Store the detected corners

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        # Convert to HSV and create a mask for green color
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, green_lower, green_upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_corners = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Ignore small noise
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

                if len(approx) == 4:  # We want exactly 4 corners (rectangle)
                    if area > max_area:  # Pick the largest valid rectangle
                        max_area = area
                        best_corners = np.array([point[0] for point in approx], dtype=np.float32)
                        #print(best_corners)

        if best_corners is not None:
            shape_corners = best_corners

            # Draw the detected rectangle
            cv2.drawContours(frame, [best_corners.astype(int)], -1, (0, 255, 0), 3)
            for (x, y) in shape_corners.astype(int):
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw red circles at corners

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return the detected rectangle's corners if found
    if shape_corners is not None and len(shape_corners) == 4:
        return shape_corners
    else:
        print("No valid rectangle detected.")
        return None

green_lower = (60, 150, 20)
green_upper = (95, 255, 150)
detect_green_rectangle()