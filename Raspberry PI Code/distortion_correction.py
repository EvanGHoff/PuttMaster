# Old version. Using projector_distortion.py instead.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Display four red dots using matplotlib
def display_red_dots():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('black')
    ax.scatter([0, 1, 0, 1], [0, 0, 1, 1], color='red', s=500)  # Four corner points
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    plt.show()

# Detect red dots from the camera feed
def detect_red_dots(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    print("Press 'q' when the dots are correctly detected.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red color range
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:  # Filter small detections
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers.append([cX, cY])
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

        cv2.imshow('Camera Feed - Detecting Red Dots', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') and len(centers) == 4:
            print(f"Detected points: {centers}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return np.float32(sorted(centers, key=lambda x: x[1]))  # Sort by Y-axis for consistency

# Compute the perspective transform and apply it
def apply_perspective_correction(detected_points, width=800, height=600):
    # Desired corner points in the corrected projection
    target_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Compute the transformation matrix
    matrix = cv2.getPerspectiveTransform(detected_points, target_points)
    return matrix

# Example usage for correcting an image
def correct_projection(matrix, image_path='plot.png', width=800, height=600):
    image = cv2.imread(image_path)
    corrected = cv2.warpPerspective(image, matrix, (width, height))
    cv2.imshow("Corrected Projection", corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    display_red_dots()  # Project red dots
    detected_points = detect_red_dots()  # Detect them via camera
    transform_matrix = apply_perspective_correction(detected_points)  # Compute mapping
    correct_projection(transform_matrix)  # Apply correction to a sample image
