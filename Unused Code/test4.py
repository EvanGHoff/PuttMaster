import cv2
import numpy as np

file = "C:\Users\zzhzz\OneDrive\Pictures\Camera Roll\WIN_20250331_15_44_00_Pro.jpg"

# Load the image (grayscale or color)
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)  # Use IMREAD_COLOR for RGB

# Define circle parameters
x, y, radius = 100, 150, 50  # Example coordinates and radius

# Create a mask
mask = np.zeros_like(image, dtype=np.uint8)
cv2.circle(mask, (x, y), radius, 255, thickness=-1)  # White filled circle

# Compute mean using the mask
mean_value = cv2.mean(image, mask=mask)[0]  # The first value is the mean intensity

print(f"Mean pixel value within the circle: {mean_value}")