import numpy as np
from scipy.integrate import cumulative_trapezoid

# Example gyroscope data (angular velocity in degrees/second)
gyro_z = np.array([0, 5, 10, 15, 20, 15, 10, 5, 0])  # Rotation around Z-axis
timestamps = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])  # Time in seconds

# Convert degrees/s to radians/s
gyro_z_rad = np.radians(gyro_z)

# Integrate angular velocity to get orientation (angle)
angle_z = np.degrees(cumulative_trapezoid(gyro_z_rad, timestamps, initial=0))  # Convert back to degrees

# Display facing angle over time
for t, ang in zip(timestamps, angle_z):
    print(f"Time: {t:.2f}s, Facing Angle: {ang:.2f}°")

# Get final facing angle
final_angle = angle_z[-1]
print(f"Final Facing Angle: {final_angle:.2f}°")
