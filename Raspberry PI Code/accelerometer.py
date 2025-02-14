import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz

# def manual_cumtrapz(y, x):
#     return np.concatenate(([0], np.cumsum((y[:-1] + y[1:]) * np.diff(x) / 2)))

# Example accelerometer data (m/s²) and timestamps (s)
accel_x = np.array([0, 1.2, 2.5, 3.8, 5.1, 6.3, 7.0]) 
timestamps = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 

# Integrate acceleration to get velocity
velocity = cumtrapz(accel_x, timestamps, initial=0)

# Find peak velocity (swing speed)
swing_speed = max(velocity)
print(f"Peak Swing Speed: {swing_speed:.2f} m/s")
