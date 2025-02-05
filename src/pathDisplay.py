import matplotlib.pyplot as plt
import sys
import tkinter as tk

# Disable Matplotlib toolbar
plt.rcParams['toolbar'] = 'None'

# Ensure there are enough command-line arguments
if len(sys.argv) != 5:
    print("Usage: python script.py x1 y1 x2 y2")
    sys.exit(1)

# Convert command-line arguments to floats
point1 = [float(sys.argv[1]), float(sys.argv[2])]
point2 = [float(sys.argv[3]), float(sys.argv[4])]

print("Point 1:", point1)
print("Point 2:", point2)

# Extract x and y coordinates
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))  

# Get the figure manager
fig_manager = plt.get_current_fig_manager()

# Check if TkAgg backend is being used
if hasattr(fig_manager, "window"):
    root = fig_manager.window
    root.overrideredirect(True)  # Remove title bar, minimize/maximize buttons
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")  # Fullscreen

# Plot the line with markers
ax.plot(x_values, y_values, marker='o', linestyle='-')

ax.set_xlim(0, 1440)
ax.set_ylim(0, 720)

# Remove the axis
ax.axis('off')

# Display the plot
plt.show()
