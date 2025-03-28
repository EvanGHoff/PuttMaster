# Phased out. Different code is used for path display.

import matplotlib.pyplot as plt
import sys
import time

# Disable Matplotlib toolbar
plt.rcParams['toolbar'] = 'None'

# Ensure a file is provided as an argument
if len(sys.argv) != 2:
    print("Usage: python script.py input_file.txt")
    sys.exit(1)

input_file = sys.argv[1]

# Create a figure and axis with green background
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor([0, 0.1, 0])  # Set figure background color
ax.set_facecolor('green')         # Set axis background color

# Set plot limits and remove axes
ax.set_xlim(0, 1440)
ax.set_ylim(0, 720)
ax.axis('off')

# Fullscreen functionality
fig_manager = plt.get_current_fig_manager()
if hasattr(fig_manager, "window"):
    root = fig_manager.window
    root.overrideredirect(True)  # Remove title bar, minimize/maximize buttons
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")  # Fullscreen

# Read the input file and update the line dynamically
with open(input_file, 'r') as f:
    for line in f:
        values = line.strip().split()
        if len(values) != 4:
            print(f"Skipping invalid line: {line.strip()}")
            continue
        try:
            x1, y1, x2, y2 = map(float, values)

            # Clear previous line
            ax.clear()
            ax.set_facecolor('black')  # Ensure background stays green
            ax.set_xlim(0, 1440)
            ax.set_ylim(0, 720)
            ax.axis('off')

            # Plot the new line in red with increased thickness
            ax.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color='white', linewidth=20)
            ax.plot([x1-100, x2-100], [y1-100, y2-100], marker='o', linestyle='-', color='yellow', linewidth=20)

            # Update the figure
            plt.draw()
            plt.pause(.01)  # Pause for 2 seconds before updating the next line

        except ValueError:
            print(f"Skipping invalid line (non-numeric values): {line.strip()}")

# Keep the last line displayed
plt.show()