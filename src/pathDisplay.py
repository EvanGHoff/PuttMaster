import matplotlib.pyplot as plt
import sys

# Ensure there are enough command-line arguments
if len(sys.argv) != 5:
    print("Usage: python script.py x1 y1 x2 y2")
    sys.exit(1)

plt.rcParams['toolbar'] = 'None'

# Convert command-line arguments to floats
point1 = [float(sys.argv[1]), float(sys.argv[2])]
point2 = [float(sys.argv[3]), float(sys.argv[4])]

print("Point 1:", point1)
print("Point 2:", point2)

# Extract x and y coordinates
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figsize for better scaling

# Maximize the window for different backends
fig_manager = plt.get_current_fig_manager()
try:
    fig_manager.window.state('zoomed')  # Works on Windows with TkAgg backend
except AttributeError:
    try:
        fig_manager.full_screen_toggle()  # Works on some Linux/MacOS setups
    except AttributeError:
        print("Fullscreen mode not supported on this backend.")

# Plot the line with markers
ax.plot(x_values, y_values, marker='o', linestyle='-')

# Set labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_title('Line between Two Points')

# Set axis limits
ax.set_xlim(0, 1440)
ax.set_ylim(0, 720)

ax.grid(False)
ax.axis('off')  

# Display the plot
plt.show()
