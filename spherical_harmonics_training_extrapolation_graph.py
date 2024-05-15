import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# Replace this with your desired number of points
num_points = 100

# Generate evenly spaced values for phi and theta
phi_flat = np.linspace(0, 2 * np.pi, num_points)
theta_flat = np.linspace(0, np.pi, num_points)

# Create a grid of phi and theta
phi, theta = np.meshgrid(phi_flat, theta_flat)

# Convert spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Identify the indices of the bottom third
bottom_third_indices = theta >= 2 * np.pi / 3
upper_half_indices = theta <= np.pi / 2
middle_indices = ~(bottom_third_indices | upper_half_indices)
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_upper_half = x[upper_half_indices].flatten()
y_upper_half = y[upper_half_indices].flatten()
z_upper_half = z[upper_half_indices].flatten()
# Plot the surface in red
extrapolation_handle = ax.plot_trisurf(x_upper_half, y_upper_half, z_upper_half, color='red', alpha=0.5)

# Flatten the arrays for the bottom third
x_bottom_third = x[bottom_third_indices].flatten()
y_bottom_third = y[bottom_third_indices].flatten()
z_bottom_third = z[bottom_third_indices].flatten()

# Color the bottom third in blue
training_handle = ax.plot_trisurf(x_bottom_third, y_bottom_third, z_bottom_third, color='orange', alpha=0.9)

# Flatten the arrays for the bottom third
x_middle = x[middle_indices].flatten()
y_middle = y[middle_indices].flatten()
z_middle = z[middle_indices].flatten()

# Color the bottom third in blue
gray_handle = ax.plot_trisurf(x_middle, y_middle, z_middle, color='gray', alpha=0.2)

legend_handles = [
    Line2D([0], [0], marker='o', markersize=8, color='orange', linestyle='', label='Training NExT'),
    Line2D([0], [0], marker='o', markersize=8, color='red', linestyle='', label='Ext NExT')
]

# Adding the legend to the axis
legend = ax.legend(handles=legend_handles, loc='upper left', shadow=True, fontsize='medium')


ax.view_init(elev=15, azim=130)
tick_size = 13.5
plt.rc('xtick', labelsize=tick_size)
plt.rc('ytick', labelsize=tick_size)
plt.rc('axes', labelsize=tick_size, titlesize=tick_size)
plt.rc('axes', titlesize=tick_size, labelsize=tick_size)
# Show the plot
plt.show()
a=1
