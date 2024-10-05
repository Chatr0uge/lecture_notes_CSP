import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import ConvexHull

# Define necessary functions for the time propagation and plotting
def T_HO(m, k, t):
    omega = np.sqrt(k/m)
    return np.array([[np.cos(omega*t), np.sin(omega*t)/omega],
                     [-np.sin(omega*t)*omega, np.cos(omega*t)]])

def T_HO_Euler(m, k, t):
    omega = np.sqrt(k/m)
    return np.array([[1.0, t],
                     [-omega**2*t, 1.0]])

def T_HO_VelocityVerlet(m, k, t):
    omega = np.sqrt(k/m)
    return np.array([[1.0 - 0.5*omega**2*t**2, t],
                     [-omega**2*t * (1.0 - 0.25*omega**2*t**2), 1.0 - 0.5*omega**2*t**2]])

def MappedGridInPhaseSpacePlot(ax, Tprop, label, color, NLines=5, xrange=(-5, 5), vxrange=(-5, 5), plot_type='lines'):
    # Plot horizontal lines (constant vx, varying x)
    for i in range(-NLines, NLines + 1):
        x = np.linspace(xrange[0], xrange[1], 100)  # More points for smoother curves
        vx = i * np.ones_like(x)
        x, vx = np.dot(Tprop, [x, vx])  # Apply transformation
        ax.plot(x, vx, label=label if i == 0 else "", color=color)  # Label for one line only

    # Plot vertical lines (constant x, varying vx)
    for i in range(-NLines, NLines + 1):
        vx = np.linspace(vxrange[0], vxrange[1], 100)  # More points for smoother curves
        x = i * np.ones_like(vx)
        x, vx = np.dot(Tprop, [x, vx])  # Apply transformation
        ax.plot(x, vx, color=color)  # No label for these

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$v_x$')

# Function to compute areas and orientations
def compute_area_and_orientation(ax, NLines, xrange, vxrange, Tprop_exact, Tprop_euler, Tprop_verlet):
    # Calculate the area enclosed by the methods
    
    area_exact =     np.linalg.det(Tprop_exact)
    area_euler =     np.linalg.det(Tprop_euler) 
    area_verlet =  np.linalg.det(Tprop_verlet)

    print("====Tprop====")
    print(f'area_exact : {Tprop_exact}')
    print(f'area_euler : {Tprop_euler}')
    print(f'area_verlet : {Tprop_verlet}')
    print("====Area====")
    print(f'area_exact : {area_exact}')
    print(f'area_euler : {area_euler}')
    print(f'area_verlet : {area_verlet}')

    area_ratio_euler_exact = area_euler - area_exact if area_exact != 0 else 0
    area_ratio_verlet_exact = area_verlet - area_exact if area_exact != 0 else 0
    # Calculate bounding box area
    width = xrange[1] - xrange[0]
    height = vxrange[1] - vxrange[0]
    area_of_square = width * height
    # Print the bounding box area
    print(f"Area of the bounding box: {area_of_square}")

    return area_ratio_euler_exact, area_ratio_verlet_exact
# Function to compute orientation differences
def compute_orientation(ax, Tprop_exact, Tprop_euler, Tprop_verlet):
    # Getting slopes for orientation
    x_exact = np.linspace(-3, 3, 100)
    vx_exact = np.dot(Tprop_exact, [x_exact, np.zeros_like(x_exact)])[1]
    x_euler = np.linspace(-3, 3, 100)
    vx_euler = np.dot(Tprop_euler, [x_euler, np.zeros_like(x_euler)])[1]
    x_verlet = np.linspace(-3, 3, 100)
    vx_verlet = np.dot(Tprop_verlet, [x_verlet, np.zeros_like(x_verlet)])[1]

    # Calculate angles in degrees
    angle_exact = np.arctan(np.gradient(vx_exact, x_exact)) * 180 / np.pi
    angle_euler = np.arctan(np.gradient(vx_euler, x_euler)) * 180 / np.pi
    angle_verlet = np.arctan(np.gradient(vx_verlet, x_verlet)) * 180 / np.pi

    return angle_exact, angle_euler, angle_verlet

# Function to update the plot with all methods in one plot
def UpdateHOPhaseSpaceMapping(ax, omegaXdelta_t, TimeStep, xrange, vxrange):
    m = 1
    k = 1
    omega = np.sqrt(k / m)
    NLines = 3

    # Clear the plot for new data
    ax.clear()

    # Plot Euler
    Tprop_euler = matrix_power(T_HO_Euler(m, k, omegaXdelta_t / omega), TimeStep)
    MappedGridInPhaseSpacePlot(ax, Tprop_euler, label="Euler", color='blue', NLines=NLines, xrange=xrange, vxrange=vxrange)

    # Plot Exact
    Tprop_exact = matrix_power(T_HO(m, k, omegaXdelta_t / omega), TimeStep)
    MappedGridInPhaseSpacePlot(ax, Tprop_exact, label=f"Exact (ω*t = {round(TimeStep * omegaXdelta_t, 2)})", color='green', NLines=NLines, xrange=xrange, vxrange=vxrange)

    # Plot Velocity Verlet
    Tprop_verlet = matrix_power(T_HO_VelocityVerlet(m, k, omegaXdelta_t / omega), TimeStep)
    MappedGridInPhaseSpacePlot(ax, Tprop_verlet, label="Velocity Verlet", color='red', NLines=NLines, xrange=xrange, vxrange=vxrange)

    ax.legend(loc="upper right")  # Add legend to the plot
    ax.set_aspect('equal')

    # Calculate areas and orientation differences
    area_ratio_euler_exact, area_ratio_verlet_exact = compute_area_and_orientation(ax, NLines, xrange, vxrange, Tprop_exact, Tprop_euler, Tprop_verlet)
    
    return area_ratio_euler_exact, area_ratio_verlet_exact

# Set up the figure and axis
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 2, 1]})

# Choose the time step, i.e., into how many steps you want to subdivide the oscillation period
NPlots = 10
omegaXdelta_t = 2 * np.pi / NPlots

# Calculate the axis limits based on the final frame
final_xrange = (-3, 3)
final_vxrange = (-3, 3)

# Create lists to store area ratios and angles for plotting
area_ratios_euler_exact = []
area_ratios_verlet_exact = []
angles_euler_exact = []
angles_verlet_exact = []

# Animation function that gets called sequentially
def animate(i):
    m = 1
    k = 1
    omega = np.sqrt(k / m)
    area_ratio_euler_exact, area_ratio_verlet_exact = UpdateHOPhaseSpaceMapping(ax1, omegaXdelta_t, i, final_xrange, final_vxrange)
    
    # Store area ratios
    area_ratios_euler_exact.append(area_ratio_euler_exact)
    area_ratios_verlet_exact.append(area_ratio_verlet_exact)
    
    # Compute orientations
    angle_exact, angle_euler, angle_verlet = compute_orientation(ax2, matrix_power(T_HO(m, k, omegaXdelta_t / omega), i), 
                                                                 matrix_power(T_HO_Euler(m, k, omegaXdelta_t / omega), i), 
                                                                 matrix_power(T_HO_VelocityVerlet(m, k, omegaXdelta_t / omega), i))
    angles_euler_exact.append(angle_euler-angle_exact)
    angles_verlet_exact.append(angle_verlet - angle_exact)
    # Update area ratio subplot
    ax2.clear()
    ax2.plot(area_ratios_euler_exact, label='Euler vs Exact', color='blue')
    ax2.plot(area_ratios_verlet_exact, label='Velocity Verlet vs Exact', color='red')
    ax2.set_title('Area Ratio Differences')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Area Ratio')
    ax2.axhline(0, color='black', lw=0.5, linestyle='--')  # Reference line at 1
    ax2.legend(loc="upper right")

    # Update orientation subplot
    ax3.clear()
    ax3.plot(angles_euler_exact,  color = 'blue')
    ax3.plot(angles_verlet_exact,  color = 'red')
    ax3.set_title('Orientation Differences')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Orientation (Degrees)')

    return ax1, ax2, ax3

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=NPlots + 1, interval=500, blit=False)

# Display the animation
#plt.tight_layout()
plt.show()

# Optionally, save the animation to a file
# ani.save('phase_space_animation.mp4', writer='ffmpeg')
