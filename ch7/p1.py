import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- Configuration ---
# Stokes Parameters (S0, S1, S2, S3)
# S0: Total intensity
# S1: Linear polarization along 0/90 degrees
# S2: Linear polarization along 45/135 degrees
# S3: Circular polarization (positive for right-handed, negative for left-handed)


S0 = 25.0  # Total intensity (can be normalized to 1)
S1 = 0  # Linear polarization component (horizontal/vertical)
S2 = 24  # Linear polarization component (+45/-45 degrees)
S3 = 7  # Circular polarization component (right/left)

# Number of points to plot for the ellipse
num_points = 100
# Propagation distance (arbitrary units for visualization)
propagation_distance = 2 * np.pi
# Speed of light (normalized for plotting, actual value is ~3e8 m/s)
c = 1.0

# --- Function to convert Stokes parameters to Electric Field components ---
def stokes_to_e_field_params(S0, S1, S2, S3):
    """
    Converts Stokes parameters (S0, S1, S2, S3) to electric field amplitudes
    (Ax, Ay) and phase difference (delta).

    Args:
        S0 (float): Total intensity.
        S1 (float): Linear polarization along 0/90 degrees.
        S2 (float): Linear polarization along 45/135 degrees.
        S3 (float): Circular polarization.

    Returns:
        tuple: (Ax, Ay, delta), where Ax and Ay are amplitudes of Ex and Ey,
               and delta is the phase difference (phi_y - phi_x).
    """
    if S0 < 0:
        raise ValueError("Intensity I must be non-negative.")
    if S0 == 0:
        return 0, 0, 0 # No field if intensity is zero

    
    # Amplitudes of Ex and Ey
    Ax = np.sqrt((S0 + S1) / 2.0)
    Ay = np.sqrt((S0 - S1) / 2.0)

    if Ax == 0 and Ay == 0:
        delta = 0.0
    elif S2 == 0 and S3 == 0:
        # Linear polarization (horizontal or vertical)
        delta = 0.0 # Or pi, depending on relative phases, but 0 is common for this case
    elif Ax * Ay == 0: # One component is zero, so it's linear.
        delta = 0.0
    else:
        delta = np.arctan2(S3, S2)

    return Ax, Ay, delta

# --- Calculate E-field parameters ---
try:
    Ax, Ay, delta = stokes_to_e_field_params(S0, S1, S2, S3)
    print(f"Derived E-field amplitudes: Ax={Ax:.3f}, Ay={Ay:.3f}")
    print(f"Derived E-field phase difference (delta): {np.degrees(delta):.3f} degrees")
except ValueError as e:
    print(f"Error in Stokes parameters: {e}")
    exit()

# --- Generate time/propagation steps ---
# We'll trace the field over one period (2*pi in phase angle)
phi = np.linspace(0, 2 * np.pi, num_points) # Omega*t or k*z

# --- Calculate E and B field components over time/space ---
Ex_vals = Ax * np.cos(phi)
Ey_vals = Ay * np.cos(phi + delta)
Ez_vals = np.zeros_like(phi) # E-field is in the xy-plane for z-propagation

Bx_vals = -Ey_vals / c
By_vals = Ex_vals / c
Bz_vals = np.zeros_like(phi) # B-field is also in the xy-plane for z-propagation

# --- Plotting ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set up the plot limits
max_val = max(Ax, Ay, np.max(np.abs(Bx_vals)), np.max(np.abs(By_vals))) * 1.2
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val]) # Z-axis for propagation or just for vector length

ax.set_xlabel('X (Electric Field Component)')
ax.set_ylabel('Y (Electric Field Component)')
ax.set_zlabel('Z (Propagation Direction)')
ax.set_title('Electric (E) and Magnetic (B) Field Vectors from Stokes Parameters')

ax.plot(Ex_vals, Ey_vals, np.zeros_like(phi), linestyle='--', color='gray', alpha=0.6, label='E-field Polarization Ellipse')

ax.plot(Bx_vals, By_vals, np.zeros_like(phi), linestyle=':', color='orange', alpha=0.6, label='B-field Polarization Ellipse')

e_vector, = ax.plot([0, Ex_vals[0]], [0, Ey_vals[0]], [0, Ez_vals[0]], color='blue', linewidth=2, label='Electric Field (E)')
b_vector, = ax.plot([0, Bx_vals[0]], [0, By_vals[0]], [0, Bz_vals[0]], color='red', linewidth=2, label='Magnetic Field (B)')

e_tip, = ax.plot([Ex_vals[0]], [Ey_vals[0]], [Ez_vals[0]], 'o', color='blue', markersize=5)

b_tip, = ax.plot([Bx_vals[0]], [By_vals[0]], [Bz_vals[0]], 'x', color='red', markersize=5)

ax.legend()

ax.view_init(elev=20, azim=-60)

def update(frame):
    """
    Update function for the animation, drawing E and B vectors at each frame.
    """
    e_vector.set_data_3d([0, Ex_vals[frame]], [0, Ey_vals[frame]], [0, Ez_vals[frame]])
    e_tip.set_data_3d([Ex_vals[frame]], [Ey_vals[frame]], [Ez_vals[frame]])

    b_vector.set_data_3d([0, Bx_vals[frame]], [0, By_vals[frame]], [0, Bz_vals[frame]])
    b_tip.set_data_3d([Bx_vals[frame]], [By_vals[frame]], [Bz_vals[frame]])

    ax.view_init(elev=20, azim=frame * 360 / num_points - 60) # Rotate azimuthally

    return e_vector, b_vector, e_tip, b_tip

# Create the animation
ani = FuncAnimation(fig, update, frames=num_points, blit=True, interval=50)

print("Saving animation to polarization_animation.gif...")
ani.save('polarization_animation.gif', writer='pillow', fps=20)
print("Animation saved!")