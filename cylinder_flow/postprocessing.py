import numpy as np
import matplotlib.pyplot as plt

def plot_cp_distribution(theta, Cp):
    plt.figure(figsize=(8, 6))
    plt.plot(np.rad2deg(theta[:-1]), Cp, '-', label='$C_p$')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('$C_p$')
    plt.title('Pressure Coefficient Distribution around a Cylinder')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    plt.show()

def plot_velocity_distribution(theta, Vt):
    plt.figure(figsize=(8, 6))
    plt.plot(np.rad2deg(theta[:-1]), Vt, '-', label='$V_t$')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('$V_t$')
    plt.title('Tangential Velocity Distribution around a Cylinder')
    plt.grid()
    plt.show()

def plot_induced_velocity_vectors(N, R, x, y):
    theta_range = np.linspace(0, 2 * np.pi, 100)
    x_subset = R * np.cos(theta_range)
    y_subset = R * np.sin(theta_range)
    x_mid_subset = (x_subset[:-1] + x_subset[1:]) / 2
    y_mid_subset = (y_subset[:-1] + y_subset[1:]) / 2
    R_pan_subset = np.sqrt(x_mid_subset**2 + y_mid_subset**2)
    t_vect_subset = np.column_stack((x_mid_subset / R_pan_subset, y_mid_subset / R_pan_subset))
    n_vect_subset = np.column_stack((t_vect_subset[:, 1], -t_vect_subset[:, 0]))
    n_induced = np.zeros((len(x_mid_subset), 2))
    t_induced = np.zeros((len(x_mid_subset), 2))
    for i in range(len(x_mid_subset)):
        for j in range(N):
            dx = x_mid_subset[i] - x[j]
            dy = y_mid_subset[i] - y[j]
            r = np.sqrt(dx**2 + dy**2)
            t_induced[i] = np.array([dx / r, dy / r])
            n_induced[i] = np.array([dy / r, -dx / r])
    plt.figure(figsize=(8, 8))
    plt.plot(x_subset, y_subset, marker='.', label='Cylinder Surface', color='black')
    plt.quiver(x_mid_subset, y_mid_subset, t_induced[:, 0], t_induced[:, 1], color='blue', scale=10, label='Induced Tangential Vectors')
    plt.quiver(x_mid_subset, y_mid_subset, n_induced[:, 0], n_induced[:, 1], color='red', scale=10, label='Induced Normal Vectors')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Induced Normal and Tangential Vectors on Cylinder')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid()
    plt.show()

def plot_convergence(N_values, Cl_values):
    """Plot Cl convergence for different values of N."""
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, Cl_values, marker='o', linestyle='-', label='Cl vs. N')
    plt.xlabel('N')
    plt.ylabel('Cl')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.title('Convergence of Cl with Increasing N')
    plt.legend()
    plt.grid(True)
    plt.show()
