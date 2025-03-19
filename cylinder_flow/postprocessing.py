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

def plot_induced_velocity_vectors(x, y, n_induced_array):
    """
    Plot the induced normal velocity vectors for each panel.

    Parameters:
    -----------
    x, y : ndarray
        x and y coordinates of the panel endpoints.
    n_induced_array : ndarray
        Array of induced normal vectors at each panel.
    """
    # Check that the lengths of x, y, and n_induced_array are consistent
    if len(x) != len(y) or len(n_induced_array) != len(x) - 1:
        raise ValueError("Inconsistent array lengths. Ensure x, y, and n_induced_array match.")
    
    # Plot each induced velocity vector at each panel midpoint
    plt.figure(figsize=(20, 18))
    for i in range(len(n_induced_array)):
        # Midpoint coordinates of the current panel
        x_mid = (x[i] + x[i + 1]) / 2
        y_mid = (y[i] + y[i + 1]) / 2
        
        # Induced velocity vector
        n_induced = n_induced_array[i]
        
        # Plot the vector at the panel midpoint
        if i == 1: 
            plt.quiver(x_mid, y_mid, n_induced[0], n_induced[1], angles='xy', scale_units='xy', scale=5, color='b', alpha=0.7,label = "normal induction vector of the last panel")
        else:
            plt.quiver(x_mid, y_mid, n_induced[0], n_induced[1], angles='xy', scale_units='xy', scale=5, color='b', alpha=0.7)
    # Plot the markers for the panel endpoints
    plt.scatter(x, y, color='r', label="Panel Endpoints", zorder=5)  # Red markers for endpoints
    # Labels and grid
    plt.title("Induced Velocity Vectors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
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

    
def plot_shape(x_mid, y_mid,n_vect,t_vect):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_mid, y_mid, color ='b', label='',zorder = 2)
    plt.quiver(x_mid, y_mid, n_vect[:,0], n_vect[:,1], angles='xy', scale_units='xy', scale=5, color='r', alpha=0.7,label = 'normal')
    plt.xlabel('x')
    plt.quiver(x_mid, y_mid, t_vect[:,0], t_vect[:,1], angles='xy', scale_units='xy', scale=1, color='g', alpha=0.7,label = 'tangential')
    plt.ylabel('y')
    plt.title('Midpoints')
    plt.axis("equal")
    plt.grid()
    plt.show()

