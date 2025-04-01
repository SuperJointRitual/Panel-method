import numpy as np
import matplotlib.pyplot as plt

# 1. DATA
N = 100  # Number of panels
U_inf = 1  # Free stream velocity
alpha = np.radians(0)  # Angle of attack
x_ref = 0  # Reference point for moment computation
R = 1  # Cylinder radius

def compute_geometry_cylinder(R, N):
    theta = np.linspace(0, 2 * np.pi, N + 1)
    x, z = R * np.cos(theta), R * np.sin(theta)
    vortex = np.vstack((R * np.cos(theta[:-1] + np.pi / N), R * np.sin(theta[:-1] + np.pi / N))).T
    node = np.vstack((R * np.cos(theta[:-1] + 2 * np.pi / (2 * N)), R * np.sin(theta[:-1] + 2 * np.pi / (2 * N)))).T
    n_vec = np.vstack((np.cos(theta[:-1]), np.sin(theta[:-1]))).T
    return x, z, vortex, node, n_vec, theta[:-1]

def compute_circulation(U_inf, alpha, vortex, node, n_vec, N):
    A, RHS = np.zeros((N, N)), np.zeros(N)
    eps = 1e-10  # Small number to prevent division by zero
    for i in range(N):
        for j in range(N):
            r_sq = (node[i, 0] - vortex[j, 0]) ** 2 + (node[i, 1] - vortex[j, 1]) ** 2
            if r_sq < eps:
                continue  # Skip self-induced velocity computation
            u, w = (node[i, 1] - vortex[j, 1]) / (2 * np.pi * r_sq), -(node[i, 0] - vortex[j, 0]) / (2 * np.pi * r_sq)
            A[i, j] = n_vec[i, 0] * u + n_vec[i, 1] * w
        RHS[i] = -U_inf * (n_vec[i, 0] * np.cos(alpha) + n_vec[i, 1] * np.sin(alpha))
    
    Gamma = np.linalg.solve(A, RHS)
    return Gamma

def compute_coefficients_dvm(U_inf, chord, alpha, x_ref, Gamma, vortex):
    Cl = 2 * np.sum(Gamma) / (U_inf * chord)
    Cm = -2 * np.cos(alpha) * np.dot(Gamma, (vortex[:, 0] - x_ref)) / (U_inf * chord ** 2)
    return Cl, Cm

def compute_pressure_coefficient(U_inf, Gamma, N):
    Cp = 1 - (Gamma / (U_inf * (2 * np.pi * R / N))) ** 2
    return Cp

def compute_tangential_velocity(U_inf, Gamma, N):
    V_tangential = U_inf + Gamma / (2 * np.pi * R / N)
    return V_tangential

# Cylinder case
x_cyl, z_cyl, vortex_cyl, node_cyl, n_vec_cyl, theta_cyl = compute_geometry_cylinder(R, N)
Gamma_cyl = compute_circulation(U_inf, alpha, vortex_cyl, node_cyl, n_vec_cyl, N)
Cl_cyl, Cm_cyl = compute_coefficients_dvm(U_inf, 2 * R, alpha, x_ref, Gamma_cyl, vortex_cyl)
Cp_cyl = compute_pressure_coefficient(U_inf, Gamma_cyl, N)
V_tangential_cyl = compute_tangential_velocity(U_inf, Gamma_cyl, N)

# Plot pressure coefficient
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta_cyl), Cp_cyl, marker='o', linestyle='-', label='Cylinder Cp')
plt.xlabel("Angle (degrees)")
plt.ylabel("Pressure Coefficient (Cp)")
plt.gca().invert_yaxis()  # Invert y-axis to match aerodynamic convention
plt.title("Pressure Coefficient Distribution around Cylinder")
plt.legend()
plt.grid()
plt.show()

# Plot tangential velocity
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta_cyl), V_tangential_cyl, marker='s', linestyle='-', color='r', label='Tangential Velocity')
plt.xlabel("Angle (degrees)")
plt.ylabel("Tangential Velocity (V_tangential)")
plt.title("Tangential Velocity Distribution around Cylinder")
plt.legend()
plt.grid()
plt.show()
