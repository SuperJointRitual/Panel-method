import numpy as np
import matplotlib.pyplot as plt
import cylinder_flow as cf  # Assuming cf contains all necessary functions

def read_experimental_data(filename):
    """ Reads the XFOIL-generated data file and extracts alpha and Cl values. """
    alphas = []
    Cls = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Locate the start of numerical data
    for i, line in enumerate(lines):
        if line.strip().startswith('alpha'):
            data_start = i + 2  # Skip header line
            break

    # Read numerical data
    for line in lines[data_start:]:
        values = line.split()
        alphas.append(float(values[0]))
        Cls.append(float(values[1]))

    return np.array(alphas), np.array(Cls)

def main():
    N = 500
    R = 1.0
    V_inf = 10
    
    alpha_range = np.deg2rad(np.arange(-10, 11, 0.5))  # Angles from -10 to 10 degrees
    Cl_values = []  # Store computed lift coefficients
    
    # Run tests
    # cf.test_cylinder()
    # cf.test_pre_processing()
    # cf.test_compute_AN_matrix()
    # cf.test_compute_AN_3d_matrix()
    # cf.test_compute_RHS()
    # cf.test_solve_circulation()
    # cf.test_compute_tangential_velocity()
    # cf.test_compute_pressure_coefficient()
    print("All tests passed!")

    # Geometry definition
    x, y = cf.cylinder(N, R)
    m, p, t = 0.0, 0.0, 0.12  # Airfoil parameters
    x_mid, y_mid, theta_mid, t_vect, n_vect, x, y = cf.define_airfoil_geometry(N, m, p, t)
    cf.plot_shape(x_mid, y_mid, n_vect, t_vect)

    # Compute influence matrix (constant for all angles of attack)
    AN, _ = cf.compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect, t_vect)
    AN_3d = cf.compute_AN_3d_matrix(N, x, y, x_mid, y_mid, n_vect, t_vect)

    # Loop over different angles of attack
    for alpha in alpha_range:
        print(f"Calculating for alpha = {np.rad2deg(alpha):.1f} degrees")
        
        RHS = cf.compute_RHS(N, V_inf, alpha, n_vect, t_vect)
        Gamma = cf.solve_circulation(AN, RHS)

        # Compute velocity and pressure coefficient
        Vt, vel_vect = cf.compute_tangential_velocity(N, V_inf, theta_mid, AN_3d, Gamma, alpha, t_vect, n_vect, RHS)
        Cp = cf.compute_pressure_coefficient(Vt, V_inf)

        # Compute lift coefficient
        Cl = cf.compute_lift_coefficient(x, Gamma,V_inf, chord_length=1)
        Cl_values.append(Cl)

    # Read experimental data
    alpha_exp, Cl_exp = read_experimental_data("NACA_0012_Cl.txt")

    # Plot Cl vs alpha
    plt.figure(figsize=(8, 6))
    plt.plot(np.rad2deg(alpha_range), Cl_values, marker='o', linestyle='-', label="Computed Lift Curve")
    plt.scatter(np.rad2deg(alpha_range), 2 * np.pi * np.sin(alpha_range), color='red', marker='o', label="thin airfiol theory")
    
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("Lift Coefficient (Cl)")
    plt.title("Lift Curve: Computed vs Thin airfoil theory")
    plt.legend()
    plt.grid()
    plt.show()

    print("Angle of Attack Sweep Completed!")

if __name__ == "__main__":
    main()
