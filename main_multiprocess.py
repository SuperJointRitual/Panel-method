import numpy as np
import multiprocessing as mp
import cylinder_flow as cf  

def process_N(N):
    # Geometry definition
    R = 1.0
    V_inf = 1.0
    alpha = 0
    
    x, y, x_mid, y_mid, theta_mid, t_vect, n_vect = cf.define_geometry(N, R)
    
    # Influence coefficients
    AN = cf.compute_influence_coefficients(N, x, y, x_mid, y_mid, n_vect)
    print(np.size(AN, 0))

    # Solve circulation
    Gamma = cf.solve_circulation(N, AN, V_inf, alpha, n_vect)
    print(sum(Gamma))
    print(np.shape(Gamma))

    # Compute velocity and pressure coefficient
    # Vt, Cp = cf.compute_velocity_and_pressure(N, V_inf, theta_mid, AN, Gamma)

    # Plot results
    # cf.plot_cp_distribution(np.linspace(0, 2 * np.pi, N+1), Cp)
    # cf.plot_velocity_distribution(np.linspace(0, 2 * np.pi, N+1), Vt)

    # Compute and print lift coefficient
    Cl = cf.compute_lift_coefficient(Gamma, V_inf, R, x)
    print(f"Lift coefficient (Cl): {Cl}")

    # Plot induced velocity vectors
    cf.plot_induced_velocity_vectors(N, R, x, y)

    return Cl

def main():
    N_values = [10, 100, 300, 500]
    
    # Set up multiprocessing Pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        Cl_values = pool.map(process_N, N_values)
    
    # After processing all the Cl values, you can use them as needed
    print("Cl values:", Cl_values)
    cf.plot_convergence(N_values, Cl_values)

if __name__ == "__main__":
    main()
