import numpy as np
import cylinder_flow as cf

def main():
    N = 100
    R = 1.0
    V_inf = 1.0
    alpha = 0

    # Geometry definition
    x, y, x_mid, y_mid, theta_mid, t_vect, n_vect = cf.define_geometry(N, R)
    # x, y, x_mid, y_mid, theta_mid, t_vect, n_vect = cf.define_airfoil_geometry(N/2)
    # Influence coefficients
    AN, AN_3d  = cf.compute_influence_coefficients(N, x, y, x_mid, y_mid, n_vect,t_vect)
    print('Shape of influence matrix 3D: ', np.shape(AN_3d))
    # print(AN_3d)
    # print('Lasto row of influence matrix',AN[-1,:])
    # Solve circulation
    Gamma = cf.solve_circulation(N, AN, V_inf, alpha, n_vect)
    print('Sum of circulations: ',np.sum(Gamma))
    # print(np.shape(Gamma))

    # Compute velocity and pressure coefficient
    Vt, Cp = cf.compute_velocity_and_pressure_3d(N, V_inf, theta_mid, AN, AN_3d, Gamma,alpha,t_vect,n_vect)
  

    # Plot results
    cf.plot_cp_distribution(np.linspace(0, 2 * np.pi, N+1), Cp)
    cf.plot_velocity_distribution(np.linspace(0, 2 * np.pi, N+1), Vt)

    # Compute and print lift coefficient
    Cl = cf.compute_lift_coefficient(Gamma, V_inf, R, x)
    print(f"Lift coefficient (Cl): {Cl}")

    # Plot induced velocity vectors
    cf.plot_induced_velocity_vectors(N, R, x, y)

if __name__ == "__main__":
    main()
