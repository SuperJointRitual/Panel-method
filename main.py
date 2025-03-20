import numpy as np
import cylinder_flow as cf

def main():
    N = 100
    R = 1.0
    V_inf = 1.0
    alpha = 0
    cf.test_cylinder()
    cf.test_pre_processing()
    cf.test_compute_AN_matrix()
    cf.test_compute_AN_3d_matrix()
    cf.test_compute_RHS()
    cf.test_solve_circulation()
    cf.test_compute_tangential_velocity()
    cf.test_compute_pressure_coefficient()
    print("All tests passed!")
    # Geometry definition
    x, y, = cf.cylinder(N, R)
    x_mid, y_mid, theta_mid, t_vect, n_vect,  x, y  = cf.pre_processing(x,y)
    # n_induced= cf.compute_n_induced(x, y, x_mid, y_mid)
    # x, y, x_mid, y_mid, theta_mid, t_vect, n_vect = cf.define_airfoil_geometry(N/2)

    AN, n_induced_array_last = cf.compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect)
    AN_3d = cf.compute_AN_3d_matrix(N, x, y, x_mid, y_mid)
    RHS = cf.compute_RHS(N, V_inf, alpha, n_vect)
    Gamma = cf.solve_circulation(AN, RHS)
    print('Shape of influence matrix 3D: ', np.shape(AN_3d))

  

    # print(AN_3d)
    # print('Lasto row of influence matrix',AN[-1,:])
    # Solve circulation
    print('Sum of circulations: ',np.sum(Gamma))
    print('circulations: ',(Gamma))
    # print(np.shape(Gamma))

    # Compute velocity and pressure coefficient
    Vt, vel_vect = cf.compute_tangential_velocity(N, V_inf, theta_mid, AN_3d, Gamma, alpha,t_vect)
    Cp = cf.compute_pressure_coefficient(Vt, V_inf)
  
    # cf.test_equation(RHS, vel_vect, n_vect )

    # Plot results
    cf.plot_cp_distribution(np.linspace(0, 2 * np.pi, N+1), Cp)
    cf.plot_velocity_distribution(np.linspace(0, 2 * np.pi, N+1), Vt)

    # Compute and print lift coefficient
    Cl = cf.compute_lift_coefficient(Gamma, V_inf, R, x)
    print(f"Lift coefficient (Cl): {Cl}")

    # Plot induced velocity vectors
    # cf.plot_induced_velocity_vectors(x, y, n_induced_array_last)
    cf.plot_shape(x_mid,y_mid,n_vect,t_vect)

if __name__ == "__main__":
    main()
