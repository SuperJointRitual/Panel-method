import numpy as np
import cylinder_flow as cf
import matplotlib.pyplot as plt

def main():
    N = 100
    R = 1.0
    V_inf = 1
    alpha = np.deg2rad(0)
    m = 0.  # Max camber (2% of chord length)
    p = 0.   # Max camber position (40% of chord length)
    t = 0.12  # Max thickness (12% of chord length)

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
    x, y = cf.cylinder(N, R)
    x_mid, y_mid, theta_mid, t_vect, n_vect,  x, y  = cf.pre_processing(x,y)

    # x_mid, y_mid, theta_mid, tb_vect, n_vect,x,y = cf.define_airfoil_geometry(N,m, p, t)
    cf.plot_shape(x_mid,y_mid,n_vect,t_vect)
    print(np.shape(t_vect))
    

    AN, n_induced_array_last = cf.compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect,t_vect)
    AN_3d = cf.compute_AN_3d_matrix(N, x, y, x_mid, y_mid,n_vect,t_vect)
    RHS = cf.compute_RHS(N, V_inf, alpha, n_vect,t_vect)
    Gamma = cf.solve_circulation(AN, RHS)
    
    print('Shape of influence matrix 3D: ', np.shape(AN_3d))

  

    print('Sum of circulations: ',np.sum(Gamma))
    print('circulations: ',(Gamma))
    # print(np.shape(Gamma))


    # Compute velocity and pressure coefficient
    Vt, vel_vect = cf.compute_tangential_velocity(N, V_inf, theta_mid, AN_3d, Gamma, alpha, t_vect,n_vect,RHS)
    Cp = cf.compute_pressure_coefficient(Vt, V_inf)
  
    # cf.test_equation(RHS, vel_vect, n_vect )

    # Plot results
    cf.plot_cp_distribution(np.linspace(0+1e-3, 2 * np.pi-1e-3, N ), Cp)
    cf.plot_velocity_distribution(np.linspace(0+1e-3, 2 * np.pi-1e-3, N ), Vt)

    # Compute and print lift coefficient
    Cl = cf.compute_lift_coefficient(x_mid, Cp, chord_length=1)
    print(f"Lift coefficient (Cl): {Cl}")

    # Plot induced velocity vectors
    cf.plot_induced_velocity_vectors(x, y, n_induced_array_last)
    cf.plot_shape(x_mid,y_mid,n_vect,t_vect)
    


if __name__ == "__main__":
    main()
