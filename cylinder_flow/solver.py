import numpy as np

def solve_circulation(N, AN, V_inf, alpha, n_vect):
    RHS = np.zeros(len(n_vect)+1)
    RHS[0:len(n_vect)] = -V_inf * np.dot(n_vect, np.array([np.cos(alpha), np.sin(alpha)]))
    RHS[-1] = 0
    print('Shape of RHS: ',np.shape(RHS))
    Gamma = np.linalg.solve(AN, RHS)
    # Gamma, residuals, rank, singular_values = np.linalg.lstsq(AN, RHS, rcond=None)
    # other approach: Gamma = np.linalg.inv(AN_T @ AN) @ AN_T @ RHS
    return Gamma

def compute_velocity_and_pressure(N, V_inf, theta_mid, AN, Gamma):
    Vt = -V_inf * np.sin(theta_mid) + np.dot(AN[0:-1], Gamma)
    Cp = 1 - (Vt / V_inf) ** 2
    return Vt, Cp

def compute_velocity_and_pressure_3d(N, V_inf, theta_mid, AN, AN_3d, Gamma,AOA,t_vect,n_vect):

     # Initialize vel_vect with the correct shape (2, np)
    vel_vect = np.zeros((2, N))  # AN_3d has shape (2, nv, np)

    print('Initial AN_3d shape:', np.shape(AN_3d))
    
    # Reshape Gamma to (nv,) for broadcasting compatibility
    Gamma = Gamma.reshape(-1)  # Ensure Gamma is a 1D vector of shape (nv,)

    # Perform the matrix product along the correct axes
    vel_vect[0,:] = np.tensordot(AN_3d[0,:,:], Gamma, axes=(0, 0))  # Sum over axis 0 (nv) for each panel
    vel_vect[1,:] = np.tensordot(AN_3d[1,:,:], Gamma, axes=(0, 0))
    print('Shape of vel_vect after matrix multiplication:', np.shape(vel_vect))

    # Velocity correction (this will be added to the 2D vel_vect)
    velocity_correction = (
    V_inf  * np.array([[np.cos(AOA)], [np.sin(AOA)]])
    )    
    #add the velocity correction to vel_vect (shape (2, np) + (2, 1))
    vel_vect -= velocity_correction 
    
    print('Shape of vel_vect after adding velocity correction:', np.shape(vel_vect))
    # print(vel_vect)
    # Calculate the velocity magnitude
    Vt = np.linalg.norm(vel_vect, axis=0)# This gives a vector of shape (np), magnitude of Vel

    Vt = vel_vect[1,:] # taking only the component in y

    # Calculate Cp (coefficient of pressure)
    Cp = 1 - (Vt/ V_inf) ** 2
    return Vt, Cp

def compute_lift_coefficient(Gamma, V_inf, R,x):
    Cl = 2 * np.sum(Gamma) / (V_inf * 2 * np.pi * R)
    return Cl

def compute_lift_coefficient(Gamma, V_inf, R,x):
    Cl = 2 * np.sum(Gamma) / (V_inf * 2 * np.pi * R)
    # Cl = 2* Gamma / (np.abs(np.max(x)) - np.abs(np.min(x)) )
    return Cl
