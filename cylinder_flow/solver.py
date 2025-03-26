import numpy as np
import cylinder_flow as cf

def compute_RHS(N, V_inf, alpha, n_vect,t_vect):
    """
    Compute the right-hand side (RHS) of the circulation equation.

    Parameters:
    -----------
    N : int
        Number of panels.
    V_inf : float
        Free-stream velocity magnitude.
    alpha : float
        Angle of attack in radians.
    n_vect : ndarray
        Normal vectors at the panel midpoints.

    Returns:
    --------
    RHS : ndarray
        Right-hand side vector of shape (N+1,).
    """
    RHS = np.zeros(N + 1)
    V_inf_vector = np.array([V_inf * np.cos(alpha), V_inf * np.sin(alpha)])

    for i in range(N):
        RHS[i] = np.dot(n_vect[i,:], V_inf_vector)
    # print(np.dot(n_vect, np.array([np.cos(alpha), np.sin(alpha)])))
    RHS[-1] = 0  # Enforce Kutta condition

    return RHS

def solve_circulation(AN, RHS):
    """
    Solve for the circulation distribution (Gamma) using the influence coefficient matrix.

    Parameters:
    -----------
    AN : ndarray
        Influence coefficient matrix of shape (N+1, N+1).
    RHS : ndarray
        Right-hand side vector of shape (N+1,).

    Returns:
    --------
    Gamma : ndarray
        Circulation strengths of shape (N+1,).
    """

    if AN.shape[0] != len(RHS):
        raise ValueError("RHS vector length must match AN matrix dimensions.")

    try:
        Gamma = np.linalg.solve(AN, RHS)
    except np.linalg.LinAlgError:
        raise ValueError("AN matrix is singular or ill-conditioned. Cannot solve for circulation.")

    return Gamma




def compute_tangential_velocity(N, V_inf, theta_mid, AN_3d, Gamma, AOA, t_vect,n_vect,RHS):
    """
    Compute the tangential velocity Vt at the panel midpoints.

    Parameters:
    -----------
    N : int
        Number of panels.
    V_inf : float
        Free-stream velocity magnitude.
    theta_mid : ndarray
        Midpoint angles of the panels.
    AN_3d : ndarray
        3D influence coefficient matrix of shape (2, N+1, N).
    Gamma : ndarray
        Circulation strengths of shape (N+1,).
    AOA : float
        Angle of attack in radians.

    Returns:
    --------
    Vt : ndarray
        Tangential velocity at the panel midpoints (shape: (N,)).
    """
    vel_vect = np.zeros((2,N))  
    Vt = np.zeros((N))
    # Gamma = Gamma.reshape(-1)  # Ensure Gamma is a 1D vector

    # print(np.shape(Gamma))

    # Compute induced velocities
    vel_vect = AN_3d@Gamma

    # for j in range(N+1):
    #     for i in range(N):
    #         # Compute velocity components using a for loop
    #         vel_vect[0,i] += AN_3d[0, i, j] * Gamma[j]
    #         vel_vect[1,i] += AN_3d[1, i, j] * Gamma[j]

    residual = np.linalg.norm(vel_vect[0,:]*n_vect[:,0] + vel_vect[1,:]*n_vect[:,1]   - RHS[:-1])
    print(f"Residual Error: {residual:.5e}")
    gamma_TE = Gamma[-1] + Gamma[0]  # Difference between last and first vortex strength
    print(f"Gamma at Trailing Edge: {gamma_TE:.5f}")
    cond_number = np.linalg.cond(AN_3d[0,:,:])
    print(f"Condition Number of AN: {cond_number:.2e}")

    # Velocity correction
    velocity_correction = V_inf * np.array([np.cos(AOA), np.sin(AOA)])
    
    vel_net = velocity_correction.reshape(-1,1) + vel_vect  
    
    Vt = np.linalg.norm(vel_net,axis = 0)
    Vt = vel_vect[0,:]*t_vect[:,0] + vel_vect[1,:]*t_vect[:,1]
    
    # Compute tangential velocity component
    # Vt =   vel_vect[1,:] + velocity_correction[:,1]
    # for i in range(N):
    #     # Total tangential velocity including freestream contribution
    #     Vt[i] = vel_vect[1,i]  + velocity_correction[1]*t_vect[i,1] + velocity_correction[0]*t_vect[i,0]

    return Vt, vel_vect

def compute_pressure_coefficient(Vt, V_inf):
    """
    Compute the pressure coefficient (Cp) from tangential velocity.

    Parameters:
    -----------
    Vt : ndarray
        Tangential velocity at the panel midpoints.
    V_inf : float
        Free-stream velocity magnitude.

    Returns:
    --------
    Cp : ndarray
        Coefficient of pressure at the panel midpoints.
    """
    if np.any(V_inf == 0):
        raise ValueError("V_inf cannot be zero.")

    Cp = 1 - (Vt / V_inf) ** 2
    return Cp


# def compute_lift_coefficient(Gamma, V_inf, R,x,y):
#     # Total circulation (sum of gamma * panel length)
#     dx = np.diff(x)
#     dy = np.diff(y)
#     theta_mid = np.arctan2(dy, dx)
#     r = np.sqrt(dx**2 + dy**2)
#     Gamma_total = np.sum(Gamma[:-1] * r)

#     # Compute C_L
#     Cl = (2 * Gamma_total) / (V_inf * 1)
#     return Cl

def compute_lift_coefficient(x, Cp, chord_length=1):
    """
    Calculate the lift coefficient (C_L) using surface pressure coefficient values.
    
    Parameters:
    x_coords (list of float): x-coordinates of the pressure points.
    cp_values (list of float): Surface pressure coefficient at corresponding x-coordinates.
    chord_length (float): Chord length of the airfoil.
    
    Returns:
    float: Computed lift coefficient (C_L)
    """
    if len(x) != len(Cp):
        raise ValueError("x_coords and cp_values must have the same length")
    
    N = len(x) - 1  # Number of panels
    Cl = sum(Cp[i] * (x[i] - x[i+1]) for i in range(N)) / chord_length
    
    return Cl





# Unit Tests
def test_compute_RHS():
    N, V_inf, alpha = 10, 1.0, np.radians(5)
    _, _, _, _, n_vect = cf.pre_processing(*cf.cylinder(N, 1.0))

    RHS = compute_RHS(N, V_inf, alpha, n_vect)

    assert RHS.shape == (N + 1,), "RHS shape is incorrect."
    assert np.isclose(RHS[-1], 0), "Last element of RHS should be zero (Kutta condition)."
    assert np.all(np.isfinite(RHS)), "RHS contains NaN or Inf values."


def test_solve_circulation():
    N, R = 10, 1.0
    x, y = cf.cylinder(N, R)
    x_mid, y_mid, _, _, n_vect = cf.pre_processing(x, y)
    
    AN = cf.compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect)
    RHS = compute_RHS(N, 1.0, np.radians(5), n_vect)

    Gamma = solve_circulation(AN, RHS)

    assert Gamma.shape == (N + 1,), "Gamma shape is incorrect."
    assert np.all(np.isfinite(Gamma)), "Gamma contains NaN or Inf values."

    # Check singularity handling
    try:
        solve_circulation(np.zeros((N+1, N+1)), RHS)
    except ValueError as e:
        assert str(e) == "AN matrix is singular or ill-conditioned. Cannot solve for circulation."

def test_compute_tangential_velocity():
    N, V_inf, AOA = 10, 1.0, np.radians(5)
    x, y = cf.cylinder(N, 1.0)
    x_mid, y_mid, theta_mid, t_vect, n_vect, x,y = cf.pre_processing(x, y)
    
    AN_3d = cf.compute_AN_3d_matrix(N, x, y, x_mid, y_mid,n_vect,t_vect)
    RHS = cf.compute_RHS(N, V_inf, AOA, n_vect,t_vect)
    AN,_ = cf.compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect,t_vect)
    Gamma = cf.solve_circulation(AN, RHS)

    Vt,_ = cf.compute_tangential_velocity(N, V_inf, theta_mid, AN_3d, Gamma, AOA,t_vect,n_vect,RHS)

    assert Vt.shape == (N,), "Vt shape is incorrect."
    assert np.all(np.isfinite(Vt)), "Vt contains NaN or Inf values."


def test_compute_pressure_coefficient():
    N, V_inf = 10, 1.0
    Vt = np.linspace(0.5, 1.5, N)

    Cp = compute_pressure_coefficient(Vt, V_inf)

    assert Cp.shape == (N,), "Cp shape is incorrect."
    assert np.all(np.isfinite(Cp)), "Cp contains NaN or Inf values."
    assert np.all((Cp >= -2) & (Cp <= 1)), "Cp values are out of expected range."

    # Test division by zero error
    try:
        compute_pressure_coefficient(Vt, 0)
    except ValueError as e:
        assert str(e) == "V_inf cannot be zero."







# Unit Tests
def test_compute_RHS():
    N, V_inf, alpha = 10, 1.0, np.radians(5)
    _, _, _, t_vect, n_vect, x,y = cf.pre_processing(*cf.cylinder(N, 1.0))

    RHS = compute_RHS(N, V_inf, alpha, n_vect,t_vect)

    assert RHS.shape == (N + 1,), "RHS shape is incorrect."
    assert np.isclose(RHS[-1], 0), "Last element of RHS should be zero (Kutta condition)."
    assert np.all(np.isfinite(RHS)), "RHS contains NaN or Inf values."


def test_solve_circulation():
    N, R = 10, 1.0
    x, y = cf.cylinder(N, R)
    x_mid, y_mid, theta_mid, t_vect, n_vect,  x, y = cf.pre_processing(x, y)
    
    AN,_ = cf.compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect,t_vect)
    RHS = compute_RHS(N, 1.0, np.radians(5), n_vect,t_vect)

    Gamma = solve_circulation(AN, RHS)

    assert Gamma.shape == (N + 1,), "Gamma shape is incorrect."
    assert np.all(np.isfinite(Gamma)), "Gamma contains NaN or Inf values."

    # Check singularity handling
    try:
        solve_circulation(np.zeros((N+1, N+1)), RHS)
    except ValueError as e:
        assert str(e) == "AN matrix is singular or ill-conditioned. Cannot solve for circulation."

# def test_equation(RHS, vel_vect, n_vect ):
#     assert (RHS + vel_vect)




    