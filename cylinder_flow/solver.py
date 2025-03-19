import numpy as np
import cylinder_flow as cf

def compute_RHS(N, V_inf, alpha, n_vect):
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
    RHS[:-1] = V_inf * np.dot(n_vect, np.array([np.cos(alpha), np.sin(alpha)]))
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




def compute_tangential_velocity(N, V_inf, theta_mid, AN_3d, Gamma, AOA):
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
    vel_vect = np.zeros((2, N))  

    # Gamma = Gamma.reshape(-1)  # Ensure Gamma is a 1D vector

    # print(np.shape(Gamma))

    # Compute induced velocity
    vel_vect = np.dot(AN_3d, Gamma)  

    print('norm vel', np.linalg.norm(vel_vect, axis = 0))

    # Velocity correction
    velocity_correction = V_inf * np.array([[np.cos(AOA)], [np.sin(AOA)]])
    
    vel_vect += velocity_correction  # Broadcasting ensures shape (2, N)
    
    print('norm vel', np.linalg.norm(vel_vect, axis = 0))

    
    # Compute tangential velocity component
    Vt = vel_vect[0, :] * np.cos(theta_mid) + vel_vect[1, :] * np.sin(theta_mid)

    return Vt

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


def compute_lift_coefficient(Gamma, V_inf, R,x):
    Cl = 2 * np.sum(Gamma) / (V_inf * 2 * np.pi * R)
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
    x_mid, y_mid, theta_mid, _, n_vect, x,y = cf.pre_processing(x, y)
    
    AN_3d = cf.compute_AN_3d_matrix(N, x, y, x_mid, y_mid)
    RHS = cf.compute_RHS(N, V_inf, AOA, n_vect)
    AN,_ = cf.compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect)
    Gamma = cf.solve_circulation(AN, RHS)

    Vt = cf.compute_tangential_velocity(N, V_inf, theta_mid, AN_3d, Gamma, AOA)

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
    _, _, _, _, n_vect, x,y = cf.pre_processing(*cf.cylinder(N, 1.0))

    RHS = compute_RHS(N, V_inf, alpha, n_vect)

    assert RHS.shape == (N + 1,), "RHS shape is incorrect."
    assert np.isclose(RHS[-1], 0), "Last element of RHS should be zero (Kutta condition)."
    assert np.all(np.isfinite(RHS)), "RHS contains NaN or Inf values."


def test_solve_circulation():
    N, R = 10, 1.0
    x, y = cf.cylinder(N, R)
    x_mid, y_mid, _, _, n_vect, x,y = cf.pre_processing(x, y)
    
    AN,_ = cf.compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect)
    RHS = compute_RHS(N, 1.0, np.radians(5), n_vect)

    Gamma = solve_circulation(AN, RHS)

    assert Gamma.shape == (N + 1,), "Gamma shape is incorrect."
    assert np.all(np.isfinite(Gamma)), "Gamma contains NaN or Inf values."

    # Check singularity handling
    try:
        solve_circulation(np.zeros((N+1, N+1)), RHS)
    except ValueError as e:
        assert str(e) == "AN matrix is singular or ill-conditioned. Cannot solve for circulation."

