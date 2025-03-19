import numpy as np
import cylinder_flow as cf


def compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect):
    """
    Compute the influence coefficient matrix AN and store the induced normal vectors (n_induced).

    Parameters:
    -----------
    N : int
        Number of panels.
    x, y : ndarray
        x and y coordinates of panel endpoints.
    x_mid, y_mid : ndarray
        x and y coordinates of panel midpoints.
    n_vect : ndarray
        Normal vectors at the panel midpoints.

    Returns:
    --------
    AN : ndarray
        Influence coefficient matrix of shape (N+1, N+1).
    n_induced_array : ndarray
        Array of induced normal vectors at each panel.
    """
    AN = np.zeros((N + 1, N + 1))
    n_induced_array = np.zeros((N, 2))  # Array to store induced normal vectors
    AN_3d = compute_AN_3d_matrix(N, x, y, x_mid, y_mid)

    for i in range(N):
        for j in range(N + 1):
            # dx = x_mid[i] - x[j]
            # dy = y_mid[i] - y[j]
            # r = np.sqrt(dx**2 + dy**2)

            # if np.isclose(r, 0):  # Avoid division by zero
            #     continue

            # # Calculate the induced normal vector
            # n_induced = np.array([dy / r, -dx / r])
            # n_induced_array[i] = n_induced  # Store the induced normal vector

            # Update the influence matrix
            AN[i, j] = np.dot(n_vect[i,:].T, AN_3d[:,i,j]) 

    AN[-1, :] = 0  # Boundary condition enforcement
    AN[-1, 0] = 1
    AN[-1, -1] = 1

    return AN, n_induced_array

def compute_AN_3d_matrix(N, x, y, x_mid, y_mid):
    """
    Compute the 3D influence coefficient matrix AN_3d.

    Parameters:
    -----------
    N : int
        Number of panels.
    x, y : ndarray
        x and y coordinates of panel endpoints.
    x_mid, y_mid : ndarray
        x and y coordinates of panel midpoints.

    Returns:
    --------
    AN_3d : ndarray
        3D influence coefficient matrix of shape (2, N+1, N).
    """
    AN_3d = np.zeros((2, N , N + 1))

    for i in range(N):
        for j in range(N + 1):
            dx = x_mid[i] - x[j]
            dy = y_mid[i] - y[j]
            r = np.sqrt(dx**2 + dy**2)

            if np.isclose(r, 0):  # Avoid division by zero
                continue


            n_induced = np.array([dy / r, -dx / r])
            AN_3d[:, i, j] = n_induced / (2 * np.pi * r)

    return AN_3d

# Unit Tests
def test_compute_AN_matrix():
    N, R = 10, 1.0
    x, y = cf.cylinder(N, R)
    x_mid, y_mid, theta_mid, t_vect, n_vect, x,y = cf.pre_processing(x, y)

    AN,_ = compute_AN_matrix(N, x, y, x_mid, y_mid, n_vect)

    assert AN.shape == (N + 1, N + 1), "AN matrix shape is incorrect."
    assert np.all(np.isfinite(AN)), "AN matrix contains NaN or Inf values."
    assert np.isclose(AN[-1, 0], 1) and np.isclose(AN[-1, -1], 1), "Boundary conditions not enforced correctly."


def test_compute_AN_3d_matrix():
    N, R = 10, 1.0
    x, y = cf.cylinder(N, R)
    x_mid, y_mid, _, _, _, x,y = cf.pre_processing(x, y)

    AN_3d = compute_AN_3d_matrix(N, x, y, x_mid, y_mid)

    assert AN_3d.shape == (2, N , N+1), "AN_3d matrix shape is incorrect."
    assert np.all(np.isfinite(AN_3d)), "AN_3d matrix contains NaN or Inf values."

    # Check that normal vector components are small at large distances
    assert np.all(np.abs(AN_3d) < 1), "AN_3d contains unexpected large values."