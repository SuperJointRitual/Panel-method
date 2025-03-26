import numpy as np
import matplotlib.pyplot as plt

def cylinder(N, R):
    """
    Generate the x and y coordinates of a circular shape.

    Parameters:
    -----------
    N : int
        Number of panels (discretization points).
    R : float
        Radius of the cylinder.

    Returns:
    --------
    x : ndarray
        x-coordinates of panel endpoints.
    y : ndarray
        y-coordinates of panel endpoints.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if R <= 0:
        raise ValueError("R must be a positive number.")

    theta = np.linspace(0+1e-3, 2 * np.pi-1e-3, N + 1)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return x, y

def pre_processing(x, y):
    """
    Compute panel midpoints, panel angles, and normal/tangential vectors, ensuring the first vortex is not at the trailing edge.

    Parameters:
    -----------
    x : ndarray
        x-coordinates of panel endpoints.
    y : ndarray
        y-coordinates of panel endpoints.

    Returns:
    --------
    x_mid : ndarray3
        x-coordinates of panel midpoints.
    y_mid : ndarray
        y-coordinates of panel midpoints.
    theta_mid : ndarray
        Angle of each panel midpoint relative to the origin.
    t_vect : ndarray
        Tangential unit vectors for each panel.
    n_vect : ndarray
        Normal unit vectors for each panel.
    """
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length.")
    if len(x) < 2:
        raise ValueError("At least two points are required to define panels.")

    # Compute dx and dy for the first panel
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Check if the first point is at the trailing edge (same as last point)
    if np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1]):
        # Move the points by half dx counterclockwise
        r = np.sqrt(dx**2 + dy**2)
        if r == 0:
            return x, y, None, None, None  # Avoid division by zero

        # Tangential unit vector of the first panel
        t_vect = np.array([dx/r, dy/r])

        # Normal unit vector (perpendicular to tangential, counterclockwise)
        n_vect = np.array([dy/r, -dx/r])

        # Displacement vector: move points by half the panel length counterclockwise
        dx_shift = 0.5 * dx
        dy_shift = 0.5 * dy

        # # Shift all points
        # x += dx_shift
        # y += dy_shift
        # Check for unwanted points
        if np.isclose(x[0], 1) and np.isclose(y[0], 0) or np.isclose(x[-1], 1) and np.isclose(y[-1], 0):
            raise ValueError("After shifting, the first point is at (1, 0). This is not expected.")


    # Compute midpoints and other quantities
    x_mid = (x[:-1] + x[1:]) / 2
    y_mid = (y[:-1] + y[1:]) / 2
    dx = np.diff(x)
    dy = np.diff(y)
    theta_mid = np.arctan2(dy, dx)
    r = np.sqrt(dx**2 + dy**2)

    # Tangential and normal vectors
    t_vect = np.array([dx/r, dy/r]).T  
    n_vect = np.array([dy/r, -dx/r]).T  

    return x_mid, y_mid, theta_mid, t_vect, n_vect, x, y



def define_airfoil_geometry(N, m, p, tc):
    # Geometrical coefficients
    a0 = 1.4845
    a1 = -0.6300
    a2 = -1.7580
    a3 = 1.4215
    a4 = -0.5075
    
    # Generation of airfoil contour; theta goes from 0 to 2*pi
    dt = (2 * np.pi) / N
    t = np.linspace(0, 2 * np.pi - dt, N+1)  # Ensure no duplication at 2*pi
    xc = 0.5 * (1 + np.cos(t))
    yt = tc * (a0 * np.sqrt(xc) + a1 * xc + a2 * xc**2 + a3 * xc**3 + a4 * xc**4)
    
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    yc = np.zeros(N+1)

    for i in range(N+1):
        if t[i] <= np.pi:  # Upper side
            if xc[i] >= p:
                yc[i] = m / (1 - p)**2 * (1 - 2*p + 2*p*xc[i] - xc[i]**2)
                dycdx = 2 * m / (1 - p)**2 * (p - xc[i])
            else:
                yc[i] = m / p**2 * (2*p*xc[i] - xc[i]**2)
                dycdx = 2 * m / p**2 * (p - xc[i])
            theta = np.arctan(dycdx)
            x[i] = xc[i] - yt[i] * np.sin(theta)
            y[i] = yc[i] + yt[i] * np.cos(theta)
        else:  # Lower side
            if xc[i] < p:
                yc[i] = m / p**2 * (2*p*xc[i] - xc[i]**2)
                dycdx = 2 * m / p**2 * (p - xc[i])
            else:
                yc[i] = m / (1 - p)**2 * (1 - 2*p + 2*p*xc[i] - xc[i]**2)
                dycdx = 2 * m / (1 - p)**2 * (p - xc[i])
            theta = np.arctan(dycdx)
            x[i] = xc[i] + yt[i] * np.sin(theta)
            y[i] = yc[i] - yt[i] * np.cos(theta)

    print("Shape of x:", np.shape(x))
    
    # Plot of airfoil
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title(f'Geometry of NACA {int(m*100)}{int(p*10)}{int(tc*100)} Airfoil')
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Compute panel midpoints, angles, and tangential/normal vectors
    x_mid = (x[:-1] + x[1:]) / 2
    y_mid = (y[:-1] + y[1:]) / 2
    dx = np.diff(x)
    dy = np.diff(y)
    theta_mid = np.arctan2(dy, dx)
    r = np.sqrt(dx**2 + dy**2)

    # Tangential and normal vectors
    t_vect = np.array([dx / r, dy / r]).T  
    n_vect = np.array([dy / r, -dx / r]).T  

    return x_mid, y_mid, theta_mid, t_vect, n_vect, x, y






# Unit Tests
def test_cylinder():
    N, R = 4, 1.0
    x, y = cylinder(N, R)

    assert len(x) == N + 1  # Should return N+1 points
    assert len(y) == N + 1
    assert np.allclose(np.sqrt(x**2 + y**2), R)  # Check all points lie on the circle

    # Test invalid inputs
    try:
        cylinder(0, R)
    except ValueError as e:
        assert str(e) == "N must be a positive integer."

    try:
        cylinder(N, -1)
    except ValueError as e:
        assert str(e) == "R must be a positive number."


def test_pre_processing():
    x, y = cylinder(10, 1.0)
    x_mid, y_mid, theta_mid, t_vect, n_vect, x,y = pre_processing(x, y)

    assert len(x_mid) == len(x) - 1
    assert len(y_mid) == len(y) - 1
    assert len(theta_mid) == len(x) - 1
    assert t_vect.shape == (len(x) - 1, 2)
    assert n_vect.shape == (len(x) - 1, 2)

    # Check that normal and tangential vectors are perpendicular
    dot_products = np.sum(t_vect * n_vect, axis=1)
    assert np.allclose(dot_products, 0, atol=1e-10), "Tangential and normal vectors should be orthogonal."

    # Check that both vectors have unit length
    t_norms = np.linalg.norm(t_vect, axis=1)
    n_norms = np.linalg.norm(n_vect, axis=1)
    assert np.allclose(t_norms, 1), "Tangential vectors should be unit vectors."
    assert np.allclose(n_norms, 1), "Normal vectors should be unit vectors."

    # Check that normal vectors point outward (for a circle)
    outward_check = np.sign(n_vect[:, 0] * x_mid + n_vect[:, 1] * y_mid)
    assert np.all(outward_check > 0), "Normal vectors should point outward."

    # Test invalid inputs
    try:
        pre_processing(np.array([1.0]), np.array([0.0]))
    except ValueError as e:
        assert str(e) == "At least two points are required to define panels."

    try:
        pre_processing(np.array([1.0, 0.0]), np.array([0.0]))
    except ValueError as e:
        assert str(e) == "x and y arrays must have the same length."