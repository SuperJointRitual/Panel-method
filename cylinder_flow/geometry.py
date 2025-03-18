import numpy as np

def define_geometry(N, R):
    ds = 2 * np.pi * R / N
    theta = np.linspace(0, 2 * np.pi, N+1)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    ds_exact = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    x_mid = (x[:-1] + x[1:]) / 2
    y_mid = (y[:-1] + y[1:]) / 2
    # theta_mid = np.arctan2(y_mid*2, x_mid*2)


    # Panel normal vectors
    dx = np.diff(x)
    dy = np.diff(y)
    theta_mid = np.arctan2(dy, dx)
    lengths = np.sqrt(dx**2 + dy**2)
    n_vect = np.array([dy, -dx]) / lengths
    n_vect = n_vect.T
    t_vect = np.array([dx, dy]) / lengths
    t_vect = t_vect.T

    return x, y, x_mid, y_mid, theta_mid, t_vect, n_vect



import numpy as np

def define_airfoil_geometry(n, m=0.02, p=0.4, tc=0.12):
    """
    Generates airfoil geometry using NACA-style thickness and camber line.
    NACA0015 - to implement. 

    Parameters:
    n  (int)  : Total number of points (including TE closure).
    m  (float): Maximum camber (e.g., 0.02 for 2%).
    p  (float): Location of max camber (e.g., 0.4 means 40% chord).
    tc (float): Maximum thickness (e.g., 0.12 for 12% thickness).

    Returns:
    tuple: x, y, x_mid, y_mid, theta_mid, t_vect, n_vect
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    # Cosine-spaced x-coordinates (full loop around airfoil)
    beta = np.linspace(0, 2*np.pi, n+1, endpoint=False)  # Closed-loop distribution
    xc = 0.5 * (1 + np.cos(beta))  # Chord normalized to 1

    # Thickness distribution (NACA 4-digit formula)
    a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1015
    yt = tc * (a0 * np.sqrt(xc) + a1 * xc + a2 * xc**2 + a3 * xc**3 + a4 * xc**4)

    # Initialize camber line
    yc = np.zeros_like(xc)
    dycdx = np.zeros_like(xc)

    # Compute camber line and its slope
    for i in range(len(xc)):
        if xc[i] < p:
            yc[i] = (m / p**2) * (2 * p * xc[i] - xc[i]**2)
            dycdx[i] = (2 * m / p**2) * (p - xc[i])
        else:
            yc[i] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*xc[i] - xc[i]**2)
            dycdx[i] = (2 * m / (1 - p)**2) * (p - xc[i])

    # Compute panel inclination
    theta = np.arctan(dycdx)

    # Generate airfoil coordinates
    x = xc - yt * np.sin(theta)
    y = yc + yt * np.cos(theta)

    # Ensure exactly `n` points
    if len(x) != n+1:
        raise RuntimeError("Generated airfoil does not have exactly n points.")

    # Compute panel midpoints
    x_mid = (x[:-1] + x[1:]) / 2
    y_mid = (y[:-1] + y[1:]) / 2

    # Compute panel angles
    theta_mid = np.arctan2(np.diff(y), np.diff(x))
    R_pan = np.sqrt(x_mid**2 + y_mid**2)
    # Compute tangent and normal vectors
    t_vect = np.column_stack((x_mid / R_pan, y_mid / R_pan))

    n_vect = np.column_stack((-t_vect[:, 1], t_vect[:, 0]))  # Rotate 90 degrees

    return x, y, x_mid, y_mid, theta_mid, t_vect, n_vect



