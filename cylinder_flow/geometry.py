import numpy as np

def define_geometry(N, R):
    ds = 2 * np.pi * R / N
    theta = np.linspace(0 +1e-3, 2 * np.pi +1e-3, N+1)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    ds_exact = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    x_mid = (x[:-1] + x[1:]) / 2
    y_mid = (y[:-1] + y[1:]) / 2
    theta_mid = np.arctan2(y_mid, x_mid)
    R_pan = np.sqrt(x_mid**2 + y_mid**2)
    # t_vect = np.column_stack((x_mid / R_pan, y_mid / R_pan))
    # n_vect = np.column_stack((t_vect[:, 1], -t_vect[:, 0]))

    # Panel normal vectors
    dx = np.diff(x)
    x_mid = dx/2
    dy = np.diff(y)
    y_mid = dy/2
    lengths = np.sqrt(dx**2 + dy**2)
    theta_mid = np.arctan2(dy,dx)
    n_vect = np.array([dy, -dx]) / lengths
    n_vect = n_vect.T
    t_vect = np.array([dx, dy]) / lengths
    t_vect = t_vect.T

    return x, y, x_mid, y_mid, theta_mid, t_vect, n_vect



def naca_4digit(N, m=0, p=0, t=12):
    """
    Generate the coordinates of a NACA 4-digit airfoil.
    
    Parameters:
        N (int): Number of points along the airfoil surface.
        m (float): Maximum camber.
        p (float): Location of maximum camber.
        t (float): Maximum thickness as a percentage of chord.
    
    Returns:
        x_airfoil (ndarray): x-coordinates of the airfoil.
        y_airfoil (ndarray): y-coordinates of the airfoil.
    """
    x = np.linspace(0, 1, N//2 + 1) 
    yt = 5 * t / 100 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line and its derivative
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if p != 0:
        idx1 = x < p
        idx2 = ~idx1
        yc[idx1] = m / p**2 * (2 * p * x[idx1] - x[idx1]**2)
        yc[idx2] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[idx2] - x[idx2]**2)
        dyc_dx[idx1] = 2 * m / p**2 * (p - x[idx1])
        dyc_dx[idx2] = 2 * m / (1 - p)**2 * (p - x[idx2])
    
    theta = np.arctan(dyc_dx)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    
    x_airfoil = np.concatenate([xu[::-1], xl[1:]])
    y_airfoil = np.concatenate([yu[::-1], yl[1:]])
    
    return x_airfoil, y_airfoil

def define_airfoil_geometry(N, airfoil_type="NACA2421"):
    """
    Define the geometry of a NACA 4-digit airfoil and compute panel properties.
    
    Parameters:
        N (int): Number of panels.
        airfoil_type (str): NACA 4-digit airfoil designation (e.g., "NACA0012").
    
    Returns:
        x_airfoil (ndarray): x-coordinates of the airfoil.
        y_airfoil (ndarray): y-coordinates of the airfoil.
        x_mid (ndarray): x-coordinates of panel midpoints.
        y_mid (ndarray): y-coordinates of panel midpoints.
        theta_mid (ndarray): Panel angles.
        t_vect (ndarray): Tangential vectors.
        n_vect (ndarray): Normal vectors.
    """
    m, p, t = 0, 0, 12  # Default to NACA0012
    if len(airfoil_type) == 6 and airfoil_type[:4] == "NACA":
        m = int(airfoil_type[4]) / 100
        p = int(airfoil_type[5]) / 10
        t = int(airfoil_type[6:])
    
    x_airfoil, y_airfoil = naca_4digit(N, m, p, t)
    
    # Panel midpoints
    dx = np.diff(x_airfoil)
    dy = np.diff(y_airfoil)
    x_mid = x_airfoil[:-1] + dx / 2
    y_mid = y_airfoil[:-1] + dy / 2
    
    # Panel lengths and angles
    lengths = np.sqrt(dx**2 + dy**2)
    theta_mid = np.arctan2(dy, dx)
    
    # Normal and tangential vectors
    n_vect = np.array([dy, -dx]) / lengths
    n_vect = n_vect.T
    t_vect = np.array([dx, dy]) / lengths
    t_vect = t_vect.T
    
    return x_airfoil, y_airfoil, x_mid, y_mid, theta_mid, t_vect, n_vect
