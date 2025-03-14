import numpy as np

def compute_influence_coefficients(N, x, y, x_mid, y_mid, n_vect, t_vect):
    AN = np.zeros((len(x), len(x)))
    AN_3d = np.zeros((2, len(x), len(x_mid)))
    t_induced_array = np.zeros((len(x_mid), 2, len(x)))  # Array to store t_induced values
    for i in range(len(x_mid)):
        for j in range(len(x)):
            dx = x_mid[i] - x[j]
            dy = y_mid[i] - y[j]
            r = np.sqrt(dx**2 + dy**2)
            t_induced = np.array([dx / r, dy / r])
            n_induced = np.array([dy/r , -dx /r])
            # print('|n_vect|, should give one:', np.sqrt(n_induced[0]**2 + n_induced[1]**2))
            # assert np.dot(t_induced, n_induced) == 0
            AN[i, j] = -np.dot(n_vect[i], n_induced) / (2 * np.pi * r)
            AN_3d[0,j,i] = -np.dot(n_vect[i], n_induced)/(2*np.pi*r)
            AN_3d[1,j,i] = -np.dot(t_vect[i], t_induced)/(2*np.pi*r)
            # AN_3d[:,j,i] = -n_induced/(2*np.pi*r)
    AN[-1, :] = 0
    AN[-1, 0] = 1
    AN[-1, -1] = 1
    return AN, AN_3d

def compute_influence_coefficients_2(N, x, y, x_mid, y_mid, n_vect, t_vect):
    AN = np.zeros((len(x), len(x)))
    AN_3d = np.zeros((2, len(x), len(x_mid)))
    t_induced_array = np.zeros((len(x_mid), 2, len(x)))  # Array to store t_induced values

    for i in range(len(x_mid)):
        for j in range(len(x)):
            # Calculate the distance between x_mid[i], y_mid[i] and x[j], y[j]
            dx = x_mid[i] - x[j]
            dy = y_mid[i] - y[j]
            r = np.sqrt(dx**2 + dy**2)
            
            # Calculate the normal and tangential influence coefficients
            
            # Normal vector at midpoint, pointing radially outward
            magnitude = np.sqrt(x_mid[i]**2 + y_mid[i]**2)
            normal_vector = np.array([x_mid[i], y_mid[i]]) / magnitude  # Unit normal vector
            
            # Tangential vector at midpoint, perpendicular to the radius
            t_vect_induced = np.array([-dy, dx])  # Rotate by 90 degrees
            t_vect_induced /= np.linalg.norm(t_vect_induced)  # Normalize the tangential vector
            
            # Example of using the tangential and normal vectors to compute influence coefficients
            # Here we assume we're calculating the influence of tangential velocity:
            t_induced_array[i, :, j] = t_vect_induced  # Store the induced tangential vectors
            
            # You can extend this by adding specific aerodynamic models for influence coefficients.
            # For now, we're just assigning them to the array.
            AN_3d[0, j, i] = np.dot(n_vect[i], normal_vector)/(2*np.pi*r)  # Normal influence coefficient (example)
            AN_3d[1, j, i] = np.dot(t_vect[i], t_vect_induced)/(2*np.pi*r)  # Tangential influence coefficient (example)

    return AN_3d, t_induced_array