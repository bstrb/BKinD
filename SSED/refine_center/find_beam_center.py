import numpy as np

def find_beam_center(x, y, h, k, l):
    """
    Finds the beam center (X_c, Y_c) and linear parameters relating (h,k,l) to (X,Y).

    Parameters
    ----------
    x, y : array-like
        Arrays of measured peak positions in pixels.
    h, k, l : array-like
        Arrays of corresponding Miller indices.

    Returns
    -------
    X_c, Y_c : float
        Estimated beam center coordinates in pixels.
    params : dict
        Dictionary containing the fitted parameters:
        {'X_c': X_c, 'A': A, 'B': B, 'C': C,
         'Y_c': Y_c, 'D': D, 'E': E, 'F': F}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    l = np.asarray(l, dtype=float)

    # Number of reflections
    N = len(x)
    if N < 4:
        raise ValueError("At least 4 peaks are required to solve for the parameters.")

    # We have the equations:
    # X_i = X_c + A*h_i + B*k_i + C*l_i
    # Y_i = Y_c + D*h_i + E*k_i + F*l_i
    #
    # We'll set up the system in the form:
    # P = M * theta, where:
    # P = [X_1, Y_1, X_2, Y_2, ..., X_N, Y_N]^T
    #
    # theta = [X_c, A, B, C, Y_c, D, E, F]^T
    #
    # M = a 2N x 8 matrix:
    # For each i:
    # Row(2i-1): [1, h_i, k_i, l_i, 0, 0, 0, 0]
    # Row(2i):   [0, 0, 0, 0, 1, h_i, k_i, l_i]

    P = np.zeros((2*N,))
    M = np.zeros((2*N, 8))

    # Fill P and M
    for i in range(N):
        P[2*i]   = x[i]
        P[2*i+1] = y[i]

        # For X: [X_c, A, B, C, Y_c, D, E, F]
        M[2*i, 0] = 1.0      # X_c
        M[2*i, 1] = h[i]     # A
        M[2*i, 2] = k[i]     # B
        M[2*i, 3] = l[i]     # C
        # Y parameters for X row are zero
        # M[2*i, 4] = 0
        # M[2*i, 5] = 0
        # M[2*i, 6] = 0
        # M[2*i, 7] = 0

        # For Y:
        M[2*i+1, 4] = 1.0    # Y_c
        M[2*i+1, 5] = h[i]   # D
        M[2*i+1, 6] = k[i]   # E
        M[2*i+1, 7] = l[i]   # F

    # Solve the least squares problem
    # theta = (M^T M)^{-1} M^T P
    theta, residuals, rank, s = np.linalg.lstsq(M, P, rcond=None)

    # Extract parameters
    X_c, A, B, C, Y_c, D, E, F = theta

    params = {
        'X_c': X_c,
        'A': A,
        'B': B,
        'C': C,
        'Y_c': Y_c,
        'D': D,
        'E': E,
        'F': F
    }

    return X_c, Y_c, params

# Example usage:
if __name__ == "__main__":
    # Example data from the earlier conversation:
    # x_data = [384.94, 678.29, 652.25, 519.45, 563.33]
    # y_data = [392.42, 617.98, 621.39, 685.25, 750.38]
    # h_data = [-5.0, 9.0, 7.0, -5.0, -4.0]
    # k_data = [-23.0, 23.0, 22.0, 23.0, 34.0]
    # l_data = [8.0, -14.0, -11.0, 6.0, 4.0]

    x_data = [384.94, 678.29, 652.25, 519.45]
    y_data = [392.42, 617.98, 621.39, 685.25]
    h_data = [-5.0, 9.0, 7.0, -5.0]
    k_data = [-23.0, 23.0, 22.0, 23.0]
    l_data = [8.0, -14.0, -11.0, 6.0]

    Xc, Yc, params = find_beam_center(x_data, y_data, h_data, k_data, l_data)
    print("Beam center estimated at X_c = {:.2f}, Y_c = {:.2f}".format(Xc, Yc))
    print("All parameters:", params)
