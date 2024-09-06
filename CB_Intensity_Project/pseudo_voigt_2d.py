# pseudo_voigt_2d.py

import numpy as np

def pseudo_voigt_2d(x, y, x0, y0, sigma_x, sigma_y, eta, amplitude):
    """
    Compute the 2D pseudo-Voigt profile.
    
    Parameters:
    - x, y: Coordinates over which to calculate the profile.
    - x0, y0: Center of the peak.
    - sigma_x, sigma_y: Standard deviations of the Gaussian/Lorentzian along the x and y axes.
    - eta: Mixing parameter (0 for Gaussian, 1 for Lorentzian).
    - amplitude: Peak intensity.
    
    Returns:
    - The value of the pseudo-Voigt function at (x, y).
    """
    # 2D Gaussian component
    gauss = np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

    # 2D Lorentzian component
    lorentz = 1 / ((1 + ((x - x0)**2 / sigma_x**2) + ((y - y0)**2 / sigma_y**2)))

    # Pseudo-Voigt as a linear combination of Gaussian and Lorentzian
    return amplitude * (eta * lorentz + (1 - eta) * gauss)
