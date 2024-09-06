# fit_pseudo_voigt.py

import numpy as np
from scipy.optimize import curve_fit
from pseudo_voigt_2d import pseudo_voigt_2d

def fit_pseudo_voigt(data):
    """Fit a 2D pseudo-Voigt to the image data to find the center and spread."""
    # Define the coordinates for fitting
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    
    # Initial guess: center of the image, reasonable sigmas, eta, max amplitude
    initial_guess = (data.shape[1] // 2, data.shape[0] // 2, 1, 1, 0, np.max(data))
    
    # Flatten data for fitting
    x_flat = x.ravel()
    y_flat = y.ravel()
    data_flat = data.ravel()
    
    # Set bounds for the parameters
    # x0 and y0: within the image range
    # sigma_x and sigma_y: positive, reasonable guess for spread
    # eta: between 0 and 1
    # amplitude: positive, constrained by the max data value
    bounds = (
        (0, 0, 0.1, 0.1, 0, 0), # Lower bounds: x0, y0, sigma_x, sigma_y, eta, amplitude
        (data.shape[1], data.shape[0], np.inf, np.inf, 1, np.max(data) * 1.5)  # Upper bounds: x0, y0, sigma_x, sigma_y, eta, amplitude
    )
    
    # Perform the curve fitting using the pseudo_voigt_2d function with bounds
    params, _ = curve_fit(
        lambda xy, x0, y0, sigma_x, sigma_y, eta, amplitude: pseudo_voigt_2d(xy[0], xy[1], x0, y0, sigma_x, sigma_y, eta, amplitude),
        (x_flat, y_flat), data_flat, p0=initial_guess, bounds=bounds
    )

    # Extract the fitted parameters
    x0, y0, sigma_x, sigma_y, eta, amplitude = params
    
    # Store or return the relevant information
    center = (x0, y0)
    return center, sigma_x, sigma_y, eta, amplitude
