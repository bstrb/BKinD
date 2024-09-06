# fit_gaussian.py

import numpy as np
from scipy.optimize import curve_fit

from gaussian_2d import gaussian_2d

def fit_gaussian(img_data):
    """Fit a 2D Gaussian to the image data to find the center and spread."""
    x = np.arange(img_data.shape[1])
    y = np.arange(img_data.shape[0])
    x, y = np.meshgrid(x, y)
    
    # Initial guess: center of the image, reasonable sigmas, max amplitude, and min offset
    initial_guess = (img_data.shape[1] // 2, img_data.shape[0] // 2, 10, 10, np.max(img_data), np.min(img_data))
    
    # Perform the curve fitting using the gaussian_2d function
    params, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), img_data.ravel(), p0=initial_guess)
    
    # Extract the fitted parameters
    x0, y0, sigma_x, sigma_y, amplitude, offset = params
    
    # Store or return the relevant information
    center = (x0, y0)
    return center, sigma_x, sigma_y, amplitude, offset
