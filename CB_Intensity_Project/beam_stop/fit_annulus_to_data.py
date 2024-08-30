# fit_annulus_to_data.py

import numpy as np
from scipy.optimize import minimize

def annular_mask(shape, center, inner_radius, outer_radius):
    """Create a binary mask with an annular region."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    return mask

def circular_mask(shape, center, radius):
    """Create a binary mask with a circular region."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def objective_function(params, data):
    """Objective function to minimize: fit the inner circle and outer Gaussian tail."""
    x0, y0, inner_radius, amplitude, sigma = params

    # Create a mask for the inner circle (dark region)
    circle_mask = circular_mask(data.shape, (x0, y0), inner_radius)
    
    # Create a mask for the Gaussian tail (outer region)
    tail_mask = ~circle_mask  # everything outside the inner circle

    # Calculate the Gaussian model on the outer region
    y, x = np.indices(data.shape)
    gaussian_model = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Calculate the difference between the data and model in the outer region
    residual = data[tail_mask] - gaussian_model[tail_mask]

    # Return the sum of squared differences
    return np.sum(residual**2)

def fit_annulus_to_data(self, img_data):
    """Fit the inner circle to the dark area and the outer Gaussian tail to the beam spot."""
    # Initial guess for center, inner radius, amplitude of Gaussian, and sigma of Gaussian
    initial_guess = (img_data.shape[1] / 2, img_data.shape[0] / 2, 10, np.max(img_data), 50)
    bounds = [
        (0, img_data.shape[1]),  # x0 bounds
        (0, img_data.shape[0]),  # y0 bounds
        (0, None),  # inner_radius bounds
        (0, None),  # amplitude bounds
        (0, None)   # sigma bounds
    ]

    # Minimize the objective function
    result = minimize(objective_function, initial_guess, args=(img_data,), bounds=bounds)
    self.center = (result.x[0], result.x[1])
    self.inner_radius = result.x[2]
    self.amplitude = result.x[3]
    self.sigma = result.x[4]

    # print(f"Fitted Annulus: Center = {self.center}, Inner Radius = {self.inner_radius:.2f}, "
    #       f"Amplitude = {self.amplitude:.2f}, Sigma = {self.sigma:.2f}")

