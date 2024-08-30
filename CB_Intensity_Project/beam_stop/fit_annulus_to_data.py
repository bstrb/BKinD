# fit_annulus_to_data.py

import numpy as np
from scipy.optimize import minimize

def dark_circle_mask(shape, center, radius):
    """Create a binary mask for the dark circular region."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def gaussian_tail_mask(shape, center, inner_radius):
    """Create a binary mask for the Gaussian tail region outside the dark circle."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    return dist_from_center > inner_radius

def objective_function(params, data):
    """Objective function to minimize: fits the dark circular region."""
    x0, y0, radius = params
    mask = dark_circle_mask(data.shape, (x0, y0), radius)
    return np.sum(data[mask])  # Minimize the sum of intensities inside the dark region

def fit_annulus_to_data(self, img_data):
    """Fit the dark circular region first, then fit a 2D Gaussian to the region outside."""
    
    # Initial guess for the dark circular region
    initial_guess = (img_data.shape[1] / 2, img_data.shape[0] / 2, 50)  # Example guess
    bounds = [(0, img_data.shape[1]), (0, img_data.shape[0]), (20, 100)]  # Constrain radius between 20 and 100

    # Fit the dark circle (beam stopper)
    result = minimize(objective_function, initial_guess, args=(img_data,), bounds=bounds)
    self.center = (result.x[0], result.x[1])
    self.inner_radius = result.x[2]

    print(f"Fitted Dark Circle: Center=({self.center[0]:.2f}, {self.center[1]:.2f}), Radius={self.inner_radius:.2f}")

    # Now fit the Gaussian tail outside the dark circle
    self.fit_gaussian_tail(img_data, self.center, self.inner_radius)
