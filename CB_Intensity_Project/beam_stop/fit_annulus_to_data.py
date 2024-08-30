# fit_annulus_to_data.py

import numpy as np
from scipy.optimize import minimize

def objective_function(params, data):
    """Objective function to minimize: difference between data and the annular model."""
    x0, y0, inner_radius, outer_radius = params
    mask = annular_mask(data.shape, (x0, y0), inner_radius, outer_radius)
    masked_data = data[mask]
    
    # Handle potential overflows by clipping large values
    masked_data = np.clip(masked_data, -1e10, 1e10)
    
    # Avoid overflow in summation
    result = -np.sum(masked_data)
    
    # Return a large value if result is nan or inf
    if not np.isfinite(result):
        return 1e10
    
    return result

def annular_mask(shape, center, inner_radius, outer_radius):
    """Create a binary mask with an annular region."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    return mask

def fit_annulus_to_data(self, img_data):
    """Fit an annular region to the data."""
    initial_guess = (img_data.shape[1] / 2, img_data.shape[0] / 2, 10, 50)  # Example initial guess
    bounds = [(0, img_data.shape[1]), (0, img_data.shape[0]), (0, None), (0, None)]

    result = minimize(objective_function, initial_guess, args=(img_data,), bounds=bounds)
    self.center = (result.x[0], result.x[1])
    self.inner_radius = result.x[2]
    self.outer_radius = result.x[3]
