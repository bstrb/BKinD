# annular_model.py

import numpy as np
from scipy.optimize import minimize

def annular_mask(center, shape, outer_radius, inner_radius):
    y, x = np.indices(shape)
    distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = (distance_from_center <= outer_radius) & (distance_from_center > inner_radius)
    return mask

def objective_function(params, img_data):
    center_x, center_y, outer_radius, inner_radius = params
    center = (center_x, center_y)
    mask = annular_mask(center, img_data.shape, outer_radius, inner_radius)
    intensity_sum = np.sum(img_data[mask])
    return -intensity_sum  # Negative for minimization

def fit_annulus_to_data(img_data):
    # Initial guesses: center at the image center, outer and inner radius
    initial_center = (img_data.shape[1] / 2, img_data.shape[0] / 2)
    initial_outer_radius = min(img_data.shape) / 3
    initial_inner_radius = initial_outer_radius / 2

    # Bounds for the parameters: center within the image, positive radii
    bounds = [
        (0, img_data.shape[1]),  # center_x
        (0, img_data.shape[0]),  # center_y
        (0, min(img_data.shape)),  # outer_radius
        (0, min(img_data.shape) / 2)  # inner_radius
    ]

    # Perform the optimization
    result = minimize(objective_function, [*initial_center, initial_outer_radius, initial_inner_radius], 
                      args=(img_data,), bounds=bounds)

    # Extract the optimal parameters
    best_center_x, best_center_y, best_outer_radius, best_inner_radius = result.x
    return (best_center_x, best_center_y), best_outer_radius, best_inner_radius

def calculate_annular_region_intensity(self, img_data, outer_radius, inner_radius):
    """Calculate the sum of intensities in an annular region."""
    
    # Get the image dimensions and generate a grid of coordinates
    y, x = np.indices(img_data.shape)
    
    # Calculate the distance of each pixel from the center
    distance_from_center = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
    
    # Create a mask for the annular region (between the inner and outer radii)
    annular_mask = (distance_from_center <= outer_radius) & (distance_from_center > inner_radius)
    
    # Sum the intensities within the masked region
    intensity_sum = np.sum(img_data[annular_mask])
    
    return intensity_sum
