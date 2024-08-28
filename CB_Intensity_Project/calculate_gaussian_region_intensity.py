# calculate_gaussian_region_intensity.py

import numpy as np

def calculate_gaussian_region_intensity(self, img_data, sigma_level=2, region="inside"):
    """Calculate the sum of intensity inside or outside the central Gaussian region within a specified sigma level."""
    row, column = self.center
    Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
    
    # Create a Gaussian mask based on the specified region
    if region == "inside":
        mask = (((X - column)**2 / self.sigma_x**2) + ((Y - row)**2 / self.sigma_y**2)) <= sigma_level**2
    elif region == "outside":
        mask = (((X - column)**2 / self.sigma_x**2) + ((Y - row)**2 / self.sigma_y**2)) > sigma_level**2
    else:
        raise ValueError("Invalid region specified. Choose 'inside' or 'outside'.")

    # Calculate the sum of intensity in the specified region
    intensity = np.sum(img_data[mask])
    return intensity
