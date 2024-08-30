# calculate_annular_region_intensity.py

from fit_annulus_to_data import annular_mask

import numpy as np

def calculate_annular_region_intensity(self, img_data, region="inside"):
    """Calculate the intensity inside or outside the annular region."""
    mask = annular_mask(img_data.shape, self.center, self.inner_radius, self.outer_radius)
    
    if region == "inside":
        return np.sum(img_data[mask])
    elif region == "outside":
        return np.sum(img_data[~mask])
    else:
        raise ValueError("Invalid region specified: choose 'inside' or 'outside'")
