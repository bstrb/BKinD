# calculate_annular_region_intensity.py

import numpy as np

def calculate_annular_region_intensity(self, img_data, region="inside"):
    """Calculate the intensity inside or outside the annular region."""
    y, x = np.indices(img_data.shape)
    dist_from_center = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
    circle_mask = dist_from_center <= self.inner_radius

    if region == "inside":
        return np.sum(img_data[circle_mask])
    elif region == "outside":
        gaussian_mask = ~circle_mask
        return np.sum(img_data[gaussian_mask])
    else:
        raise ValueError("Invalid region specified: choose 'inside' or 'outside'")
