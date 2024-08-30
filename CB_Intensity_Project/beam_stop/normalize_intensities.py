# normalize_intensities.py

import numpy as np

def normalize_intensities(self, inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values):
    normalization_method = self.normalization_var.get()

    if normalization_method == "sum":
        # Normalize to their respective sums
        inside_intensity_values /= np.sum(inside_intensity_values)
        outside_intensity_values /= np.sum(outside_intensity_values)
        total_intensity_values /= np.sum(total_intensity_values)
        absolute_difference_values /= np.sum(absolute_difference_values)

    elif normalization_method == "min_max":
        # Min-Max Normalization to [0, 1]
        inside_intensity_values = (inside_intensity_values - np.min(inside_intensity_values)) / (np.max(inside_intensity_values) - np.min(inside_intensity_values))
        outside_intensity_values = (outside_intensity_values - np.min(outside_intensity_values)) / (np.max(outside_intensity_values) - np.min(outside_intensity_values))
        total_intensity_values = (total_intensity_values - np.min(total_intensity_values)) / (np.max(total_intensity_values) - np.min(total_intensity_values))
        absolute_difference_values = (absolute_difference_values - np.min(absolute_difference_values)) / (np.max(absolute_difference_values) - np.min(absolute_difference_values))

    elif normalization_method == "z_score":
        # Z-Score Normalization
        inside_intensity_values = (inside_intensity_values - np.mean(inside_intensity_values)) / np.std(inside_intensity_values)
        outside_intensity_values = (outside_intensity_values - np.mean(outside_intensity_values)) / np.std(outside_intensity_values)
        total_intensity_values = (total_intensity_values - np.mean(total_intensity_values)) / np.std(total_intensity_values)
        absolute_difference_values = (absolute_difference_values - np.mean(absolute_difference_values)) / np.std(absolute_difference_values)

    elif normalization_method == "total":
        # Normalize relative to Total Intensity
        inside_intensity_values /= total_intensity_values
        outside_intensity_values /= total_intensity_values
        absolute_difference_values /= total_intensity_values

    elif normalization_method == "log":
        # Log Transformation
        inside_intensity_values = np.log(inside_intensity_values + 1)  # Adding 1 to avoid log(0)
        outside_intensity_values = np.log(outside_intensity_values + 1)
        total_intensity_values = np.log(total_intensity_values + 1)
        absolute_difference_values = np.log(absolute_difference_values + 1)

    else:
        raise ValueError("Invalid normalization method selected!")
