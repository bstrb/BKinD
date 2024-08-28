# normalize_intensities.py

import numpy as np

def normalize_intensities(self, inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values):
    """Normalize intensity values based on the selected normalization method."""
    normalization_method = self.normalization_var.get()

    if normalization_method == "sum":
        # Normalize the intensities to their respective sums
        inside_intensity_values /= np.sum(inside_intensity_values)
        outside_intensity_values /= np.sum(outside_intensity_values)
        total_intensity_values /= np.sum(total_intensity_values)
        absolute_difference_values /= np.sum(absolute_difference_values)

    elif normalization_method == "min_max":
        # Min-Max Normalization to [0, 1]
        def min_max_norm(arr):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        inside_intensity_values[:] = min_max_norm(inside_intensity_values)
        outside_intensity_values[:] = min_max_norm(outside_intensity_values)
        total_intensity_values[:] = min_max_norm(total_intensity_values)
        absolute_difference_values[:] = min_max_norm(absolute_difference_values)

    elif normalization_method == "z_score":
        # Z-Score Normalization
        def z_score_norm(arr):
            return (arr - np.mean(arr)) / np.std(arr)

        inside_intensity_values[:] = z_score_norm(inside_intensity_values)
        outside_intensity_values[:] = z_score_norm(outside_intensity_values)
        total_intensity_values[:] = z_score_norm(total_intensity_values)
        absolute_difference_values[:] = z_score_norm(absolute_difference_values)

    elif normalization_method == "total":
        # Normalization relative to Total Intensity
        inside_intensity_values /= total_intensity_values
        outside_intensity_values /= total_intensity_values
        absolute_difference_values /= total_intensity_values

    elif normalization_method == "log":
        # Log Transformation
        inside_intensity_values[:] = np.log(inside_intensity_values + 1)  # Adding 1 to avoid log(0)
        outside_intensity_values[:] = np.log(outside_intensity_values + 1)
        total_intensity_values[:] = np.log(total_intensity_values + 1)
        absolute_difference_values[:] = np.log(absolute_difference_values + 1)

    else:
        raise ValueError("Invalid normalization method selected!")