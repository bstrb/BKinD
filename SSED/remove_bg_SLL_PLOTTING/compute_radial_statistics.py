import numpy as np

def compute_radial_statistics(radii_filtered, image_filtered, bins):
    num_bins = len(bins)
    radial_medians = []
    radial_stds = []
    radial_distances = []
    for i in range(1, num_bins):
        bin_mask = (radii_filtered >= bins[i - 1]) & (radii_filtered < bins[i])
        if np.any(bin_mask):
            median_intensity = np.median(image_filtered[bin_mask])
            std_intensity = np.std(image_filtered[bin_mask])
            radial_medians.append(median_intensity)
            radial_stds.append(std_intensity)
            radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center
    return np.array(radial_medians), np.array(radial_stds), np.array(radial_distances)
