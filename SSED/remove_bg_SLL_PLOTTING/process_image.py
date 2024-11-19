import numpy as np
import matplotlib.pyplot as plt

from compute_radial_statistics import compute_radial_statistics

def process_image(image, center_x, center_y):
    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Flatten the image and radii
    image_flat = image.flatten()
    radii_flat = radii.flatten()

    # Define max_radius as maximum radius to consider
    max_radius = int(np.min(image.shape) / np.sqrt(2) - 10)

    # Filter data within max_radius
    within_limit_mask = radii_flat <= max_radius
    radii_filtered = radii_flat[within_limit_mask]
    image_filtered = image_flat[within_limit_mask]

    # Bin the data to compute median intensity in each bin
    num_bins = int(max_radius)
    bins = np.linspace(0, max_radius, num_bins)

    # Compute radial statistics
    radial_medians, radial_stds, radial_distances = compute_radial_statistics(
        radii_filtered, image_filtered, bins)

    # Reverse arrays to start from max_radius moving towards the center
    radial_medians_rev = radial_medians[::-1]
    radial_distances_rev = radial_distances[::-1]
    radial_stds_rev = radial_stds[::-1]

    # Compute gradient to detect sharp intensity drop due to beam stopper
    gradient_rev = np.gradient(radial_medians_rev)

    # Dynamic drop threshold based on data statistics
    gradient_median = np.median(gradient_rev)
    gradient_std = np.std(gradient_rev)
    drop_threshold = gradient_median - 1 * gradient_std  # Threshold at 1 standard deviation below median

    # Find indices where gradient drops below the threshold
    drop_indices = np.where(gradient_rev <= drop_threshold)[0]

    if len(drop_indices) > 0:
        # Exclude points where intensity drops sharply
        exclude_index = drop_indices[0]
        radial_medians_rev = radial_medians_rev[:exclude_index]
        radial_distances_rev = radial_distances_rev[:exclude_index]
        radial_stds_rev = radial_stds_rev[:exclude_index]

    # Reverse back to correct order
    radial_medians_filtered = radial_medians_rev[::-1]
    radial_distances_filtered = radial_distances_rev[::-1]
    radial_stds_filtered = radial_stds_rev[::-1]

    # Exclude additional points with lowest radius (closest to the center)
    ex_points = 0 # Adjust this number as needed
    radial_medians_filtered = radial_medians_filtered[ex_points:]
    radial_distances_filtered = radial_distances_filtered[ex_points:]
    radial_stds_filtered = radial_stds_filtered[ex_points:]

    # Plot the radial means
    plt.figure()
    plt.errorbar(radial_distances_filtered, radial_medians_filtered, yerr=radial_stds_filtered, fmt='o', ecolor='r', capsize=2)
    plt.xlabel('Radius')
    plt.ylabel('Median Intensity')
    plt.title('Radial Median Intensity Profile')
    plt.grid(True)
    plt.show()

    return image
