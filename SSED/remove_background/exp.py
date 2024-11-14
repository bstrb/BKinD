import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the exponential function
def exponential(r, A, k, C):
    return A * np.exp(-k * r) + C

def process_image(image, center_x, center_y):
    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Flatten the image and radii
    image_flat = image.flatten()
    radii_flat = radii.flatten()

    # Define max_radius as maximum radius to consider
    max_radius = int(np.min(image.shape) / np.sqrt(2) - 5)
    print(f"Max radius considered: {max_radius}")

    # Filter data within max_radius
    within_limit_mask = radii_flat <= max_radius
    radii_filtered = radii_flat[within_limit_mask]
    image_filtered = image_flat[within_limit_mask]

    # Bin the data to compute median intensity in each bin
    num_bins = int(max_radius)
    bins = np.linspace(0, max_radius, num_bins)
    bin_indices = np.digitize(radii_filtered, bins)

    radial_medians = []
    radial_stds = []
    radial_distances = []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            median_intensity = np.median(image_filtered[bin_mask])
            std_intensity = np.std(image_filtered[bin_mask])
            radial_medians.append(median_intensity)
            radial_stds.append(std_intensity)
            radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center

    radial_medians = np.array(radial_medians)
    radial_stds = np.array(radial_stds)
    radial_distances = np.array(radial_distances)

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
    ex_points = 2  # Adjust this number as needed
    radial_medians_filtered = radial_medians_filtered[ex_points:]
    radial_distances_filtered = radial_distances_filtered[ex_points:]
    radial_stds_filtered = radial_stds_filtered[ex_points:]

    # Adjust initial guesses based on data
    A_initial = np.max(radial_medians_filtered) - np.min(radial_medians_filtered)
    k_initial = 1 / (np.max(radial_distances_filtered) - np.min(radial_distances_filtered))
    C_initial = np.min(radial_medians_filtered)

    initial_guess_exp = [A_initial, k_initial, C_initial]

    # Fit the exponential function
    try:
        popt_exp, _ = curve_fit(
            exponential,
            radial_distances_filtered,
            radial_medians_filtered,
            p0=initial_guess_exp,
            sigma=radial_stds_filtered,
            absolute_sigma=True,
            maxfev=50000
        )
        print("Exponential fitting successful.")
    except RuntimeError as e:
        print("Exponential curve fitting failed:", e)
        popt_exp = initial_guess_exp  # Use initial guess if fitting fails

    # Compute residuals
    fitted_exp = exponential(radial_distances_filtered, *popt_exp)
    residuals_exp = radial_medians_filtered - fitted_exp

    # Compute the background over the entire image
    background_exp = exponential(radii, *popt_exp)

    # Subtract background from the original image
    corrected_image = image - background_exp

    # Plotting results
    plot_results(
        image,
        corrected_image,
        radial_distances_filtered,
        radial_medians_filtered,
        popt_exp,
        residuals_exp
    )

    return corrected_image

def plot_results(image, corrected_image, radial_distances, radial_medians, popt_exp, residuals_exp):
    # Determine the brightness scale based on the original image
    vmin = np.nanmin(image)
    vmax = np.nanmax(image)

    # Plot the original image
    plt.figure()
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Plot the background-corrected image
    plt.figure()
    plt.title('Background Corrected Image')
    plt.imshow(corrected_image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Plot the radial medians and the fitted exponential curve
    plt.figure()
    plt.title('Radial Intensity Profile and Exponential Fit')
    plt.plot(radial_distances, radial_medians, 'bo', label='Radial medians')
    r_fit = np.linspace(min(radial_distances), max(radial_distances), 1000)
    fitted_exp_curve = exponential(r_fit, *popt_exp)
    plt.plot(r_fit, fitted_exp_curve, 'r-', label='Fitted Exponential')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Median Intensity')
    plt.legend()

    # Plot the residuals
    plt.figure()
    plt.title('Residuals after Exponential Fit')
    plt.plot(radial_distances, residuals_exp, 'ro', label='Residuals')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()

    plt.show()
