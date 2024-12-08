import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Damped sinusoid function
def damped_sinusoid(r, A_ds, k, phi, tau):
    return A_ds * np.sin(k * r + phi) * np.exp(-r / tau)

# Sum of N damped sinusoids
def sum_damped_sinusoids(r, *params):
    N = len(params) // 4  # Number of damped sinusoids
    result = np.zeros_like(r)
    for i in range(N):
        A_ds = params[4 * i]
        k = params[4 * i + 1]
        phi = params[4 * i + 2]
        tau = params[4 * i + 3]
        result += damped_sinusoid(r, A_ds, k, phi, tau)
    return result

def process_image(image, center_x, center_y):
    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Flatten the image and radii
    image_flat = image.flatten()
    radii_flat = radii.flatten()

    # Define max_radius as maximum radius to consider
    max_radius = int(np.min(image.shape) / np.sqrt(2) - 10)
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

    # Fit a sum of damped sinusoids to the radial medians
    N_sinusoids = 6 # Number of damped sinusoids to sum
    initial_guess_ds = []
    for i in range(N_sinusoids):
        # Estimate initial parameters based on radial medians
        A_ds = (np.max(radial_medians_filtered) - np.min(radial_medians_filtered)) / N_sinusoids
        k = 2 * np.pi / (50 * (i + 1))  # Adjust the periodicity
        phi = np.pi * i / N_sinusoids
        tau = np.max(radial_medians_filtered)  # Adjust the damping factor
        initial_guess_ds.extend([A_ds, k, phi, tau])

    # Fit the sum of damped sinusoids to the radial medians
    try:
        popt_ds, _ = curve_fit(
            sum_damped_sinusoids,
            radial_distances_filtered,
            radial_medians_filtered,
            p0=initial_guess_ds,
            sigma=radial_stds_filtered,
            absolute_sigma=True,
            maxfev=100000
        )
        print("Damped sinusoids fitting successful.")
    except RuntimeError as e:
        print("Damped sinusoids curve fitting failed:", e)
        popt_ds = initial_guess_ds  # Use initial guess if fitting fails

    # Compute residuals after damped sinusoids fit
    #residuals_ds = radial_medians_filtered - sum_damped_sinusoids(radial_distances_filtered, *popt_ds)

    # Evaluate the damped sinusoids model over the entire image
    background_ds = sum_damped_sinusoids(radii, *popt_ds)

    # Subtract the background from the original image
    corrected_image = image - background_ds  # Version without masking applied

    # Plotting results
    # plot_results(
    #     image,
    #     corrected_image,
    #     radial_distances_filtered,
    #     radial_medians_filtered,
    #     popt_ds,
    #     residuals_ds,
    #     N_sinusoids
    # )

    return corrected_image#, corrected_image, radial_distances_filtered, radial_medians_filtered, popt_ds

def plot_results(image, corrected_image, radial_distances, radial_medians, popt_ds, residuals_ds, N_sinusoids):
    # Determine the brightness scale based on the original image
    vmin = np.nanmin(image)
    vmax = np.nanmax(image)

    # Plot the original image
    plt.figure()
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Plot the background corrected image (no mask exclusion)
    plt.figure()
    plt.title('Background Corrected Image')
    plt.imshow(corrected_image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Plot the radial medians and the fitted model
    plt.figure()
    plt.title('Radial Intensity Profile with Damped Sinusoids Fit')
    plt.plot(radial_distances, radial_medians, 'bo', label='Radial medians')
    r_fit = np.linspace(0, max(radial_distances), 2000)
    fitted_ds = sum_damped_sinusoids(r_fit, *popt_ds)
    plt.plot(r_fit, fitted_ds, 'g-', label='Damped Sinusoids Fit')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('median Intensity')
    plt.legend()

    # Plot the residuals after damped sinusoids fit
    plt.figure()
    plt.title('Residuals after Damped Sinusoids Fit')
    plt.plot(radial_distances, residuals_ds, 'ro', label='Residuals')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()

    # Plot each damped sinusoid component
    plt.figure()
    plt.title('Damped Sinusoid Components')
    for i in range(N_sinusoids):
        A_ds = popt_ds[4 * i]
        k = popt_ds[4 * i + 1]
        phi = popt_ds[4 * i + 2]
        tau = popt_ds[4 * i + 3]
        ds_component = damped_sinusoid(r_fit, A_ds, k, phi, tau)
        plt.plot(r_fit, ds_component, label=f'Damped Sinusoid {i+1}')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Intensity')
    plt.legend()

    # Show all plots simultaneously
    plt.show()
