import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from compute_radial_statistics import compute_radial_statistics
import statsmodels.api as sm

def enhanced_super_lorentzian(r, A, mu, gamma, n, C, D):
    # Add an additional term D / (r + epsilon) to capture steepness near r = 0
    epsilon = 1e-6  # Small value to prevent division by zero
    return A / (1 + ((r - mu) / gamma) ** 2) ** n + C + D / (r + epsilon)

def process_image(image, center_x, center_y):
    # Validate input parameters
    if not (0 <= center_x < image.shape[1] and 0 <= center_y < image.shape[0]):
        raise ValueError("Center coordinates are out of image bounds")

    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Flatten the image and radii
    image_flat = image.flatten()
    radii_flat = radii.flatten()

    # Define max_radius as maximum radius to consider
    max_radius = int(np.min(image.shape) / np.sqrt(2)) - 10

    # Filter data within max_radius
    within_limit_mask = radii_flat <= max_radius
    radii_filtered = radii_flat[within_limit_mask]
    image_filtered = image_flat[within_limit_mask]

    # Bin the data to compute median intensity in each bin
    # Use finer bins near the center to capture steep gradient
    num_bins = int(max_radius)
    bins_center = np.linspace(0, 20, 100)  # Finer bins from 0 to 20 pixels
    bins_outer = np.linspace(20, max_radius, num_bins - 100)
    bins = np.concatenate((bins_center, bins_outer))

    # Compute radial statistics
    radial_medians, radial_stds, radial_distances = compute_radial_statistics(
        radii_filtered, image_filtered, bins)

    # Smooth the radial medians for gradient calculation
    radial_medians_smooth = gaussian_filter1d(radial_medians, sigma=2)

    # Compute gradient to detect sharp intensity drop due to beam stopper
    gradient = np.gradient(radial_medians_smooth)

    # Dynamic drop threshold based on data statistics
    gradient_median = np.median(gradient)
    gradient_std = np.std(gradient)
    drop_threshold = gradient_median - gradient_std  # Threshold at 1 standard deviation below median

    # Find index where gradient drops below the threshold
    drop_indices = np.where(gradient <= drop_threshold)[0]

    if len(drop_indices) > 0:
        # Exclude points inside the beam stopper region (after the sharp drop)
        include_index = drop_indices[0]
        radial_medians = radial_medians[include_index:]
        radial_distances = radial_distances[include_index:]
        radial_stds = radial_stds[include_index:]

    print(np.min(radial_stds))
    # Adjust initial guesses based on data
    A_initial = np.max(radial_medians)
    mu_initial = 0  # Centered at radius 0
    gamma_initial = np.min(radial_distances)
    n_initial = 2  # Start with a reasonable value
    C_initial = np.min(radial_medians)
    D_initial = A_initial * gamma_initial  # Initial guess for the additional term

    initial_guess = [A_initial, mu_initial, gamma_initial, n_initial, C_initial, D_initial]

    # Set parameter bounds to constrain the fitting
    bounds = (
        [0, -1, 0, 0, 0, -np.inf],  # Lower bounds
        [np.inf, 1, np.max(radial_distances), np.inf, np.max(radial_medians), np.inf]  # Upper bounds
    )

    # Create weights to emphasize low and high radius points
    # Normalize radial distances to range [0, 1]
    radial_distances_normalized = (radial_distances - np.min(radial_distances)) / (np.max(radial_distances) - np.min(radial_distances))

    # Compute weights: high at low and high radii, low in between
    weights = np.exp(-((radial_distances_normalized - 0) ** 2) / (2 * 0.1 ** 2))  # Low radius weights
    weights += np.exp(-((radial_distances_normalized - 1) ** 2) / (2 * 0.1 ** 2))  # High radius weights
    weights += 0.1  # Ensure minimum weight to avoid zero weights

    # Invert weights for use with sigma (since sigma ~ 1 / weight)
    sigma = 1 / weights
    # Fit the enhanced super-Lorentzian function
    try:
        popt, _ = curve_fit(
            enhanced_super_lorentzian,
            radial_distances,
            radial_medians,
            p0=initial_guess,
            sigma=sigma,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=100000
        )
        print("Enhanced Super-Lorentzian fitting successful.")
    except RuntimeError as e:
        print("Enhanced Super-Lorentzian curve fitting failed:", e)
        popt = initial_guess  # Use initial guess if fitting fails

    # Compute the background over the entire image
    background = enhanced_super_lorentzian(radii, *popt)

    # Subtract background from the original image
    # We'll update this later after including the LOWESS corrections
    # corrected_image = image - background

    # Plotting results
    plt.figure()
    plt.title('Radial Intensity Profile and Enhanced Super-Lorentzian Fit')
    plt.plot(radial_distances, radial_medians, 'bo', label='Radial Medians')
    r_fit = np.linspace(min(radial_distances), max(radial_distances), 1000)
    fitted_curve = enhanced_super_lorentzian(r_fit, *popt)
    plt.plot(r_fit, fitted_curve, 'r-', label='Enhanced Super-Lorentzian Fit')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Median Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Iteratively apply LOWESS to flatten the residuals
    residuals = radial_medians - enhanced_super_lorentzian(radial_distances, *popt)
    iteration = 0
    residual_threshold = np.min(radial_stds) / 2
    print(f"Residual threshold: {residual_threshold}")

    # Initialize total LOWESS correction
    total_lowess_correction = np.zeros_like(radial_medians)

    while True:
        lowess = sm.nonparametric.lowess(residuals, radial_distances, frac=0.05, delta=1)
        lowess_x = lowess[:, 0]
        lowess_y = lowess[:, 1]

        # Plot the LOWESS fit along with the residuals
        plt.figure()
        plt.title(f'Iteration {iteration}: Residuals with LOWESS Fit')
        plt.plot(radial_distances, residuals, 'ro', label='Residuals')
        plt.plot(lowess_x, lowess_y, 'b-', label='LOWESS Fit')
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Residual Intensity')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Accumulate the LOWESS correction
        lowess_correction = np.interp(radial_distances, lowess_x, lowess_y)
        total_lowess_correction += lowess_correction

        # Subtract the LOWESS fit from the residuals
        residuals -= lowess_correction

        # Check if the residuals are sufficiently flattened
        if np.max(abs(residuals)) < residual_threshold:
            print(f"Convergence achieved after {iteration} iterations.")
            break

        iteration += 1
        if iteration >= 10:
            print("Maximum number of iterations reached without full convergence.")
            break

    # Interpolate the total LOWESS correction over the entire image
    # Flatten the radii and total_lowess_correction for interpolation
    radii_flat = radii.flatten()
    lowess_correction_full = np.interp(radii_flat, radial_distances, total_lowess_correction)
    lowess_correction_full = lowess_correction_full.reshape(radii.shape)

    # Update the background with the LOWESS corrections
    background_full = background + lowess_correction_full

    # Subtract the updated background from the original image
    corrected_image = image - background_full

    vmin = np.nanmin(image)
    vmax = np.nanmax(image)

    # Optionally, plot the final background model
    plt.figure()
    plt.title('Final Background Model Including LOWESS Corrections')
    plt.imshow(background_full, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Background Intensity')
    plt.show()

    # Optionally, plot the corrected image
    plt.figure()
    plt.title('Corrected Image')
    plt.imshow(corrected_image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Corrected Intensity')
    plt.show()

    return corrected_image
