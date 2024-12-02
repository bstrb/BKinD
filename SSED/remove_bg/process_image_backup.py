import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from compute_radial_statistics import compute_radial_statistics
import statsmodels.api as sm

def enhanced_super_lorentzian(r, A, mu, gamma, n, C, D):
    epsilon = 1e-6  # Small value to prevent division by zero
    safe_n = np.clip(n, 0, 10)  # Limit n to avoid overflow
    safe_gamma = np.maximum(gamma, epsilon)  # Ensure gamma is not too small
    base = 1 + ((r - mu) / safe_gamma) ** 2
    return A / np.power(base, safe_n) + C + D / (r + epsilon)

def process_image(image, center_x, center_y):
    try:
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
            [0, -1, 1e-3, 0, -np.inf, -np.inf],  # Lower bounds
            [np.inf, 1, np.max(radial_distances), 10, np.max(radial_medians), np.inf]  # Upper bounds
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
            # print("Enhanced Super-Lorentzian fitting successful.")
        except RuntimeError as e:
            print("Enhanced Super-Lorentzian curve fitting failed:", e)
            popt = initial_guess  # Use initial guess if fitting fails

        # Compute the background over the entire image
        background = enhanced_super_lorentzian(radii, *popt)

        # Iteratively apply LOWESS to flatten the residuals
        residuals = radial_medians - enhanced_super_lorentzian(radial_distances, *popt)
        iteration = 0
        
        # Initialize total LOWESS correction
        total_lowess_correction = np.zeros_like(radial_medians)
        prev_lowess_correction = np.zeros_like(radial_medians)
        
        mean_change_threshold = 0.01 * np.mean(np.abs(residuals))  # Adjust the multiplier as needed
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            lowess = sm.nonparametric.lowess(residuals, radial_distances, frac=0.05, delta=1)
            lowess_x = lowess[:, 0]
            lowess_y = lowess[:, 1]
        
            # Interpolate LOWESS correction
            lowess_correction = np.interp(radial_distances, lowess_x, lowess_y)
            
            # Compute mean absolute change
            mean_absolute_change = np.mean(np.abs(lowess_correction - prev_lowess_correction))

            # Accumulate the LOWESS correction
            total_lowess_correction += lowess_correction
            
            # Subtract the LOWESS fit from the residuals
            residuals -= lowess_correction
            
            # Check if the mean absolute change is less than the threshold
            if mean_absolute_change < mean_change_threshold:
                # print(f"Convergence achieved after {iteration} iterations.")
                break
            
            # Update prev_lowess_correction
            prev_lowess_correction = lowess_correction.copy()
            
            iteration += 1
        # else:
        #     print("Maximum number of iterations reached without full convergence.")


        # Interpolate the total LOWESS correction over the entire image
        # Flatten the radii and total_lowess_correction for interpolation
        radii_flat = radii.flatten()
        lowess_correction_full = np.interp(radii_flat, radial_distances, total_lowess_correction)
        lowess_correction_full = lowess_correction_full.reshape(radii.shape)

        # Update the background with the LOWESS corrections
        background_full = background + lowess_correction_full

        # Subtract the updated background from the original image
        corrected_image = image - background_full


        return corrected_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None