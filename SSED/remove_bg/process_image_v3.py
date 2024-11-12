import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from multi_pseudo_voigt import multi_pseudo_voigt

def process_image(image, mask, center_x, center_y):

    # Verify the masked image after excluding peaks
    image_masked = np.where(mask, image, np.nan)
    print(f"Masked image stats after excluding peaks - min: {np.nanmin(image_masked) if np.sum(~np.isnan(image_masked)) > 0 else 'N/A'}, max: {np.nanmax(image_masked) if np.sum(~np.isnan(image_masked)) > 0 else 'N/A'}, count of valid pixels: {np.sum(~np.isnan(image_masked))}")

    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Limit the radius to 500 pixels
    radius_limit = 500
    within_radius_mask = (radii <= radius_limit) & mask
    print(f"Number of pixels within radius and unmasked: {np.sum(within_radius_mask)}")

    # Check if there are any valid pixels left after masking
    if np.sum(within_radius_mask) == 0:
        raise ValueError("No valid pixels left after masking. Consider reducing the exclusion radius or adjusting the mask.")

    # Flatten arrays and remove masked values
    masked_image = image[within_radius_mask]
    masked_radii = radii[within_radius_mask]

    # Verify the radial distances and masked image
    print(f"Masked radial distances - min: {masked_radii.min()}, max: {masked_radii.max()}, count: {len(masked_radii)}")
    print(f"Masked image intensities - min: {masked_image.min()}, max: {masked_image.max()}, count: {len(masked_image)}")

    # Bin the data
    num_bins = 1000  # Adjust the number of bins as needed
    bins = np.linspace(0, masked_radii.max(), num_bins)
    bin_indices = np.digitize(masked_radii, bins)

    # Compute mean intensity for each bin
    radial_medians = []
    radial_distances = []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            median_intensity = np.median(masked_image[bin_mask])
            radial_medians.append(median_intensity)
            radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center

    radial_medians = np.array(radial_medians)
    radial_distances = np.array(radial_distances)

    # Automatically determine the number of peaks
    peaks, _ = find_peaks(radial_medians, height=np.nanmax(radial_medians) * 0.1)
    peak_count = len(peaks) #+ 1  # Add one for the center peak
    print(f"Automatically determined number of peaks: {peak_count}")

    # Refined initial guess for the parameters (adjust as needed)
    initial_guess = []
    # Always include a center peak
    A_center = np.nanmax(radial_medians)
    mu_center = np.nanmin(radial_distances)  # Center peak should be near the minimum radius
    sigma_center = 20
    gamma_center = 20
    eta_center = 0.5
    initial_guess.extend([A_center, mu_center, sigma_center, gamma_center, eta_center])
    for i in range(peak_count - 1):  # Iterate over the remaining peaks
        A = radial_medians[peaks[i]]
        mu = radial_distances[peaks[i]]
        sigma = 20
        gamma = 20
        eta = 0.5
        initial_guess.extend([A, mu, sigma, gamma, eta])

    # Parameter bounds for better convergence
    lower_bounds = [0, 0, 1, 1, 0] * peak_count
    upper_bounds = [np.inf, masked_radii.max(), 50, 50, 1] * peak_count

    # Perform the curve fitting with retry mechanism
    for attempt in range(3):
        try:
            popt, pcov = curve_fit(
                multi_pseudo_voigt,
                radial_distances,
                radial_medians,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=30000  # Increase max function evaluations for better convergence
            )
            break
        except RuntimeError as e:
            print(f"Curve fitting attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                # Modify initial guesses slightly for retry
                initial_guess = [val * (1 + 0.1 * np.random.randn()) for val in initial_guess]
            else:
                popt = initial_guess  # Use initial guess if fitting fails after all attempts

    # Generate the background model over the entire image within the radius limit
    background = multi_pseudo_voigt(radii, *popt)
    background[~within_radius_mask] = 0

    # Subtract the background from the original image
    corrected_image = np.where(mask, image - background, np.nan)  # Only subtract background where mask is valid
    corrected_image_no_mask = np.copy(image)  # Version without masking applied
    corrected_image_no_mask -= background  # Subtract background from entire image

    # Plot the results
    plot_results(image, corrected_image_no_mask, radial_distances, radial_medians, popt)

    return corrected_image, corrected_image_no_mask, radial_distances, radial_medians, popt


def plot_results(image, corrected_image_no_mask, radial_distances, radial_medians, popt):
    # Determine the brightness scale based on the original image
    vmin = min(np.nanmin(image), np.nanmin(corrected_image_no_mask))
    vmax = max(np.nanmax(image), np.nanmax(corrected_image_no_mask))

    # Plot the original image
    plt.figure()
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.draw()
    plt.pause(0.001)

    # Plot the background corrected image (no mask exclusion)
    plt.figure()
    plt.title('Background Corrected Image (No Mask Exclusion)')
    plt.imshow(corrected_image_no_mask, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.draw()
    plt.pause(0.001)

    # Plot the radial medians and the fitted background model
    plt.figure()
    plt.title('Radial Intensity Profile')
    plt.plot(radial_distances, radial_medians, 'bo', label='Radial Medians')
    fitted_profile = multi_pseudo_voigt(radial_distances, *popt)
    plt.plot(radial_distances, fitted_profile, 'r-', label='Fitted Background')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Median Intensity')
    plt.legend()
    plt.draw()
    plt.pause(0.001)

    # Plot the residuals (difference between radial medians and fitted background)
    residuals = radial_medians - multi_pseudo_voigt(radial_distances, *popt)
    plt.figure()
    plt.title('Residuals of Fit')
    plt.plot(radial_distances, residuals, 'bo', label='Residuals')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()
    plt.draw()
    plt.pause(0.001)

    # Keep plots open until manually closed
    plt.show()
