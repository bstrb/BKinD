import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# General Pseudo-Voigt function definition
def pseudo_voigt(r, A, mu, sigma, gamma, eta):
    gaussian = np.exp(-((r - mu) ** 2) / (2 * sigma ** 2))
    lorentzian = gamma ** 2 / ((r - mu) ** 2 + gamma ** 2)
    return A * (eta * lorentzian + (1 - eta) * gaussian)

def process_image(image, mask, center_x, center_y):
    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Apply the mask
    within_radius_mask = mask
    print(f"Number of pixels unmasked: {np.sum(within_radius_mask)}")

    # Check if there are any valid pixels left after masking
    if np.sum(within_radius_mask) == 0:
        raise ValueError("No valid pixels left after masking. Consider adjusting the mask.")

    # Flatten arrays and remove masked values
    masked_image = image[within_radius_mask]
    masked_radii = radii[within_radius_mask]

    # Bin the data to compute mean intensity in each bin
    num_bins = 500  # Adjust the number of bins as needed
    bins = np.linspace(0, masked_radii.max(), num_bins)
    bin_indices = np.digitize(masked_radii, bins)

    # Compute mean intensity for each bin
    radial_means = []
    radial_distances = []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            mean_intensity = np.mean(masked_image[bin_mask])
            radial_means.append(mean_intensity)
            radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center

    radial_means = np.array(radial_means)
    radial_distances = np.array(radial_distances)

    # Fit a Pseudo-Voigt peak to the center beam using the mean intensities
    initial_guess = [
        np.nanmax(radial_means),  # A (amplitude)
        0,                        # mu (position, assuming center at 0 radius)
        10,                       # sigma (initial Gaussian width)
        10,                       # gamma (initial Lorentzian width)
        0.5                       # eta (mixing parameter)
    ]

    try:
        popt_pv, _ = curve_fit(
            pseudo_voigt,
            radial_distances,
            radial_means,
            p0=initial_guess,
            maxfev=50000  # Increase max function evaluations for better convergence
        )
        print("Pseudo-Voigt fitting successful.")
    except RuntimeError as e:
        print("Pseudo-Voigt curve fitting failed:", e)
        popt_pv = initial_guess  # Use initial guess if fitting fails

    # Compute residuals after Pseudo-Voigt fit
    residuals_pv = radial_means - pseudo_voigt(radial_distances, *popt_pv)

    # Compute residuals
    residuals_pv = radial_means - pseudo_voigt(radial_distances, *popt_pv)

    # Plot the residuals of the PV fit
    plt.figure()
    plt.title('Residuals after Pseudo-Voigt Fit')
    plt.plot(radial_distances, residuals_pv, 'bo', label='Residuals')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()
    plt.savefig('pv_fit_residual.png')
    plt.close()

    # Fit a spline to the residuals after Pseudo-Voigt subtraction
    spline = UnivariateSpline(radial_distances, residuals_pv, s=len(radial_distances) * 0.08)

    # Generate the total background model at binned radial distances
    total_background_binned = pseudo_voigt(radial_distances, *popt_pv) + spline(radial_distances)

    # Compute residuals after total fit at binned data points
    residuals_total = radial_means - total_background_binned

    # For per-pixel total background, we need to be careful
    # Evaluate spline only within the range of radial_distances used for fitting
    spline_fit = np.zeros_like(radii)
    valid_radii_mask = (radii >= radial_distances.min()) & (radii <= radial_distances.max())
    spline_fit[valid_radii_mask] = spline(radii[valid_radii_mask])

    # Generate the background model for the Pseudo-Voigt fit over the entire image
    background_pv = pseudo_voigt(radii, *popt_pv)
    background_pv[~within_radius_mask] = 0

    # Combine the Pseudo-Voigt and spline background models
    total_background = background_pv + spline_fit

    # Subtract the total background from the original image
    corrected_image = np.where(mask, image - total_background, np.nan)
    corrected_image_no_mask = image - total_background  # Version without masking applied

    # Plotting results
    plot_results(
        image,
        corrected_image_no_mask,
        radial_distances,
        radial_means,
        popt_pv,
        spline,
        residuals_pv,
        residuals_total
    )

    return corrected_image, corrected_image_no_mask, radial_distances, radial_means, popt_pv

def plot_results(image, corrected_image_no_mask, radial_distances, radial_means, popt_pv, spline, residuals_pv, residuals_total):
    # Determine the brightness scale based on the original image
    vmin = min(np.nanmin(image), np.nanmin(corrected_image_no_mask))
    vmax = max(np.nanmax(image), np.nanmax(corrected_image_no_mask))

    # Plot the original image
    plt.figure()
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig('original_image.png')
    plt.close()

    # Plot the background corrected image (no mask exclusion)
    plt.figure()
    plt.title('Background Corrected Image (No Mask Exclusion)')
    plt.imshow(corrected_image_no_mask, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig('background_corrected_image_no_mask.png')
    plt.close()

    # Plot the radial means and the fitted Pseudo-Voigt model
    plt.figure()
    plt.title('Radial Intensity Profile with Pseudo-Voigt Fit')
    plt.plot(radial_distances, radial_means, 'bo', label='Radial Means')
    r_fit = np.linspace(0, max(radial_distances), 2000)
    fitted_pv = pseudo_voigt(r_fit, *popt_pv)
    plt.plot(r_fit, fitted_pv, 'r-', label='Fitted Pseudo-Voigt')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Mean Intensity')
    plt.legend()
    plt.savefig('radial_intensity_profile.png')
    plt.close()

    # Plot the residuals after Pseudo-Voigt fit and the spline fit to residuals
    plt.figure()
    plt.title('Residuals after Pseudo-Voigt Fit with Spline Fit')
    plt.plot(radial_distances, residuals_pv, 'bo', label='Residuals')
    spline_fit_residuals = spline(radial_distances)
    plt.plot(radial_distances, spline_fit_residuals, 'g-', label='Spline Fit')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()
    plt.savefig('spline_fit_to_residuals.png')
    plt.close()

    # Plot the residuals after total fit (Pseudo-Voigt + Spline) at binned data points
    plt.figure()
    plt.title('Residuals after Total Fit (Pseudo-Voigt + Spline)')
    plt.plot(radial_distances, residuals_total, 'bo', label='Total Residuals')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()
    plt.savefig('residuals_after_total_fit.png')
    plt.close()

# Example usage:
# Assuming you have the variables `image`, `mask`, `center_x`, and `center_y` defined
# corrected_image, corrected_image_no_mask, radial_distances, radial_means, popt_pv = process_image(image, mask, center_x, center_y)
