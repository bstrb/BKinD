import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# General Pseudo-Voigt function definition
def pseudo_voigt(r, A, mu, sigma, gamma, eta):
    gaussian = np.exp(-((r - mu) ** 2) / (2 * sigma ** 2))
    lorentzian = gamma ** 2 / ((r - mu) ** 2 + gamma ** 2)
    return A * (eta * lorentzian + (1 - eta) * gaussian)

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

# Combined model: Pseudo-Voigt plus sum of damped sinusoids
def combined_model(r, A_pv, mu, sigma, gamma, eta, *ds_params):
    pv = pseudo_voigt(r, A_pv, mu, sigma, gamma, eta)
    ds = sum_damped_sinusoids(r, *ds_params)
    return pv #+ ds

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

    # Limit the data to radius in pixels
    max_radius = 500
    within_limit_mask = masked_radii <= max_radius
    masked_image = masked_image[within_limit_mask]
    masked_radii = masked_radii[within_limit_mask]

    # Bin the data to compute mean intensity in each bin
    num_bins = max_radius  # Adjust the number of bins as needed
    bins = np.linspace(0, masked_radii.max(), num_bins)
    bin_indices = np.digitize(masked_radii, bins)

    # Compute mean intensity and standard deviation for each bin
    radial_means = []
    radial_stds = []
    radial_distances = []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            mean_intensity = np.mean(masked_image[bin_mask])
            std_intensity = np.std(masked_image[bin_mask])
            radial_means.append(mean_intensity)
            radial_stds.append(std_intensity)
            radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center

    radial_means = np.array(radial_means)
    radial_stds = np.array(radial_stds)
    radial_distances = np.array(radial_distances)

    # Fit a Pseudo-Voigt peak to the center beam using the mean intensities
    initial_guess_pv = [
        np.nanmax(radial_means)*3, # A (amplitude)
        0,                        # mu (position, assuming center at 0 radius)
        50,                       # sigma (initial Gaussian width)
        50,                       # gamma (initial Lorentzian width)
        0.5                       # eta (mixing parameter)
    ]

    try:
        popt_pv, _ = curve_fit(
            pseudo_voigt,
            radial_distances,
            radial_means,
            p0=initial_guess_pv,
            sigma=radial_stds,
            absolute_sigma=True,
            maxfev=50000  # Increase max function evaluations for better convergence
        )
        print("Pseudo-Voigt fitting successful.")
    except RuntimeError as e:
        print("Pseudo-Voigt curve fitting failed:", e)
        # popt_pv = initial_guess_pv  # Use initial guess if fitting fails

    # Compute residuals after Pseudo-Voigt fit
    residuals_pv = radial_means - pseudo_voigt(radial_distances, *popt_pv)

    # Fit a sum of damped sinusoids to the residuals
    N_sinusoids = 5 # Number of damped sinusoids to sum
    initial_guess_ds = []
    for i in range(N_sinusoids):
        # Estimate initial parameters based on residuals
        A_ds = (np.max(residuals_pv) - np.min(residuals_pv)) / N_sinusoids
        k = 2 * np.pi / (100 * (i + 1))  # Adjust the periodicity
        phi = np.pi*i / 7
        tau = 500  # Adjust the damping factor
        initial_guess_ds.extend([A_ds, k, phi, tau])

    # Combine initial guesses for combined model
    # initial_guess_combined = initial_guess_pv + initial_guess_ds
    initial_guess_combined = list(popt_pv) + initial_guess_ds

    # Fit the combined model to the radial means
    try:
        popt_combined, _ = curve_fit(
            combined_model,
            radial_distances,
            radial_means,
            p0=initial_guess_combined,
            # sigma=radial_stds,
            # absolute_sigma=True,
            maxfev=100000
        )
        print("Combined fitting (PV + damped sinusoids) successful.")
    except RuntimeError as e:
        print("Combined curve fitting failed:", e)
        popt_combined = initial_guess_combined  # Use initial guess if fitting fails

    # Extract PV and damped sinusoid parameters
    popt_pv_combined = popt_combined[:5]
    popt_ds_combined = popt_combined[5:]

    # Compute total background model
    total_background_binned = combined_model(radial_distances, *popt_combined)

    # Compute residuals after total fit
    residuals_total = radial_means - total_background_binned

    # Evaluate the combined model over the entire image
    background_pv = pseudo_voigt(radii, *popt_pv_combined)
    background_ds = sum_damped_sinusoids(radii, *popt_ds_combined)
    total_background = background_pv + background_ds
    total_background[~within_radius_mask] = 0

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
        popt_pv_combined,
        popt_ds_combined,
        residuals_pv,
        residuals_total,
        N_sinusoids
    )

    return corrected_image, corrected_image_no_mask, radial_distances, radial_means, popt_pv

def plot_results(image, corrected_image_no_mask, radial_distances, radial_means, popt_pv, popt_pv_combined, popt_ds_combined, residuals_pv, residuals_total, N_sinusoids):
    # Determine the brightness scale based on the original image
    vmin = min(np.nanmin(image), np.nanmin(corrected_image_no_mask))
    vmax = max(np.nanmax(image), np.nanmax(corrected_image_no_mask))

    # Plot the original image
    plt.figure()
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Plot the background corrected image (no mask exclusion)
    plt.figure()
    plt.title('Background Corrected Image (No Mask Exclusion)')
    plt.imshow(corrected_image_no_mask, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Plot the radial means and the fitted models
    plt.figure()
    plt.title('Radial Intensity Profile with Fits')
    plt.plot(radial_distances, radial_means, 'bo', label='Radial Means')
    r_fit = np.linspace(0, max(radial_distances), 2000)
    fitted_pv = pseudo_voigt(r_fit, *popt_pv)
    fitted_combined = combined_model(r_fit, *(list(popt_pv_combined) + list(popt_ds_combined)))
    plt.plot(r_fit, fitted_pv, 'r--', label='Fitted Pseudo-Voigt')
    plt.plot(r_fit, fitted_combined, 'g-', label='Combined Fit (PV + Damped Sinusoids)')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Mean Intensity')
    plt.legend()

    # Plot the residuals after Pseudo-Voigt fit
    plt.figure()
    plt.title('Residuals after Pseudo-Voigt fit')
    plt.plot(radial_distances, residuals_pv, 'ro', label='Residuals after Pseudo-Voigt Fit')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()

    # Plot the residuals after Pseudo-Voigt fit and combined fit
    plt.figure()
    plt.title('Residuals after Fits')
    residuals_combined = radial_means - combined_model(radial_distances, *(list(popt_pv_combined) + list(popt_ds_combined)))
    plt.plot(radial_distances, residuals_combined, 'ro', label='Residuals after Combined Fit')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()

    # Plot each damped sinusoid component
    plt.figure()
    plt.title('Damped Sinusoid Components')
    for i in range(N_sinusoids):
        A_ds = popt_ds_combined[4 * i]
        k = popt_ds_combined[4 * i + 1]
        phi = popt_ds_combined[4 * i + 2]
        tau = popt_ds_combined[4 * i + 3]
        ds_component = damped_sinusoid(r_fit, A_ds, k, phi, tau)
        plt.plot(r_fit, ds_component, label=f'Damped Sinusoid {i+1}')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Intensity')
    plt.legend()

    # Plot the total damped sinusoid contribution
    total_ds = sum_damped_sinusoids(r_fit, *popt_ds_combined)
    plt.figure()
    plt.title('Total Damped Sinusoid Contribution')
    plt.plot(r_fit, total_ds, 'm-', label='Total Damped Sinusoids')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Intensity')
    plt.legend()

    # Show all plots simultaneously
    plt.show()
