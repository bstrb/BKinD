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

    # Limit the data to radius <= 520 pixels
    max_radius = 50
    within_limit_mask = masked_radii <= max_radius
    masked_image = masked_image[within_limit_mask]
    masked_radii = masked_radii[within_limit_mask]

    # Bin the data to compute median intensity in each bin
    num_bins = max_radius  # Adjust the number of bins as needed
    bins = np.linspace(0, masked_radii.max(), num_bins)
    bin_indices = np.digitize(masked_radii, bins)

    # Compute median intensity and standard deviation for each bin
    radial_medians = []
    radial_stds = []
    radial_distances = []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            median_intensity = np.median(masked_image[bin_mask])
            std_intensity = np.std(masked_image[bin_mask])
            radial_medians.append(median_intensity)
            radial_stds.append(std_intensity)
            radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center

    radial_medians = np.array(radial_medians)
    radial_stds = np.array(radial_stds)
    radial_distances = np.array(radial_distances)

    # Fit a sum of damped sinusoids to the radial medians
    N_sinusoids = 5 # Number of damped sinusoids to sum
    initial_guess_ds = []
    for i in range(N_sinusoids):
        # Estimate initial parameters based on radial medians
        A_ds = (np.max(radial_medians) - np.min(radial_medians)) / N_sinusoids
        k = 2 * np.pi / (50 * (i + 1))  # Adjust the periodicity
        phi = np.pi * i / N_sinusoids
        tau = np.max(radial_medians)  # Adjust the damping factor
        initial_guess_ds.extend([A_ds, k, phi, tau])

    # Fit the sum of damped sinusoids to the radial medians
    try:
        popt_ds, _ = curve_fit(
            sum_damped_sinusoids,
            radial_distances,
            radial_medians,
            p0=initial_guess_ds,
            sigma=radial_stds,
            absolute_sigma=True,
            maxfev=100000
        )
        print("Damped sinusoids fitting successful.")
    except RuntimeError as e:
        print("Damped sinusoids curve fitting failed:", e)
        popt_ds = initial_guess_ds  # Use initial guess if fitting fails

    # Compute residuals after damped sinusoids fit
    residuals_ds = radial_medians - sum_damped_sinusoids(radial_distances, *popt_ds)

    # Evaluate the damped sinusoids model over the entire image
    background_ds = sum_damped_sinusoids(radii, *popt_ds)
    background_ds[~within_radius_mask] = 0

    # Subtract the background from the original image
    corrected_image = np.where(mask, image - background_ds, np.nan)
    corrected_image_no_mask = image - background_ds  # Version without masking applied

    # Plotting results
    plot_results(
        image,
        corrected_image_no_mask,
        radial_distances,
        radial_medians,
        popt_ds,
        residuals_ds,
        N_sinusoids
    )

    return corrected_image, corrected_image_no_mask, radial_distances, radial_medians, popt_ds

def plot_results(image, corrected_image_no_mask, radial_distances, radial_medians, popt_ds, residuals_ds, N_sinusoids):
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
