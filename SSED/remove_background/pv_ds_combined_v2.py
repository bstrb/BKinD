import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the Pseudo-Voigt function
def pseudo_voigt(r, A, mu, sigma, gamma, eta):
    gaussian = np.exp(-((r - mu) ** 2) / (2 * sigma ** 2))
    lorentzian = gamma ** 2 / ((r - mu) ** 2 + gamma ** 2)
    return A * (eta * lorentzian + (1 - eta) * gaussian)

def process_image(image, center_x, center_y):
    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Flatten the image and radii
    image_flat = image.flatten()
    radii_flat = radii.flatten()

    # Define max_radius as maximum radius to consider
    max_radius = np.min(image.shape)/np.sqrt(2) - 10
    # print(max_radius)

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
    drop_threshold = -np.max(np.abs(gradient_rev)) * 0.5  # Adjust the factor as needed
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

    # Initial guesses based on data
    A_initial = np.max(radial_medians_filtered)*3
    mu_initial = 0  # Centered at 0
    sigma_initial = np.std(radial_distances_filtered)/10
    gamma_initial = sigma_initial  # Assuming gamma similar to sigma
    eta_initial = 0.5  # Mix of Gaussian and Lorentzian

    initial_guess_pv = [A_initial, mu_initial, sigma_initial, gamma_initial, eta_initial]

    # Fit the Pseudo-Voigt function
    try:
        popt_pv, _ = curve_fit(
            pseudo_voigt,
            radial_distances_filtered,
            radial_medians_filtered,
            p0=initial_guess_pv,
            sigma=radial_stds_filtered,
            absolute_sigma=True,
            maxfev=50000
        )
        print("Pseudo-Voigt fitting successful.")
    except RuntimeError as e:
        print("Pseudo-Voigt curve fitting failed:", e)
        popt_pv = initial_guess_pv  # Use initial guess if fitting fails

    # Compute residuals
    fitted_pv = pseudo_voigt(radial_distances_filtered, *popt_pv)
    residuals_pv = radial_medians_filtered - fitted_pv

    # Compute the background over the entire image
    background_pv = pseudo_voigt(radii, *popt_pv)

    # Subtract background from the original image
    corrected_image = image - background_pv

    # Plotting results
    plot_results(
        image,
        corrected_image,
        radial_distances_filtered,
        radial_medians_filtered,
        popt_pv,
        residuals_pv
    )

    return corrected_image

def plot_results(image, corrected_image, radial_distances, radial_medians, popt_pv, residuals_pv):
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

    # Plot the radial medians and the fitted Pseudo-Voigt curve
    plt.figure()
    plt.title('Radial Intensity Profile and Pseudo-Voigt Fit')
    plt.plot(radial_distances, radial_medians, 'bo', label='Radial medians')
    r_fit = np.linspace(min(radial_distances), max(radial_distances), 1000)
    fitted_pv_curve = pseudo_voigt(r_fit, *popt_pv)
    plt.plot(r_fit, fitted_pv_curve, 'r-', label='Fitted Pseudo-Voigt')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('median Intensity')
    plt.legend()

    # Plot the residuals
    plt.figure()
    plt.title('Residuals after Pseudo-Voigt Fit')
    plt.plot(radial_distances, residuals_pv, 'ro', label='Residuals')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Residual Intensity')
    plt.legend()

    plt.show()
