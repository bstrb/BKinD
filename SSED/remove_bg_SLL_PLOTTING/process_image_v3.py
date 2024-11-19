import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from compute_radial_statistics import compute_radial_statistics
import statsmodels.api as sm

def super_lorentzian(r, A, mu, gamma, k, C):
    return A / (1 + ((r - mu) / gamma) ** 2) ** k + C

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
    num_bins = int(max_radius)
    bins = np.linspace(0, max_radius, num_bins)

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
    k_initial = 2 #np.max(np.gradient(radial_medians))
    C_initial = np.min(radial_medians)

    initial_guess_sl = [A_initial, mu_initial, gamma_initial, k_initial, C_initial]
    print(initial_guess_sl)
    # Fit the super-Lorentzian function
    try:
        popt_sl, _ = curve_fit(
            super_lorentzian,
            radial_distances,
            radial_medians,
            p0=initial_guess_sl,
            sigma=radial_stds,
            absolute_sigma=True,
            maxfev=100000
        )
        print("Super-Lorentzian fitting successful.")
    except RuntimeError as e:
        print("Super-Lorentzian curve fitting failed:", e)
        popt_sl = initial_guess_sl  # Use initial guess if fitting fails

    print(popt_sl)
    # Compute the super-Lorentzian background over the entire image
    background_sl = super_lorentzian(radii, *popt_sl)

    # Subtract background from the original image
    corrected_image = image - background_sl

    # Plotting results
    plt.figure()
    plt.title('Radial Intensity Profile and Super-Lorentzian Fit')
    plt.plot(radial_distances, radial_medians, 'bo', label='Radial Medians')
    r_fit = np.linspace(min(radial_distances), max(radial_distances), 1000)
    fitted_sl_curve = super_lorentzian(r_fit, *popt_sl)
    plt.plot(r_fit, fitted_sl_curve, 'r-', label='Fitted Super-Lorentzian')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Median Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Iteratively apply LOWESS (Locally Weighted Scatterplot Smoothing) to flatten the residuals
    residuals = radial_medians - super_lorentzian(radial_distances, *popt_sl)
    iteration = 0
    residual_threshold = 0.005 * np.std(radial_medians)  # Set a threshold for residual convergence
    while True:
        lowess = sm.nonparametric.lowess(residuals, radial_distances, frac=0.1, delta = 1)
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

        # Subtract the LOWESS fit from the residuals
        residuals -= np.interp(radial_distances, lowess_x, lowess_y)

        # Check if the residuals are sufficiently flattened
        if np.std(residuals) < residual_threshold:
            print(f"Convergence achieved after {iteration} iterations.")
            break

        iteration += 1
        if iteration >= 1:#10:  # Limit the number of iterations to avoid infinite loops
            print("Maximum number of iterations reached without full convergence.")
            break

    return corrected_image
