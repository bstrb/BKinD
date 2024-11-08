import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from multi_pseudo_voigt import multi_pseudo_voigt


def process_image(image, mask, center_x, center_y, n_peaks, peak_x_positions, peak_y_positions, peak_intensities):
    # Define exclusion zones around peaks based on their intensity
    exclusion_radius_factor = 0.05  # Reduced factor to avoid overly large exclusion zones
    max_exclusion_radius = 5  # Set a maximum limit for the exclusion radius
    for px, py, intensity in zip(peak_x_positions[:n_peaks], peak_y_positions[:n_peaks], peak_intensities[:n_peaks]):
        exclusion_radius = min(max(5, intensity * exclusion_radius_factor), max_exclusion_radius)  # Set a minimum radius of 5 pixels and a maximum limit
        y_indices, x_indices = np.ogrid[:image.shape[0], :image.shape[1]]
        distance_from_peak = np.sqrt((x_indices - px) ** 2 + (y_indices - py) ** 2)
        mask[distance_from_peak <= exclusion_radius] = 0

    # Verify the masked image after excluding peaks
    image_masked = np.where(mask, image, np.nan)
    print(f"Masked image stats after excluding peaks - min: {np.nanmin(image_masked) if np.sum(~np.isnan(image_masked)) > 0 else 'N/A'}, max: {np.nanmax(image_masked) if np.sum(~np.isnan(image_masked)) > 0 else 'N/A'}, count of valid pixels: {np.sum(~np.isnan(image_masked))}")

    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Limit the radius to 450 pixels
    radius_limit = 450
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
    num_bins = 100  # Adjust the number of bins as needed
    bins = np.linspace(0, masked_radii.max(), num_bins)
    bin_indices = np.digitize(masked_radii, bins)

    # Compute mean intensity for each bin
    radial_means = []
    radial_distances = []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            mean_intensity = masked_image[bin_mask].mean()
            radial_means.append(mean_intensity)
            radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center

    radial_means = np.array(radial_means)
    radial_distances = np.array(radial_distances)

    # Fit a multi-component Pseudo-Voigt curve to the radial means
    # Refined initial guess for the parameters (adjust as needed)
    initial_guess = [
        np.nanmax(radial_means),      # A1
        185,                          # mu1 (first bump position refined)
        10,                           # sigma1 (reduced to capture sharpness)
        15,                           # gamma1 (adjusted for sharper peak)
        0.5,                          # eta1
        np.nanmax(radial_means) / 2,  # A2
        320,                          # mu2 (second bump position)
        20,                           # sigma2
        20,                           # gamma2
        0.5,                          # eta2
        np.nanmax(radial_means) / 4,  # A3
        np.nanmedian(radial_distances),  # mu3 (center peak)
        50,                           # sigma3
        50,                           # gamma3
        0.5                           # eta3
    ]

    # Boundaries for the parameters to ensure physical meaningfulness
    param_bounds = (
        [0, 180, 1, 5, 0, 0, 300, 5, 5, 0, 0, 0, 10, 10, 0],   # Lower bounds (more specific)
        [np.inf, 190, 15, 25, 1, np.inf, 340, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, 1]  # Upper bounds
    )

    # Perform the curve fitting
    try:
        popt, pcov = curve_fit(
            multi_pseudo_voigt,
            radial_distances,
            radial_means,
            p0=initial_guess,
            bounds=param_bounds,
            maxfev=20000  # Increase max function evaluations for better convergence
        )
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        popt = initial_guess  # Use initial guess if fitting fails

    # Generate the background model over the entire image within the radius limit
    background = multi_pseudo_voigt(radii, *popt)
    background[~within_radius_mask] = 0

    # Subtract the background from the original image
    corrected_image = np.where(mask, image - background, np.nan)  # Only subtract background where mask is valid
    corrected_image_no_mask = np.copy(image)  # Version without masking applied
    corrected_image_no_mask -= background  # Subtract background from entire image

    return corrected_image, corrected_image_no_mask, radial_distances, radial_means, popt


def plot_results(image, corrected_image, corrected_image_no_mask, radial_distances, radial_means, popt):
    # Plot the original and corrected images
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', origin='lower')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title('Background Corrected Image (Masked Regions Excluded)')
    plt.imshow(corrected_image, cmap='gray', origin='lower')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title('Background Corrected Image (No Mask Exclusion)')
    plt.imshow(corrected_image_no_mask, cmap='gray', origin='lower')
    plt.colorbar()

    plt.show()

    # Plot the radial means and the fitted background model
    plt.figure()
    plt.title('Radial Intensity Profile')
    plt.plot(radial_distances, radial_means, 'bo', label='Radial Means')
    r_fit = np.linspace(0, max(radial_distances), 2000)  # Increase the number of points for better resolution
    plt.plot(r_fit, multi_pseudo_voigt(r_fit, *popt), 'r-', label='Fitted Background')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Mean Intensity')
    plt.legend()
    plt.show()
