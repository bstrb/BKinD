import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# General Pseudo-Voigt function definition with multiple components
def multi_pseudo_voigt(r, *params):
    """
    Multi-component Pseudo-Voigt function to model background with multiple peaks.
    
    Parameters:
    - r: Radial distance array.
    - params: Flattened list of parameters for each component (A, mu, sigma, gamma, eta).
    
    Returns:
    - The sum of all Pseudo-Voigt components.
    """
    num_components = len(params) // 5
    total_pv = np.zeros_like(r)
    
    for i in range(num_components):
        A = params[i * 5]
        mu = params[i * 5 + 1]
        sigma = params[i * 5 + 2]
        gamma = params[i * 5 + 3]
        eta = params[i * 5 + 4]
        
        gaussian = np.exp(-((r - mu) ** 2) / (2 * sigma ** 2))
        lorentzian = gamma ** 2 / ((r - mu) ** 2 + gamma ** 2)
        pv = A * (eta * lorentzian + (1 - eta) * gaussian)
        
        total_pv += pv
    
    return total_pv

def process_image(image, mask, center_x, center_y, peaks = 3):

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
    num_bins = 500  # Adjust the number of bins as needed
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

    # Fit a multi-component Pseudo-Voigt curve to the radial means
    # Refined initial guess for the parameters (adjust as needed)
    peak_count = peaks
    initial_guess = []
    for i in range(peak_count):
        A = np.nanmax(radial_medians) / (2 ** i)
        mu = np.nanmin(radial_distances) + (np.nanmax(radial_distances) - np.nanmin(radial_distances)) * (i + 1) / (peak_count + 1)
        sigma = 20
        gamma = 20
        eta = 0.5
        initial_guess.extend([A, mu, sigma, gamma, eta])

    # Perform the curve fitting
    try:
        popt, pcov = curve_fit(
            multi_pseudo_voigt,
            radial_distances,
            radial_medians,
            p0=initial_guess,
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

    return corrected_image, corrected_image_no_mask, radial_distances, radial_medians, popt
