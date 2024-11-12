import matplotlib.pyplot as plt
import numpy as np


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
