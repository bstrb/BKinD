import matplotlib.pyplot as plt
import numpy as np
from triple_pseudo_voigt import multi_pseudo_voigt

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
    r_fit = np.linspace(0, max(radial_distances), 2000)  # Increase the number of points for better resolution
    fitted_profile = multi_pseudo_voigt(r_fit, *popt)
    plt.plot(r_fit, fitted_profile, 'r-', label='Fitted Background')
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
