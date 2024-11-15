import numpy as np
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

def plot_results(image, corrected_image, radial_distances, radial_medians, popt_ds, residuals_ds, N_sinusoids):
    # Determine the brightness scale based on the original image
    vmin = np.nanmin(image)
    vmax = np.nanmax(image)

    # Plot the original image
    plt.figure()
    plt.title('Original Image')
    plt.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Plot the background corrected image (no mask exclusion)
    plt.figure()
    plt.title('Background Corrected Image')
    plt.imshow(corrected_image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
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
