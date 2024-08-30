# fit_gaussian_tail.py
import numpy as np
from scipy.optimize import curve_fit

def gaussian_tail(xy, amplitude, x0, y0, sigma, offset):
    """2D Gaussian function for the tail."""
    x, y = xy
    return offset + amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def fit_gaussian_tail(self, img_data, center, inner_radius):
    """Fit a 2D Gaussian tail to the region outside the dark circle."""
    y, x = np.indices(img_data.shape)
    mask = (x - center[0])**2 + (y - center[1])**2 > inner_radius**2

    # Flatten arrays and keep only the masked region
    x = x[mask]
    y = y[mask]
    data = img_data[mask]

    # Initial guess based on the provided constraints
    initial_guess = (np.max(data) - np.min(data), center[0], center[1], 50, np.min(data))

    # Define bounds based on the constraints
    bounds = (
        [0, center[0] - 20, center[1] - 20, 20, np.min(data)],  # Lower bounds
        [np.inf, center[0] + 20, center[1] + 20, 100, np.max(data)]  # Upper bounds
    )

    # Fit the Gaussian tail
    params, _ = curve_fit(
        gaussian_tail, (x, y), data, p0=initial_guess, bounds=bounds
    )

    amplitude, x0, y0, sigma, offset = params
    self.center = (x0, y0)
    self.amplitude = amplitude
    self.sigma = sigma

    print(f"Fitted Gaussian Tail: Center=({x0:.2f}, {y0:.2f}), Amplitude={amplitude:.2f}, Sigma={sigma:.2f}, Offset={offset:.2f}")
