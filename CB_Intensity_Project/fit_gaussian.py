# fit_gaussian.py

import numpy as np
from scipy.optimize import curve_fit

def fit_gaussian(self, img_data):
    """Fit a 2D Gaussian to the image data to find the center and spread."""
    x = np.arange(img_data.shape[1])
    y = np.arange(img_data.shape[0])
    x, y = np.meshgrid(x, y)
    
    initial_guess = (img_data.shape[1]//2, img_data.shape[0]//2, 10, 10, np.max(img_data), np.min(img_data))
    params, _ = curve_fit(self.gaussian_2d, (x.ravel(), y.ravel()), img_data.ravel(), p0=initial_guess)
    
    x0, y0, sigma_x, sigma_y, amplitude, offset = params
    self.center = (x0, y0)
    self.sigma_x = sigma_x
    self.sigma_y = sigma_y