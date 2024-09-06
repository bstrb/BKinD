# load_and_display_image.py

import fabio
import matplotlib.pyplot as plt
import tkinter as tk

from fit_gaussian import fit_gaussian
from fit_pseudo_voigt import fit_pseudo_voigt

def load_and_display_image(self, method='gaussian'):
    # Load image data using fabio
    img_data = fabio.open(self.file_path).data

    if method == 'gaussian':
        # Fit a Gaussian to the image to determine the center and spread
        center, sigma_x, sigma_y, amplitude, offset = fit_gaussian(img_data)
        print(f"Gaussian Fitted Center: {self.center} and Amplitude: {amplitude}")
        print(f"Offset: {offset}, Sigma X: {sigma_x}, Sigma Y: {sigma_y}")
    elif method == 'pseudo_voigt':
        # Fit a pseudo-Voigt to the image to determine the center and spread
        center, sigma_x, sigma_y, eta, amplitude = fit_pseudo_voigt(img_data)
        print(f"Pseudo Voigt Fitted Center: {center} and Amplitude: {amplitude}")
        print(f"Eta: {eta}, Sigma X: {sigma_x}, Sigma Y: {sigma_y}")
    else:
        raise ValueError("Invalid fitting method. Choose 'gaussian' or 'pseudo_voigt'.")

    # Store the fitted parameters in the class (assuming this is part of a class)
    self.center = center
    self.sigma_x = sigma_x
    self.sigma_y = sigma_y

    # Get the sigma level from the user input
    sigma_level = float(self.sigma_level_entry.get())

    # Plot the image and the fitted ellipse
    fig, ax = plt.subplots()
    ax.imshow(img_data, cmap='gray')
    plt.colorbar(ax.imshow(img_data, cmap='gray'))

    # Draw the fitted ellipse with the specified sigma level
    ellipse = plt.Circle(self.center, sigma_level * max(self.sigma_x, self.sigma_y), color='red', fill=False, linewidth=1)
    ax.add_patch(ellipse)
    plt.title(f'Fitted {method.capitalize()} Center and Spread (Sigma Level = {sigma_level})')
    plt.show()

    print(f"Fitted Center: {self.center}")
    print(f"Sigma X: {self.sigma_x}, Sigma Y: {self.sigma_y}")

    # Enable the calculate button after displaying the image with the Gaussian or pseudo-Voigt area
    self.calculate_button.config(state=tk.NORMAL)
