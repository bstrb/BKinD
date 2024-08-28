# load_and_display_image.py

import fabio
import matplotlib.pyplot as plt
import tkinter as tk

def load_and_display_image(self):
    img_data = fabio.open(self.file_path).data

    # Fit a Gaussian to the image to determine the center and spread
    self.fit_gaussian(img_data)

    # Get the sigma level from the user input
    sigma_level = float(self.sigma_level_entry.get())

    # Plot the image and the fitted Gaussian ellipse
    fig, ax = plt.subplots()
    ax.imshow(img_data, cmap='gray')
    plt.colorbar(ax.imshow(img_data, cmap='gray'))

    # Draw the fitted Gaussian ellipse with the specified sigma level
    ellipse = plt.Circle(self.center, sigma_level * max(self.sigma_x, self.sigma_y), color='red', fill=False, linewidth=1)
    ax.add_patch(ellipse)
    plt.title(f'Fitted Gaussian Center and Spread (Sigma Level = {sigma_level})')
    plt.show()

    print(f"Fitted Center: {self.center}")
    print(f"Sigma X: {self.sigma_x}, Sigma Y: {self.sigma_y}")

    # Enable the calculate button after displaying the image with the Gaussian area
    self.calculate_button.config(state=tk.NORMAL)