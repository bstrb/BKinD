# load_and_display_image.py

import fabio
import matplotlib.pyplot as plt
import tkinter as tk

def load_and_display_image(self):
    img_data = fabio.open(self.file_path).data

    # Fit the annulus to the selected image
    self.fit_annulus_to_data(img_data)

    # Retrieve the fitted parameters
    x0, y0 = self.center
    inner_radius = self.inner_radius
    outer_radius = self.outer_radius

    # Update the display image button state after fitting
    self.calculate_button.config(state=tk.NORMAL)

    # Plot the image and the fitted annular region
    fig, ax = plt.subplots()
    ax.imshow(img_data, cmap='gray')
    plt.colorbar(ax.imshow(img_data, cmap='gray'))

    # Draw the fitted annular region
    outer_circle = plt.Circle(self.center, outer_radius, color='red', fill=False, linewidth=1)
    inner_circle = plt.Circle(self.center, inner_radius, color='red', fill=False, linewidth=1)
    ax.add_patch(outer_circle)
    ax.add_patch(inner_circle)
    
    plt.title(f'Fitted Annulus: Center=({x0:.2f}, {y0:.2f}), Inner Radius={inner_radius:.2f}, Outer Radius={outer_radius:.2f}')
    plt.show()

    print(f"Fitted Annulus: Center=({x0:.2f}, {y0:.2f}), Inner Radius={inner_radius:.2f}, Outer Radius={outer_radius:.2f}")

    # Enable the calculate button after displaying the image with the annular region
    self.calculate_button.config(state=tk.NORMAL)
