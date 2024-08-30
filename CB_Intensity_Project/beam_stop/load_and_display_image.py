# load_and_display_image.py
import fabio
import matplotlib.pyplot as plt
import tkinter as tk

def load_and_display_image(self):
    img_data = fabio.open(self.file_path).data

    # Fit the dark circle and Gaussian tail to the selected image
    self.fit_annulus_to_data(img_data)
    self.fit_gaussian_tail(img_data, self.center, self.inner_radius)

    # Retrieve the fitted parameters
    x0, y0 = self.center
    inner_radius = self.inner_radius
    sigma = self.sigma

    # Update the display image button state after fitting
    self.calculate_button.config(state=tk.NORMAL)

    # Plot the image and the fitted regions
    fig, ax = plt.subplots()
    ax.imshow(img_data, cmap='gray')
    plt.colorbar(ax.imshow(img_data, cmap='gray'))

    # Draw the dark circle
    inner_circle = plt.Circle(self.center, inner_radius, color='red', fill=False, linewidth=1)
    ax.add_patch(inner_circle)

    # Draw the Gaussian tail (approximated by an ellipse)
    tail_circle = plt.Circle(self.center, sigma, color='blue', fill=False, linewidth=1)
    ax.add_patch(tail_circle)
    
    plt.title(f'Fitted Model: Center=({x0:.2f}, {y0:.2f}), Inner Radius={inner_radius:.2f}, Sigma={sigma:.2f}')
    plt.show()

    # print(f"Fitted Model: Center=({x0:.2f}, {y0:.2f}), Inner Radius={inner_radius:.2f}, Sigma={sigma:.2f}")
