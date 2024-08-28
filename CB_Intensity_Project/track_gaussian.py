# track_gaussian.py

import tkinter as tk
from tkinter import messagebox
import fabio
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.optimize import curve_fit
import plotly.graph_objects as go



class CenterBeamIntensityApp:

    from create_widgets import create_widgets
    from browse_image import browse_image
    
    def __init__(self, root):
        self.root = root
        self.root.title("Center Beam Intensity Calculator")
        
        self.file_path = ""
        self.center = None
        self.sigma_x = None
        self.sigma_y = None

        # Create GUI components
        self.create_widgets()

    def toggle_smoothing(self):
        """Enable or disable the smoothing slider based on the checkbox state."""
        if self.smooth_var.get():
            self.smooth_slider.config(state=tk.NORMAL)
        else:
            self.smooth_slider.config(state=tk.DISABLED)

    def gaussian_2d(self, xy, x0, y0, sigma_x, sigma_y, amplitude, offset):
        """2D Gaussian function."""
        x, y = xy
        return amplitude * np.exp(-(((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2)))) + offset

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

    def smooth_intensity_values(self, intensity_values, window_size):
        """Smooth the intensity values using a moving average."""
        smoothed_values = np.convolve(intensity_values, np.ones(window_size) / window_size, mode='valid')
        
        # To maintain the same length as the original array, we'll pad the start of the smoothed array
        pad_size = len(intensity_values) - len(smoothed_values)
        return np.pad(smoothed_values, (pad_size // 2, pad_size - pad_size // 2), mode='edge')
    
    def calculate_inside_gaussian_region_intensity(self, img_data, sigma_level=2):
        """Calculate the sum of intensity inside the central Gaussian region within a specified sigma level."""
        row, column = self.center
        Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
        # Create a Gaussian mask within the specified sigma level
        mask = (((X - column)**2 / self.sigma_x**2) + ((Y - row)**2 / self.sigma_y**2)) <= sigma_level**2
        
        # Calculate the sum of intensity inside the central region
        inside_intensity = np.sum(img_data[mask])
        return inside_intensity

    def calculate_outside_gaussian_region_intensity(self, img_data, sigma_level=2):
        """Calculate the sum of intensity outside the central Gaussian region within a specified sigma level."""
        row, column = self.center
        Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
        # Create a Gaussian mask for outside the specified sigma level region
        mask = (((X - column)**2 / self.sigma_x**2) + ((Y - row)**2 / self.sigma_y**2)) > sigma_level**2
        
        # Calculate the sum of intensity outside the central region
        outside_intensity = np.sum(img_data[mask])
        return outside_intensity

    def calculate_intensities(self):
        # Get the directory containing the selected file
        img_directory = os.path.dirname(self.file_path)
        img_files = glob.glob(os.path.join(img_directory, '*.img'))
        
        if not img_files:
            messagebox.showwarning("No .img files found", "No .img files were found in the directory.")
            return

        inside_intensity_values = []
        outside_intensity_values = []
        total_intensity_values = []

        # Retrieve the sigma level entered by the user
        sigma_level = float(self.sigma_level_entry.get())

        for img_file in img_files:
            # Load the image
            img_data = fabio.open(img_file).data
            
            # Calculate sum of intensities inside, outside, and total
            inside_intensity = self.calculate_inside_gaussian_region_intensity(img_data, sigma_level=sigma_level)
            outside_intensity = self.calculate_outside_gaussian_region_intensity(img_data, sigma_level=sigma_level)
            total_intensity = np.sum(img_data)  # Calculate the sum of all pixel intensities in the image

            inside_intensity_values.append(inside_intensity)
            outside_intensity_values.append(outside_intensity)
            total_intensity_values.append(total_intensity)

        # Convert to numpy arrays and cast to float64 for safe division
        inside_intensity_values = np.array(inside_intensity_values, dtype=np.float64)
        outside_intensity_values = np.array(outside_intensity_values, dtype=np.float64)
        total_intensity_values = np.array(total_intensity_values, dtype=np.float64)
        absolute_difference_values = np.abs(inside_intensity_values - outside_intensity_values)

        # Get the selected normalization method from the dropdown
        normalization_method = self.normalization_var.get()

        if normalization_method == "sum":
            # Normalize the intensities to their respective sums
            inside_intensity_values /= np.sum(inside_intensity_values)
            outside_intensity_values /= np.sum(outside_intensity_values)
            total_intensity_values /= np.sum(total_intensity_values)
            absolute_difference_values /= np.sum(absolute_difference_values)

        elif normalization_method == "min_max":
            # Min-Max Normalization to [0, 1]
            inside_intensity_values = (inside_intensity_values - np.min(inside_intensity_values)) / (np.max(inside_intensity_values) - np.min(inside_intensity_values))
            outside_intensity_values = (outside_intensity_values - np.min(outside_intensity_values)) / (np.max(outside_intensity_values) - np.min(outside_intensity_values))
            total_intensity_values = (total_intensity_values - np.min(total_intensity_values)) / (np.max(total_intensity_values) - np.min(total_intensity_values))
            absolute_difference_values = (absolute_difference_values - np.min(absolute_difference_values)) / (np.max(absolute_difference_values) - np.min(absolute_difference_values))

        elif normalization_method == "z_score":
            # Z-Score Normalization
            inside_intensity_values = (inside_intensity_values - np.mean(inside_intensity_values)) / np.std(inside_intensity_values)
            outside_intensity_values = (outside_intensity_values - np.mean(outside_intensity_values)) / np.std(outside_intensity_values)
            total_intensity_values = (total_intensity_values - np.mean(total_intensity_values)) / np.std(total_intensity_values)
            absolute_difference_values = (absolute_difference_values - np.mean(absolute_difference_values)) / np.std(absolute_difference_values)

        elif normalization_method == "total":
            # Normalization relative to Total Intensity
            inside_intensity_values /= total_intensity_values
            outside_intensity_values /= total_intensity_values
            absolute_difference_values /= total_intensity_values

        elif normalization_method == "log":
            # Log Transformation
            inside_intensity_values = np.log(inside_intensity_values + 1)  # Adding 1 to avoid log(0)
            outside_intensity_values = np.log(outside_intensity_values + 1)
            total_intensity_values = np.log(total_intensity_values + 1)
            absolute_difference_values = np.log(absolute_difference_values + 1)

        else:
            raise ValueError("Invalid normalization method selected!")

        # Apply smoothing if the checkbox is selected
        if self.smooth_var.get():
            window_size = self.smooth_slider.get()
            inside_intensity_values = self.smooth_intensity_values(inside_intensity_values, window_size)
            outside_intensity_values = self.smooth_intensity_values(outside_intensity_values, window_size)
            total_intensity_values = self.smooth_intensity_values(total_intensity_values, window_size)
            absolute_difference_values = self.smooth_intensity_values(absolute_difference_values, window_size)

        # Create an interactive Plotly plot
        fig = go.Figure()

        # Plot based on selected checkboxes
        if self.plot_inside_var.get():
            fig.add_trace(go.Scatter(x=list(range(1, len(inside_intensity_values) + 1)), y=inside_intensity_values,
                                    mode='lines', name='Inside Gaussian Region', line=dict(color='blue')))
        if self.plot_outside_var.get():
            fig.add_trace(go.Scatter(x=list(range(1, len(outside_intensity_values) + 1)), y=outside_intensity_values,
                                    mode='lines', name='Outside Gaussian Region', line=dict(color='red')))
        if self.plot_total_var.get():
            fig.add_trace(go.Scatter(x=list(range(1, len(total_intensity_values) + 1)), y=total_intensity_values,
                                    mode='lines', name='Total Intensity', line=dict(color='green')))
        if self.plot_difference_var.get():
            fig.add_trace(go.Scatter(x=list(range(1, len(absolute_difference_values) + 1)), y=absolute_difference_values,
                                    mode='lines', name='|Inside - Outside|', line=dict(color='magenta')))

        fig.update_layout(
            title=f'Normalized ({normalization_method}) Sum of Intensities vs. Frame (CB Gaussian Sigma = {sigma_level})',
            xaxis_title='Frame Number',
            yaxis_title='Normalized Sum of Intensity',
            legend_title='Intensity Types',
            hovermode='x unified'
        )

        fig.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CenterBeamIntensityApp(root)
    root.mainloop()
