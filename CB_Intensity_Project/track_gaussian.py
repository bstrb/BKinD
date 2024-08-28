# track_gaussian.py

import tkinter as tk
from tkinter import messagebox
import fabio
import numpy as np
import os
import glob
import plotly.graph_objects as go


class CenterBeamIntensityApp:

    from create_widgets import create_widgets
    from browse_image import browse_image
    from load_and_display_image import load_and_display_image
    from gaussian_2d import gaussian_2d
    from fit_gaussian import fit_gaussian
    from calculate_gaussian_region_intensity import calculate_gaussian_region_intensity
    
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

    def smooth_intensity_values(self, intensity_values, window_size):
        """Smooth the intensity values using a moving average."""
        smoothed_values = np.convolve(intensity_values, np.ones(window_size) / window_size, mode='valid')
        
        # To maintain the same length as the original array, we'll pad the start of the smoothed array
        pad_size = len(intensity_values) - len(smoothed_values)
        return np.pad(smoothed_values, (pad_size // 2, pad_size - pad_size // 2), mode='edge')

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
            
            # # Calculate sum of intensities inside, outside, and total
            inside_intensity = self.calculate_gaussian_region_intensity(img_data, sigma_level=sigma_level, region="inside")
            outside_intensity = self.calculate_gaussian_region_intensity(img_data, sigma_level=sigma_level, region="outside")
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
