import tkinter as tk
from tkinter import messagebox
import fabio
import numpy as np
import os
import glob
import re

# Import all functions from other modules
from create_widgets import create_widgets
from browse_image import browse_image
from load_and_display_image import load_and_display_image
from gaussian_2d import gaussian_2d
from fit_gaussian import fit_gaussian
from calculate_gaussian_region_intensity import calculate_gaussian_region_intensity
from plot_intensities import plot_intensities
from normalize_intensities import normalize_intensities
from toggle_smoothing import toggle_smoothing
from smooth_intensity_values import smooth_intensity_values

class CenterBeamIntensityApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Center Beam Intensity Calculator")
        
        self.file_path = ""
        self.center = None
        self.sigma_x = None
        self.sigma_y = None

        # Create GUI components
        self.create_widgets()

    def __getattr__(self, item):
        """Fallback for missing attributes to dynamically add them."""
        if item in self.__function_map__:
            func = self.__function_map__[item]
            setattr(self, item, func.__get__(self))
            return getattr(self, item)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    @property
    def __function_map__(self):
        return {
            'create_widgets': create_widgets,
            'browse_image': browse_image,
            'load_and_display_image': load_and_display_image,
            'gaussian_2d': gaussian_2d,
            'fit_gaussian': fit_gaussian,
            'calculate_gaussian_region_intensity': calculate_gaussian_region_intensity,
            'plot_intensities': plot_intensities,
            'normalize_intensities': normalize_intensities,
            'toggle_smoothing': toggle_smoothing,
            'smooth_intensity_values': smooth_intensity_values
        }
    
    # def calculate_intensities(self):
    #     # Get the directory containing the selected file
    #     img_directory = os.path.dirname(self.file_path)
    #     img_files = glob.glob(os.path.join(img_directory, '*.img'))
        
    #     def natural_sort_key(s):
    #         return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    #     if not img_files:
    #         messagebox.showwarning("No .img files found", "No .img files were found in the directory.")
    #         return

    #     # Sort files by numerical order
    #     img_files.sort(key=natural_sort_key)

    #     # Retrieve the sigma level entered by the user
    #     sigma_level = float(self.sigma_level_entry.get())

    #     inside_intensity_values = []
    #     outside_intensity_values = []
    #     total_intensity_values = []
    #     frame_numbers = []

    #     for idx, img_file in enumerate(img_files, start=1):
    #         # Load the image
    #         img_data = fabio.open(img_file).data
            
    #         # Calculate sum of intensities inside, outside, and total
    #         inside_intensity = self.calculate_gaussian_region_intensity(img_data, sigma_level=sigma_level, region="inside")
    #         outside_intensity = self.calculate_gaussian_region_intensity(img_data, sigma_level=sigma_level, region="outside")
    #         total_intensity = np.sum(img_data)  # Calculate the sum of all pixel intensities in the image

    #         # Append the calculated values to their respective lists
    #         inside_intensity_values.append(inside_intensity)
    #         outside_intensity_values.append(outside_intensity)
    #         total_intensity_values.append(total_intensity)
    #         frame_numbers.append(idx)

    #     # Convert to numpy arrays and cast to float64 for safe division
    #     inside_intensity_values = np.array(inside_intensity_values, dtype=np.float64)
    #     outside_intensity_values = np.array(outside_intensity_values, dtype=np.float64)
    #     total_intensity_values = np.array(total_intensity_values, dtype=np.float64)

    #     # Normalize the intensity values based on the selected normalization method
    #     self.normalize_intensities(inside_intensity_values, outside_intensity_values, total_intensity_values)

    #     # Save the normalized CBI values to a CSV file
    #     output_csv_path = os.path.join(os.path.dirname(img_directory), "frame_cbi.csv")
    #     with open(output_csv_path, 'w', newline='') as csvfile:
    #         csvfile.write("Frame,CBI\n")
    #         for frame, cbi in zip(frame_numbers, inside_intensity_values):
    #             csvfile.write(f"{frame},{cbi}\n")

    #     absolute_difference_values = np.abs(inside_intensity_values - outside_intensity_values)

    #     # Apply smoothing if the checkbox is selected
    #     if self.smooth_var.get():
    #         window_size = self.smooth_slider.get()
    #         inside_intensity_values = self.smooth_intensity_values(inside_intensity_values, window_size)
    #         outside_intensity_values = self.smooth_intensity_values(outside_intensity_values, window_size)
    #         total_intensity_values = self.smooth_intensity_values(total_intensity_values, window_size)
    #         absolute_difference_values = self.smooth_intensity_values(absolute_difference_values, window_size)

    #     # Create and display the plot
    #     self.plot_intensities(img_directory, inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values)
    
    from annular_model import calculate_annular_region_intensity, fit_annulus_to_data

    def calculate_intensities(self):
        # Get the directory containing the selected file
        img_directory = os.path.dirname(self.file_path)
        img_files = glob.glob(os.path.join(img_directory, '*.img'))
        
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        if not img_files:
            messagebox.showwarning("No .img files found", "No .img files were found in the directory.")
            return

        img_files.sort(key=natural_sort_key)

        # Retrieve the sigma level entered by the user
        sigma_level = float(self.sigma_level_entry.get())

        inside_intensity_values = []
        outside_intensity_values = []
        total_intensity_values = []
        frame_numbers = []

        for idx, img_file in enumerate(img_files, start=1):
            img_data = fabio.open(img_file).data
            
            if self.fitting_method.get() == "Gaussian":
                # Perform Gaussian fitting
                self.fit_gaussian(img_data)
                inside_intensity = self.calculate_gaussian_region_intensity(img_data, sigma_level=sigma_level, region="inside")
                total_intensity = np.sum(img_data)
                outside_intensity = total_intensity - inside_intensity

            elif self.fitting_method.get() == "Annular":
                # Perform annular fitting
                center, outer_radius, inner_radius = self.fit_annulus_to_data(img_data)
                inside_intensity = self.calculate_annular_region_intensity(img_data, outer_radius, inner_radius)
                total_intensity = np.sum(img_data)
                outside_intensity = total_intensity - inside_intensity

            inside_intensity_values.append(inside_intensity)
            outside_intensity_values.append(outside_intensity)
            total_intensity_values.append(total_intensity)
            frame_numbers.append(idx)

        inside_intensity_values = np.array(inside_intensity_values, dtype=np.float64)
        outside_intensity_values = np.array(outside_intensity_values, dtype=np.float64)
        total_intensity_values = np.array(total_intensity_values, dtype=np.float64)

        self.normalize_intensities(inside_intensity_values, outside_intensity_values, total_intensity_values)

        output_csv_path = os.path.join(os.path.dirname(img_directory), "frame_cbi.csv")
        with open(output_csv_path, 'w', newline='') as csvfile:
            csvfile.write("Frame,CBI\n")
            for frame, cbi in zip(frame_numbers, inside_intensity_values):
                csvfile.write(f"{frame},{cbi}\n")

        absolute_difference_values = np.abs(inside_intensity_values - outside_intensity_values)

        if self.smooth_var.get():
            window_size = self.smooth_slider.get()
            inside_intensity_values = self.smooth_intensity_values(inside_intensity_values, window_size)
            outside_intensity_values = self.smooth_intensity_values(outside_intensity_values, window_size)
            total_intensity_values = self.smooth_intensity_values(total_intensity_values, window_size)
            absolute_difference_values = self.smooth_intensity_values(absolute_difference_values, window_size)

        self.plot_intensities(img_directory, inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values)

if __name__ == "__main__":
    root = tk.Tk()
    app = CenterBeamIntensityApp(root)
    root.mainloop()