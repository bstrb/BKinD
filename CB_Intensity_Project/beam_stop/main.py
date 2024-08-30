# main.py

import tkinter as tk
from tkinter import messagebox
import os
import glob
import re
import fabio
import numpy as np

# Import all functions from other modules
from create_widgets import create_widgets
from browse_image import browse_image
from load_and_display_image import load_and_display_image
from fit_gaussian_tail import fit_gaussian_tail
from fit_annulus_to_data import fit_annulus_to_data, dark_circle_mask, gaussian_tail_mask
from calculate_annular_region_intensity import calculate_annular_region_intensity
from plot_intensities import plot_intensities
from normalize_intensities import normalize_intensities
from toggle_smoothing import toggle_smoothing
from smooth_intensity_values import smooth_intensity_values

class AnnularBeamIntensityApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Annular Beam Intensity Calculator")
        
        self.file_path = ""
        self.center = None
        self.inner_radius = None
        self.sigma = None  # Store sigma for the Gaussian tail

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
            'fit_annulus_to_data': fit_annulus_to_data,
            'fit_gaussian_tail': fit_gaussian_tail,
            'dark_circle_mask': dark_circle_mask,
            'gaussian_tail_mask': gaussian_tail_mask,
            'calculate_annular_region_intensity': calculate_annular_region_intensity,
            'plot_intensities': plot_intensities,
            'normalize_intensities': normalize_intensities,
            'toggle_smoothing': toggle_smoothing,
            'smooth_intensity_values': smooth_intensity_values
        }

    def calculate_intensities(self):
        # Get the directory containing the selected file
        img_directory = os.path.dirname(self.file_path)
        img_files = glob.glob(os.path.join(img_directory, '*.img'))
        
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        if not img_files:
            messagebox.showwarning("No .img files found", "No .img files were found in the directory.")
            return

        # Sort files by numerical order
        img_files.sort(key=natural_sort_key)

        # Initialize lists to store intensity values
        inside_intensity_values = []
        outside_intensity_values = []
        total_intensity_values = []
        frame_numbers = []

        # Use the fitted dark circle and Gaussian tail from the first image to process all images
        for idx, img_file in enumerate(img_files, start=1):
            img_data = fabio.open(img_file).data
            
            # Create the mask for the fitted annular region using the pre-calculated parameters
            dark_mask = dark_circle_mask(img_data.shape, self.center, self.inner_radius)
            tail_mask = gaussian_tail_mask(img_data.shape, self.center, self.inner_radius)
            combined_mask = dark_mask | tail_mask

            # Calculate sum of intensities inside and outside the annular region
            inside_intensity = np.sum(img_data[combined_mask])
            outside_intensity = np.sum(img_data[~combined_mask])
            total_intensity = np.sum(img_data)  # Calculate the sum of all pixel intensities in the image

            # Append the calculated values to their respective lists
            inside_intensity_values.append(inside_intensity)
            outside_intensity_values.append(outside_intensity)
            total_intensity_values.append(total_intensity)
            frame_numbers.append(idx)

        # Convert to numpy arrays and cast to float64 for safe division
        inside_intensity_values = np.array(inside_intensity_values, dtype=np.float64)
        outside_intensity_values = np.array(outside_intensity_values, dtype=np.float64)
        total_intensity_values = np.array(total_intensity_values, dtype=np.float64)
        absolute_difference_values = np.abs(inside_intensity_values - outside_intensity_values)

        # Normalize the intensity values based on the selected normalization method
        self.normalize_intensities(inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values)

        # Save the normalized CBI values to a CSV file
        output_csv_path = os.path.join(os.path.dirname(img_directory), "frame_cbi.csv")
        with open(output_csv_path, 'w', newline='') as csvfile:
            csvfile.write("Frame,CBI\n")
            for frame, cbi in zip(frame_numbers, inside_intensity_values):
                csvfile.write(f"{frame},{cbi}\n")

        messagebox.showinfo("Success", f"CBI values saved to {output_csv_path}")

        # Create and display the plot
        self.plot_intensities(img_directory, inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnularBeamIntensityApp(root)
    root.mainloop()
