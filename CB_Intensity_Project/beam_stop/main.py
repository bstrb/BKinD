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
from fit_annulus_to_data import fit_annulus_to_data, annular_mask
from fit_gaussian_tail import fit_gaussian_tail
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
        self.outer_radius = None

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
            'annular_mask': annular_mask,
            'calculate_annular_region_intensity': calculate_annular_region_intensity,
            'plot_intensities': plot_intensities,
            'fit_gaussian_tail': fit_gaussian_tail,
            'normalize_intensities': normalize_intensities,
            'toggle_smoothing': toggle_smoothing,
            'smooth_intensity_values': smooth_intensity_values
        }

    def calculate_intensities(self):
        img_directory = os.path.dirname(self.file_path)
        img_files = glob.glob(os.path.join(img_directory, '*.img'))

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        if not img_files:
            messagebox.showwarning("No .img files found", "No .img files were found in the directory.")
            return

        img_files.sort(key=natural_sort_key)

        inside_intensity_values = []
        outside_intensity_values = []
        total_intensity_values = []
        frame_numbers = []

        first_img_data = fabio.open(img_files[0]).data
        self.fit_annulus_to_data(first_img_data)  # Fit the initial annulus
        self.fit_gaussian_tail(first_img_data, self.center, self.inner_radius)  # Fit Gaussian tail

        for idx, img_file in enumerate(img_files, start=1):
            img_data = fabio.open(img_file).data

            # Create mask for dark circle and Gaussian tail
            y, x = np.indices(img_data.shape)
            dist_from_center = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
            circle_mask = dist_from_center <= self.inner_radius
            gaussian_mask = ~circle_mask

            inside_intensity = np.sum(img_data[circle_mask])
            outside_intensity = np.sum(img_data[gaussian_mask])
            total_intensity = np.sum(img_data)

            inside_intensity_values.append(inside_intensity)
            outside_intensity_values.append(outside_intensity)
            total_intensity_values.append(total_intensity)
            frame_numbers.append(idx)

        inside_intensity_values = np.array(inside_intensity_values, dtype=np.float64)
        outside_intensity_values = np.array(outside_intensity_values, dtype=np.float64)
        total_intensity_values = np.array(total_intensity_values, dtype=np.float64)
        absolute_difference_values = np.abs(inside_intensity_values - outside_intensity_values)

        self.normalize_intensities(inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values)

        output_csv_path = os.path.join(os.path.dirname(img_directory), "frame_cbi.csv")
        with open(output_csv_path, 'w', newline='') as csvfile:
            csvfile.write("Frame,CBI\n")
            for frame, cbi in zip(frame_numbers, inside_intensity_values):
                csvfile.write(f"{frame},{cbi}\n")

        messagebox.showinfo("Success", f"CBI values saved to {output_csv_path}")
        self.plot_intensities(img_directory, inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values)


if __name__ == "__main__":
    root = tk.Tk()
    app = AnnularBeamIntensityApp(root)
    root.mainloop()
