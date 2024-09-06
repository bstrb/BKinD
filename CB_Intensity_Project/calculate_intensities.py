# calculate_intensities.py

import re
import os
import glob
import fabio
import numpy as np

from tkinter import messagebox


from calculate_gaussian_region_intensity import calculate_gaussian_region_intensity
from normalize_intensities import normalize_intensities
from plot_intensities import plot_intensities
from smooth_intensity_values import smooth_intensity_values

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

    # Retrieve the sigma level entered by the user
    sigma_level = float(self.sigma_level_entry.get())

    inside_intensity_values = []
    outside_intensity_values = []
    total_intensity_values = []
    frame_numbers = []

    for idx, img_file in enumerate(img_files, start=1):
        # Load the image
        img_data = fabio.open(img_file).data
        
        # Calculate sum of intensities inside, outside, and total
        inside_intensity = calculate_gaussian_region_intensity(self, img_data, sigma_level=sigma_level, region="inside")
        outside_intensity = calculate_gaussian_region_intensity(self, img_data, sigma_level=sigma_level, region="outside")
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

    # Normalize the intensity values based on the selected normalization method
    normalize_intensities(self, inside_intensity_values, outside_intensity_values, total_intensity_values)

    # Save the normalized CBI values to a CSV file
    output_csv_path = os.path.join(os.path.dirname(img_directory), "frame_cbi.csv")
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvfile.write("Frame,CBI\n")
        for frame, cbi in zip(frame_numbers, inside_intensity_values):
            csvfile.write(f"{frame},{cbi}\n")

    absolute_difference_values = np.abs(inside_intensity_values - outside_intensity_values)

    # Apply smoothing if the checkbox is selected
    if self.smooth_var.get():
        window_size = self.smooth_slider.get()
        inside_intensity_values = smooth_intensity_values(inside_intensity_values, window_size)
        outside_intensity_values = smooth_intensity_values(outside_intensity_values, window_size)
        total_intensity_values = smooth_intensity_values(total_intensity_values, window_size)
        absolute_difference_values = smooth_intensity_values(absolute_difference_values, window_size)

    # Create and display the plot
    plot_intensities(self, img_directory, inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values)
