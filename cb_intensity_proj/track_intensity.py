# track_cbi.py

import tkinter as tk
from tkinter import filedialog, messagebox
import fabio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os
import glob

class CenterBeamIntensityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Center Beam Intensity Calculator")
        
        self.file_path = ""
        self.center = None
        self.radius = None

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Button to browse and select the initial .img file
        self.browse_button = tk.Button(self.root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        # # Radio buttons to select inside or outside the circle
        # self.measurement_option = tk.StringVar(value="outside")
        # self.inside_radio = tk.Radiobutton(self.root, text="Measure Inside Circle", variable=self.measurement_option, value="inside")
        # self.inside_radio.pack(pady=5)
        # self.outside_radio = tk.Radiobutton(self.root, text="Measure Outside Circle", variable=self.measurement_option, value="outside")
        # self.outside_radio.pack(pady=5)

        # Checkbox to enable/disable smoothing
        self.smooth_var = tk.BooleanVar(value=False)
        self.smooth_checkbox = tk.Checkbutton(self.root, text="Smoothen Curve", variable=self.smooth_var, command=self.toggle_smoothing)
        self.smooth_checkbox.pack(pady=10)

        # Slider to adjust smoothing window size
        self.smooth_slider = tk.Scale(self.root, from_=3, to=51, orient=tk.HORIZONTAL, label="Smoothing Window Size", state=tk.DISABLED)
        self.smooth_slider.pack(pady=10)

        # Button to calculate the center beam intensity in the folder
        self.calculate_button = tk.Button(self.root, text="Calculate Intensities", command=self.calculate_intensities, state=tk.DISABLED)
        self.calculate_button.pack(pady=10)

    def toggle_smoothing(self):
        """Enable or disable the smoothing slider based on the checkbox state."""
        if self.smooth_var.get():
            self.smooth_slider.config(state=tk.NORMAL)
        else:
            self.smooth_slider.config(state=tk.DISABLED)

    def browse_image(self):
        # Open file dialog to select an .img file
        self.file_path = filedialog.askopenfilename(filetypes=[("IMG files", "*.img")], title="Select an .img file")
        
        if not self.file_path:
            messagebox.showwarning("No file selected", "Please select an .img file.")
            return

        # Load and display the image
        self.load_and_display_image()

    def load_and_display_image(self):
        img_data = fabio.open(self.file_path).data

        # Plot the image and allow the user to select the center and radius of the circle
        fig, ax = plt.subplots()
        ax.imshow(img_data, cmap='gray')
        plt.colorbar(ax.imshow(img_data, cmap='gray'))

        plt.title('Click to select the center of the circle')
        self.center = plt.ginput(1)[0]  # User clicks to select the center of the circle
        
        plt.title('Click to select the edge of the circle')
        edge_point = plt.ginput(1)[0]  # User clicks to select the edge of the circle

        # Calculate radius
        self.radius = np.sqrt((self.center[0] - edge_point[0])**2 + (self.center[1] - edge_point[1])**2)

        # Draw the circle
        circle = Circle(self.center, self.radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        plt.title('Selected Center and Radius')
        plt.show()

        print(f"Selected Center: {self.center}")
        print(f"Selected Radius: {self.radius}")

        # Enable the calculate button after selecting the center and radius
        self.calculate_button.config(state=tk.NORMAL)

    def smooth_intensity_values(self, intensity_values, window_size):
        """Smooth the intensity values using a moving average."""
        smoothed_values = np.convolve(intensity_values, np.ones(window_size) / window_size, mode='valid')
        
        # To maintain the same length as the original array, we'll pad the start of the smoothed array
        pad_size = len(intensity_values) - len(smoothed_values)
        return np.pad(smoothed_values, (pad_size // 2, pad_size - pad_size // 2), mode='edge')

    # def calculate_inside_circle_intensity(self, img_data):
    #     """Calculate the mean intensity inside the central circular region."""
    #     row, column = self.center
    #     Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
    #     # Create a circular mask
    #     dist_from_center = np.sqrt((X - column)**2 + (Y - row)**2)
    #     mask = dist_from_center <= self.radius  # Mask to include areas inside the circle
        
    #     # Calculate the mean intensity inside the central circle
    #     inside_circle_intensity = np.sum(img_data[mask])
    #     return inside_circle_intensity

    # def calculate_outside_circle_intensity(self, img_data):
    #     """Calculate the mean intensity outside the central circular region."""
    #     row, column = self.center
    #     Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
    #     # Create a circular mask
    #     dist_from_center = np.sqrt((X - column)**2 + (Y - row)**2)
    #     mask = dist_from_center > self.radius  # Invert the mask to include areas outside the circle
        
    #     # Calculate the mean intensity outside the central circle
    #     outside_circle_intensity = np.sum(img_data[mask])
    #     return outside_circle_intensity
    
    # def calculate_intensities(self):
    #     # Get the directory containing the selected file
    #     img_directory = os.path.dirname(self.file_path)
    #     img_files = glob.glob(os.path.join(img_directory, '*.img'))
        
    #     if not img_files:
    #         messagebox.showwarning("No .img files found", "No .img files were found in the directory.")
    #         return

    #     inside_intensity_values = []
    #     outside_intensity_values = []
    #     total_intensity_values = []

    #     for img_file in img_files:
    #         # Load the image
    #         img_data = fabio.open(img_file).data
            
    #         # Calculate intensities
    #         inside_intensity = self.calculate_inside_circle_intensity(img_data)
    #         outside_intensity = self.calculate_outside_circle_intensity(img_data)
    #         total_intensity = np.sum(img_data)

    #         # Store the intensities
    #         inside_intensity_values.append(inside_intensity)
    #         outside_intensity_values.append(outside_intensity)
    #         total_intensity_values.append(total_intensity)

    #         print(f"File: {os.path.basename(img_file)}, Inside Circle Intensity: {inside_intensity:.2f}, Outside Circle Intensity: {outside_intensity:.2f}, Total Intensity: {total_intensity:.2f}")

    #     # Convert to numpy arrays for further processing
    #     inside_intensity_values = np.array(inside_intensity_values)
    #     outside_intensity_values = np.array(outside_intensity_values)
    #     total_intensity_values = np.array(total_intensity_values)
        
    #     # Handle zero values in the intensity data
    #     for i in range(1, len(inside_intensity_values) - 1):
    #         if inside_intensity_values[i] == 0.0:
    #             if inside_intensity_values[i+1] == 0.0:
    #                 inside_intensity_values[i] = inside_intensity_values[i-1]  # Replace with previous valid value
    #             else:
    #                 inside_intensity_values[i] = (inside_intensity_values[i-1] + inside_intensity_values[i+1]) / 2  # Replace with mean of surrounding values

    #         if outside_intensity_values[i] == 0.0:
    #             if outside_intensity_values[i+1] == 0.0:
    #                 outside_intensity_values[i] = outside_intensity_values[i-1]  # Replace with previous valid value
    #             else:
    #                 outside_intensity_values[i] = (outside_intensity_values[i-1] + outside_intensity_values[i+1]) / 2  # Replace with mean of surrounding values

    #         if total_intensity_values[i] == 0.0:
    #             if total_intensity_values[i+1] == 0.0:
    #                 total_intensity_values[i] = total_intensity_values[i-1]  # Replace with previous valid value
    #             else:
    #                 total_intensity_values[i] = (total_intensity_values[i-1] + total_intensity_values[i+1]) / 2  # Replace with mean of surrounding values

    #     # Apply smoothing if the checkbox is selected
    #     if self.smooth_var.get():
    #         window_size = self.smooth_slider.get()
    #         inside_intensity_values = self.smooth_intensity_values(inside_intensity_values, window_size)
    #         outside_intensity_values = self.smooth_intensity_values(outside_intensity_values, window_size)
    #         total_intensity_values = self.smooth_intensity_values(total_intensity_values, window_size)
        
    #     # Plot all three intensity curves
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(inside_intensity_values, linestyle='-', color='b', label='Inside Circle')
    #     plt.plot(outside_intensity_values, linestyle='-', color='r', label='Outside Circle')
    #     plt.plot(total_intensity_values, linestyle='-', color='g', label='Total Intensity')
    #     plt.title('Intensity vs. Frame')
    #     plt.xlabel('Frame Number')
    #     plt.ylabel('Intensity')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    def calculate_inside_circle_intensity(self, img_data):
        """Calculate the sum of intensity inside the central circular region."""
        row, column = self.center
        Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
        # Create a circular mask
        dist_from_center = np.sqrt((X - column)**2 + (Y - row)**2)
        mask = dist_from_center <= self.radius  # Mask to include areas inside the circle
        
        # Calculate the sum of intensity inside the central circle
        inside_circle_intensity = np.sum(img_data[mask])
        return inside_circle_intensity

    def calculate_outside_circle_intensity(self, img_data):
        """Calculate the sum of intensity outside the central circular region."""
        row, column = self.center
        Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
        # Create a circular mask
        dist_from_center = np.sqrt((X - column)**2 + (Y - row)**2)
        mask = dist_from_center > self.radius  # Invert the mask to include areas outside the circle
        
        # Calculate the sum of intensity outside the central circle
        outside_circle_intensity = np.sum(img_data[mask])
        return outside_circle_intensity
    
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

        for img_file in img_files:
            # Load the image
            img_data = fabio.open(img_file).data
            
            # Calculate sum of intensities inside, outside, and total
            inside_intensity = self.calculate_inside_circle_intensity(img_data)
            outside_intensity = self.calculate_outside_circle_intensity(img_data)
            total_intensity = np.sum(img_data)  # Calculate the sum of all pixel intensities in the image

            inside_intensity_values.append(inside_intensity)
            outside_intensity_values.append(outside_intensity)
            total_intensity_values.append(total_intensity)

        # Convert to numpy arrays and cast to float64 for safe division
        inside_intensity_values = np.array(inside_intensity_values, dtype=np.float64)
        outside_intensity_values = np.array(outside_intensity_values, dtype=np.float64)
        total_intensity_values = np.array(total_intensity_values, dtype=np.float64)

        # Normalize the intensities to their respective sums
        inside_intensity_values /= np.sum(inside_intensity_values)
        outside_intensity_values /= np.sum(outside_intensity_values)
        total_intensity_values /= np.sum(total_intensity_values)

        # Apply smoothing if the checkbox is selected
        if self.smooth_var.get():
            window_size = self.smooth_slider.get()
            inside_intensity_values = self.smooth_intensity_values(inside_intensity_values, window_size)
            outside_intensity_values = self.smooth_intensity_values(outside_intensity_values, window_size)
            total_intensity_values = self.smooth_intensity_values(total_intensity_values, window_size)

        # Plot the normalized intensities over the frames
        plt.figure(figsize=(10, 6))
        plt.plot(inside_intensity_values, linestyle='-', color='b', label='Inside Circle')
        plt.plot(outside_intensity_values, linestyle='-', color='r', label='Outside Circle')
        plt.plot(total_intensity_values, linestyle='-', color='g', label='Total Intensity')
        plt.title('Normalized Sum of Intensities vs. Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Normalized Sum of Intensity')
        plt.legend()
        plt.grid(True)
        plt.show()
    
if __name__ == "__main__":
    root = tk.Tk()
    app = CenterBeamIntensityApp(root)
    root.mainloop()
