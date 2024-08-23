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

        # Radio buttons to select inside or outside the circle
        self.measurement_option = tk.StringVar(value="outside")
        self.inside_radio = tk.Radiobutton(self.root, text="Measure Inside Circle", variable=self.measurement_option, value="inside")
        self.inside_radio.pack(pady=5)
        self.outside_radio = tk.Radiobutton(self.root, text="Measure Outside Circle", variable=self.measurement_option, value="outside")
        self.outside_radio.pack(pady=5)

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

    def calculate_inside_circle_intensity(self, img_data):
        """Calculate the mean intensity inside the central circular region."""
        row, column = self.center
        Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
        # Create a circular mask
        dist_from_center = np.sqrt((X - column)**2 + (Y - row)**2)
        mask = dist_from_center <= self.radius  # Mask to include areas inside the circle
        
        # Calculate the mean intensity inside the central circle
        inside_circle_intensity = np.mean(img_data[mask])
        return inside_circle_intensity

    def calculate_outside_circle_intensity(self, img_data):
        """Calculate the mean intensity outside the central circular region."""
        row, column = self.center
        Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
        # Create a circular mask
        dist_from_center = np.sqrt((X - column)**2 + (Y - row)**2)
        mask = dist_from_center > self.radius  # Invert the mask to include areas outside the circle
        
        # Calculate the mean intensity outside the central circle
        outside_circle_intensity = np.mean(img_data[mask])
        return outside_circle_intensity
    
    def calculate_intensities(self):
        # Get the directory containing the selected file
        img_directory = os.path.dirname(self.file_path)
        img_files = glob.glob(os.path.join(img_directory, '*.img'))
        
        if not img_files:
            messagebox.showwarning("No .img files found", "No .img files were found in the directory.")
            return

        intensity_values = []

        for img_file in img_files:
            # Load the image
            img_data = fabio.open(img_file).data
            
            # Measure intensity based on selected option
            if self.measurement_option.get() == "inside":
                intensity = self.calculate_inside_circle_intensity(img_data)
                print(f"File: {os.path.basename(img_file)}, Inside Circle Intensity: {intensity:.2f}")
            else:
                intensity = self.calculate_outside_circle_intensity(img_data)
                print(f"File: {os.path.basename(img_file)}, Outside Circle Intensity: {intensity:.2f}")

            intensity_values.append(intensity)

        # Convert to numpy array for further processing
        intensity_values = np.array(intensity_values)
        
        # Handle zero values in the intensity data
        for i in range(1, len(intensity_values) - 1):
            if intensity_values[i] == 0.0:
                if intensity_values[i+1] == 0.0:
                    intensity_values[i] = intensity_values[i-1]  # Replace with previous valid value
                else:
                    intensity_values[i] = (intensity_values[i-1] + intensity_values[i+1]) / 2  # Replace with mean of surrounding values

        # Handle the case where the last values are 0.0
        if intensity_values[-1] == 0.0:
            intensity_values[-1] = intensity_values[-2]  # Replace with the previous value

        if intensity_values[-2] == 0.0:  # If the second-to-last value is also 0
            intensity_values[-2] = intensity_values[-3]  # Replace with the previous valid value

        # Apply smoothing if the checkbox is selected
        if self.smooth_var.get():
            window_size = self.smooth_slider.get()
            intensity_values = self.smooth_intensity_values(intensity_values, window_size)
       
        # Determine the title based on the measurement option
        if self.measurement_option.get() == "inside":
            title = 'Inside Circle Intensity vs. Frame'
        else:
            title = 'Outside Circle Intensity vs. Frame'

        # Plot the intensity over the frames
        plt.figure(figsize=(10, 6))
        plt.plot(intensity_values, linestyle='-', color='b')
        plt.title(title)
        plt.xlabel('Frame Number')
        plt.ylabel('Circle Intensity')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CenterBeamIntensityApp(root)
    root.mainloop()
