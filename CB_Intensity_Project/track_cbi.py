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

        # Button to calculate the center beam intensity in the folder
        self.calculate_button = tk.Button(self.root, text="Calculate Intensities", command=self.calculate_intensities, state=tk.DISABLED)
        self.calculate_button.pack(pady=10)

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

    def calculate_center_beam_intensity(self, img_data):
        """Calculate the mean intensity of the central circular region."""
        row, column = self.center
        Y, X = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
        
        # Create a circular mask
        dist_from_center = np.sqrt((X - column)**2 + (Y - row)**2)
        mask = dist_from_center <= self.radius
        
        # Calculate the mean intensity of the central circle (center beam intensity)
        center_beam_intensity = np.mean(img_data[mask])
        return center_beam_intensity

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
            
            # Calculate center beam intensity
            intensity = self.calculate_center_beam_intensity(img_data)
            intensity_values.append(intensity)
            print(f"File: {os.path.basename(img_file)}, Center Beam Intensity: {intensity:.2f}")

        # Convert to numpy array for further processing
        intensity_values = np.array(intensity_values)
        
        # Assuming intensity_values is a NumPy array with the calculated intensities
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


        # Plot the center beam intensity over the frames
        plt.figure(figsize=(10, 6))
        plt.plot(intensity_values, linestyle='-', color='b')
        plt.title('Center Beam Intensity vs. Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Center Beam Intensity')
        plt.grid(True)
        plt.show()

        # Optionally save results to a file
        save_option = messagebox.askyesno("Save Results", "Do you want to save the intensity results to a file?")
        if save_option:
            output_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="Save Intensity Results")
            if output_path:
                # Combine frame numbers with intensity values
                frame_numbers = np.arange(1, len(intensity_values) + 1)  # Assuming frame numbers start at 1
                results = np.column_stack((frame_numbers, intensity_values))
                
                # Save the results to a file with appropriate header
                np.savetxt(output_path, results, header="Frame Number\tCenter Beam Intensities", fmt="%d\t%.2f")
                messagebox.showinfo("Saved", f"Results saved to {output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CenterBeamIntensityApp(root)
    root.mainloop()
