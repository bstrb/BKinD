import tkinter as tk
from tkinter import filedialog, messagebox
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

class MaskCreator:
    def __init__(self, img):
        self.img = img
        self.circle = None
        self.rect = None
        self.center = None
        self.radius = None
        self.angle = None

    def create_mask(self):
        # Create a figure and axis to display the image
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img, cmap='gray')

        # Connect the onclick event to define circle and strip
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Add a button for confirming the mask
        ax_confirm = plt.axes([0.7, 0.01, 0.1, 0.05])
        self.btn_confirm = Button(ax_confirm, 'Confirm')
        self.btn_confirm.on_clicked(self.confirm)

        plt.show()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        if self.circle is None:
            # Define circle center on first click
            self.center = (event.xdata, event.ydata)
            self.circle = Circle(self.center, 1, color='darkred', fill=False)
            self.ax.add_patch(self.circle)
        elif self.radius is None:
            # Define circle radius on second click
            self.radius = np.sqrt((event.xdata - self.center[0])**2 + (event.ydata - self.center[1])**2)
            self.circle.set_radius(self.radius)
            plt.draw()
        elif self.rect is None:
            # Define strip mask on third click (starting point)
            self.rect_start = (event.xdata, event.ydata)

    def confirm(self, event):
        if self.center is None or self.radius is None:
            messagebox.showerror("Error", "Please define the circle mask first.")
            return

        # Optionally, get the strip angle and width from the user
        self.angle = float(input("Enter the angle of the strip in degrees: "))
        self.strip_width = float(input("Enter the strip width in pixels: "))

        # Create the strip mask
        self.create_strip(self.center, self.angle, self.strip_width)
        self.combined_mask = self.get_mask(self.img.shape)  # Store the created mask
        plt.draw()


    def create_strip(self, center, angle, width):
        x0, y0 = center

        # Define the rectangle representing the strip
        x_start = x0
        y_start = y0 - width / 2
        self.rect = Rectangle((x_start, y_start), 520, width, angle=angle, edgecolor='darkblue', fill=False)
        self.rect2 = Rectangle((x_start, y_start), -520, width, angle=angle, edgecolor='darkblue', fill=False)
        self.ax.add_patch(self.rect)
        self.ax.add_patch(self.rect2)

    def get_mask(self, shape):
        """Return a combined mask with the circle and strip."""
        # Create circle mask
        Y, X = np.ogrid[:shape[0], :shape[1]]
        dist_from_center = np.sqrt((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2)
        circle_mask = dist_from_center <= self.radius

        # Create strip mask
        angle_rad = np.deg2rad(self.angle)
        strip_mask = (np.abs((X - self.center[0]) * np.cos(angle_rad) + (Y - self.center[1]) * np.sin(angle_rad)) < self.strip_width / 2)

        # Combine masks
        return circle_mask | strip_mask
    
    def get_combined_mask(self):
        return self.combined_mask

class MaskingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 Frame Masking")
        self.h5_file_path = None
        self.image = None

        # Load button
        self.load_button = tk.Button(root, text="Load HDF5 File", command=self.load_h5_file)
        self.load_button.pack(pady=10)

        # Create mask button (initially disabled)
        self.mask_button = tk.Button(root, text="Create Mask", command=self.create_mask, state=tk.DISABLED)
        self.mask_button.pack(pady=10)

        # Save mask button (initially disabled)
        self.save_button = tk.Button(root, text="Save Mask", command=self.save_mask, state=tk.DISABLED)
        self.save_button.pack(pady=10)

    def load_h5_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
        if file_path:
            self.h5_file_path = file_path
            self.load_random_frame()
            self.mask_button.config(state=tk.NORMAL)

    def load_random_frame(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            dataset = f['/entry/data/images']
            total_frames = dataset.shape[0]
            random_index = np.random.randint(0, total_frames)
            self.image = dataset[random_index, :, :]

    def create_mask(self):
        if self.image is not None:
            self.mask_creator = MaskCreator(self.image)
            self.mask_creator.create_mask()
            self.save_button.config(state=tk.NORMAL)  # Enable save button after mask creation
        else:
            messagebox.showerror("Error", "No image loaded.")

    def save_mask(self):
        if self.mask_creator:
            # Get the combined mask
            # mask = self.mask_creator.get_combined_mask()
            mask = ~self.mask_creator.get_combined_mask()  # Invert the mask

            
            # Rotate the mask 90 degrees
            rotated_mask = np.rot90(mask)

            # Ask for the save location
            output_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5")])
            if output_path:
                with h5py.File(output_path, 'w') as f:
                    f.create_dataset('mask', data=rotated_mask, dtype='uint8')
                messagebox.showinfo("Success", f"Mask saved to {output_path}")
        else:
            messagebox.showerror("Error", "No mask to save.")


if __name__ == "__main__":
    root = tk.Tk()
    app = MaskingGUI(root)
    root.mainloop()
