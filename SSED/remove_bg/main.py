import tkinter as tk
from tkinter import filedialog, messagebox
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

# class MaskCreator:
#     def __init__(self, img):
#         self.img = img
#         self.circle = None
#         self.rect = None
#         self.center = None
#         self.radius = None
#         self.angle = None

#     def create_mask(self):
#         # Create a figure and axis to display the image
#         self.fig, self.ax = plt.subplots()
#         self.ax.imshow(self.img, cmap='gray')

#         # Connect the onclick event to define circle and strip
#         self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

#         # Add a button for confirming the mask
#         ax_confirm = plt.axes([0.7, 0.01, 0.1, 0.05])
#         self.btn_confirm = Button(ax_confirm, 'Confirm')
#         self.btn_confirm.on_clicked(self.confirm)

#         plt.show()

#     def onclick(self, event):
#         if event.inaxes != self.ax:
#             return
#         if self.circle is None:
#             # Define circle center on first click
#             self.center = (event.xdata, event.ydata)
#             self.circle = Circle(self.center, 1, color='r', fill=False)
#             self.ax.add_patch(self.circle)
#         elif self.radius is None:
#             # Define circle radius on second click
#             self.radius = np.sqrt((event.xdata - self.center[0])**2 + (event.ydata - self.center[1])**2)
#             self.circle.set_radius(self.radius)
#             plt.draw()
#         elif self.rect is None:
#             # Define strip mask on third click (starting point)
#             self.rect_start = (event.xdata, event.ydata)

#     def confirm(self, event):
#         if self.center is None or self.radius is None:
#             messagebox.showerror("Error", "Please define the circle mask first.")
#             return

#         # Optionally, get the strip angle and width from the user
#         self.angle = float(input("Enter the angle of the strip in degrees: "))
#         self.strip_width = float(input("Enter the strip width in pixels: "))

#         # Create the strip mask
#         self.create_strip(self.center, self.radius, self.angle, self.strip_width)
#         plt.draw()

#     def create_strip(self, center, radius, angle, width):
#         angle_rad = np.deg2rad(angle)
#         x0, y0 = center

#         # Define the rectangle representing the strip
#         x_start = x0 - radius
#         y_start = y0 - width / 2
#         self.rect = Rectangle((-10, y_start), 1200, width, angle=angle, edgecolor='blue', fill=False)
#         self.ax.add_patch(self.rect)

#     def get_mask(self, shape):
#         """Return a combined mask with the circle and strip."""
#         # Create circle mask
#         Y, X = np.ogrid[:shape[0], :shape[1]]
#         dist_from_center = np.sqrt((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2)
#         circle_mask = dist_from_center <= self.radius

#         # Create strip mask
#         angle_rad = np.deg2rad(self.angle)
#         strip_mask = (np.abs((X - self.center[0]) * np.cos(angle_rad) + (Y - self.center[1]) * np.sin(angle_rad)) < self.strip_width / 2)

#         # Combine masks
#         return circle_mask | strip_mask
class MaskCreator:
    def __init__(self, img):
        self.img = img
        self.circle = None
        self.rect = None
        self.circle_center = None
        self.rect_center = None
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
            self.circle_center = (event.xdata, event.ydata)
            self.circle = Circle(self.circle_center, 1, color='r', fill=False)
            self.ax.add_patch(self.circle)
        elif self.radius is None:
            # Define circle radius on second click
            self.radius = np.sqrt((event.xdata - self.circle_center[0]) ** 2 + (event.ydata - self.circle_center[1]) ** 2)
            self.circle.set_radius(self.radius)
            plt.draw()
        elif self.rect_center is None:
            # Define rectangle center on third click
            self.rect_center = (event.xdata, event.ydata)
            self.rect = Rectangle(self.rect_center, 1, 1, color='b', fill=False)  # Placeholder rectangle
            self.ax.add_patch(self.rect)
            plt.draw()

    def confirm(self, event):
        """Let the user define the strip and create the mask."""
        if self.circle_center is None or self.radius is None or self.rect_center is None:
            print("Please select the circle center, circle radius, and rectangle center.")
            return

        # Now, ask for the angle and width of the rectangle
        self.angle = float(input("Enter the angle of the rectangle in degrees: "))
        self.strip_width = float(input("Enter the strip width in pixels: "))

        # Create the strip mask using the new rectangle center
        self.create_strip(self.rect_center, self.radius, self.angle, self.strip_width)
        plt.draw()

    def create_strip(self, center, radius, angle, width):
        """Create a rectangular region extending from the circle."""
        angle_rad = np.deg2rad(angle)
        x_center, y_center = center

        # Define rectangle width and height
        rect_width = 2 * radius
        rect_height = max(self.img.shape)

        # Calculate the starting point (top-left corner) of the rectangle
        x0 = x_center - rect_width / 2
        y0 = y_center - rect_height / 2

        # Calculate the coordinates of the rotated rectangle's corners
        corners = np.array([
            [x0, y0],
            [x0 + rect_width, y0],
            [x0 + rect_width, y0 + rect_height],
            [x0, y0 + rect_height]
        ])

        # Rotation matrix
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        rotated_corners = (corners - np.array([x_center, y_center])) @ rot_matrix + np.array([x_center, y_center])

        # Draw the rectangle
        self.rect.set_xy(rotated_corners[0])
        self.rect.set_width(rect_width)
        self.rect.set_height(rect_height)
        self.rect.angle = angle

        # Mask the circle and rectangle
        circle_mask = self.create_circle_mask(self.img.shape, self.circle_center, self.radius)
        rect_mask = self.create_rect_mask(self.img.shape, rotated_corners)
        combined_mask = circle_mask | rect_mask  # Combine the two masks for final use

        return combined_mask

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
            mask_creator = MaskCreator(self.image)
            mask_creator.create_mask()
        else:
            messagebox.showerror("Error", "No image loaded.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskingGUI(root)
    root.mainloop()
