import os
import h5py
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

def create_h5_mask_from_png_inverted(png_path):
    """
    Converts a PNG mask to an HDF5 file, inverting the mask values:
    - White (255) → Valid (1)
    - Black (0) → Invalid (0)
    """
    try:
        # Load the PNG image
        img = Image.open(png_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)

        # Invert the mask: 1 for valid (was black), 0 for invalid (was white)
        mask = (img_array == 0).astype(np.uint8)

        # Define the output HDF5 file path
        output_h5_path = os.path.join(
            os.path.dirname(png_path),
            os.path.splitext(os.path.basename(png_path))[0] + "_inverted_mask.h5"
        )

        # Save the mask as an HDF5 file
        with h5py.File(output_h5_path, 'w') as h5_file:
            h5_file.create_dataset('mask', data=mask, dtype=np.uint8)

        # Notify the user of success
        messagebox.showinfo("Success", f"Inverted mask saved to: {output_h5_path}")
    except Exception as e:
        # Handle errors
        messagebox.showerror("Error", f"An error occurred: {e}")

def browse_and_convert_inverted():
    """
    Opens a file dialog to browse for a PNG file and converts it to an inverted HDF5 mask.
    """
    # Open file dialog to select PNG
    png_path = filedialog.askopenfilename(
        title="Select a PNG Mask File",
        filetypes=[("PNG Files", "*.png")]
    )
    if png_path:
        create_h5_mask_from_png_inverted(png_path)

if __name__ == "__main__":
    # Create the main GUI window
    root = tk.Tk()
    root.title("PNG to Inverted HDF5 Mask Converter")

    # Configure the window size
    root.geometry("400x200")

    # Add a label with instructions
    label = tk.Label(root, text="Select a PNG file to create an inverted HDF5 mask", pady=20)
    label.pack()

    # Add a "Browse" button
    browse_button = tk.Button(
        root,
        text="Browse and Convert (Inverted)",
        command=browse_and_convert_inverted,
        padx=20,
        pady=10
    )
    browse_button.pack()

    # Start the GUI event loop
    root.mainloop()
