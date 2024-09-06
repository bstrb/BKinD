# # browse_image.py

# import os

# import tkinter as tk
# from tkinter import filedialog, messagebox

# def browse_image(self):
#     # Open file dialog to select an .img file

#     self.file_path = filedialog.askopenfilename(filetypes=[("IMG files", "*.img")], title="Select an .img file")
    
#     if not self.file_path:
#         messagebox.showwarning("No file selected", "Please select an .img file.")
#         return

#     # Update the label with the selected file name
#     file_name = os.path.basename(self.file_path)
#     self.file_name_label.config(text=f"Selected File: {file_name}")

#     # Enable the display image button after an image is selected
#     self.display_gaussian_button.config(state=tk.NORMAL)

#     # Enable the display image button after an image is selected
#     self.display_voigt_button.config(state=tk.NORMAL)

import os
import tkinter as tk
from tkinter import filedialog, messagebox

def browse_image(self):
    # Default file path for trials
    default_file_path = "/mnt/c/Users/bubl3932/Desktop/3DED-DATA/LTA/LTA1/images/00014.img"
    
    # Open file dialog to select an .img file
    self.file_path = filedialog.askopenfilename(filetypes=[("IMG files", "*.img")], title="Select an .img file")
    
    # If no file is selected, use the default file path
    if not self.file_path:
        messagebox.showinfo("Using Default File", f"No file selected. Using default file:\n{default_file_path}")
        self.file_path = default_file_path

    # Update the label with the selected or default file name
    file_name = os.path.basename(self.file_path)
    self.file_name_label.config(text=f"Selected File: {file_name}")

    # Enable the display image button after an image is selected
    self.display_gaussian_button.config(state=tk.NORMAL)
    self.display_voigt_button.config(state=tk.NORMAL)
