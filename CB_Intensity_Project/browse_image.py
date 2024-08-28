# browse_image.py

import os

import tkinter as tk
from tkinter import filedialog, messagebox

def browse_image(self):
    # Open file dialog to select an .img file
    self.file_path = filedialog.askopenfilename(filetypes=[("IMG files", "*.img")], title="Select an .img file")
    
    if not self.file_path:
        messagebox.showwarning("No file selected", "Please select an .img file.")
        return

    # Update the label with the selected file name
    file_name = os.path.basename(self.file_path)
    self.file_name_label.config(text=f"Selected File: {file_name}")

    # Enable the display image button after an image is selected
    self.display_image_button.config(state=tk.NORMAL)
