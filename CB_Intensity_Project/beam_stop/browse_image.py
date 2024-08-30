# browse_image.py

import tkinter as tk
from tkinter import filedialog

def browse_image(self):
    self.file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.img *.tif *.tiff *.cbf")]
    )
    if self.file_path:
        self.file_name_label.config(text=self.file_path)
        self.display_image_button.config(state=tk.NORMAL)
        self.calculate_button.config(state=tk.DISABLED)
