# get_dir.py

# Standard library imports
import os

# Third-party imports
from tkinter import messagebox

def get_dir(self, xray):
    
    if not xray:
        xds_dir = self.xds_dir.get()
        shelx_dir = self.shelx_dir.get()
        output_dir = self.output_dir.get()

        if not (os.path.exists(xds_dir) and os.path.exists(shelx_dir) and os.path.exists(output_dir)):
            messagebox.showerror("Directory Error", "One or more directories are invalid or do not exist. Please ensure the XDS, SHELX, and Output directories are correctly specified.")
            return None, None, None
    else:
        xds_dir = None
        shelx_dir = self.shelx_dir_xray.get()
        output_dir = self.output_dir_xray.get()
        if not (os.path.exists(shelx_dir) and os.path.exists(output_dir)):
            messagebox.showerror("Directory Error", "One or more directories are invalid or do not exist. Please ensure the SHELX and Output directories are correctly specified.")
            return None, None, None

    return xds_dir, shelx_dir, output_dir
