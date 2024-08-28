# toggle_smoothing.py

import tkinter as tk

def toggle_smoothing(self):
    """Enable or disable the smoothing slider based on the checkbox state."""
    self.smooth_slider.config(state=tk.NORMAL if self.smooth_var.get() else tk.DISABLED)