# toggle_smoothing.py

import tkinter as tk

def toggle_smoothing(self):
    """Enable or disable the smoothing slider based on the checkbox state."""
    if self.smooth_var.get():
        self.smooth_slider.config(state=tk.NORMAL)
    else:
        self.smooth_slider.config(state=tk.DISABLED)
