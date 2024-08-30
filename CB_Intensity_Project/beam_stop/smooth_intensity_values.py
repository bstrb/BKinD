# smooth_intensity_values.py

import numpy as np

def smooth_intensity_values(self, intensity_values, window_size):
    """Smooth the intensity values using a moving average."""
    smoothed_values = np.convolve(intensity_values, np.ones(window_size) / window_size, mode='valid')
    
    # To maintain the same length as the original array, we'll pad the start of the smoothed array
    pad_size = len(intensity_values) - len(smoothed_values)
    return np.pad(smoothed_values, (pad_size // 2, pad_size - pad_size // 2), mode='edge')
