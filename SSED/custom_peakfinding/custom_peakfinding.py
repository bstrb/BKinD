import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters for peakfinding
intensity_threshold = 100
min_peak_distance = 5
local_bg_radius = 5  # Background radius can be adjusted for sensitivity

def load_image_from_h5(h5_file_path, image_index=0):
    """
    Load a specific image from an HDF5 file at a given index.
    Args:
        h5_file_path (str): Path to the HDF5 file.
        image_index (int): Index of the image to load.
    Returns:
        np.array: The selected image as a 2D numpy array.
    """
    with h5py.File(h5_file_path, 'r') as f:
        images = f['/entry/data/images']
        image = images[image_index]  # Select the image at the specified index
    return image

def detect_peaks_in_image(image, intensity_threshold, min_peak_distance, local_bg_radius):
    """
    Detect peaks in a 2D image array using a simple peakfinding approach.
    Args:
        image (np.array): 2D numpy array representing the image.
        intensity_threshold (int): Minimum intensity to consider a peak.
        min_peak_distance (int): Minimum distance between peaks.
        local_bg_radius (int): Radius for local background subtraction.
    Returns:
        list of (x, y): Coordinates of detected peaks.
    """
    # Background subtraction for the entire image
    background_subtracted = np.copy(image)
    for row_idx in range(image.shape[0]):
        for col_idx in range(image.shape[1]):
            # Define local background region
            row_start = max(0, row_idx - local_bg_radius)
            row_end = min(image.shape[0], row_idx + local_bg_radius + 1)
            col_start = max(0, col_idx - local_bg_radius)
            col_end = min(image.shape[1], col_idx + local_bg_radius + 1)
            local_bg = np.mean(image[row_start:row_end, col_start:col_end])
            background_subtracted[row_idx, col_idx] -= local_bg

    # Find peaks along each row
    peak_coords = []
    for row_idx, row in enumerate(background_subtracted):
        peaks, _ = find_peaks(row, height=intensity_threshold, distance=min_peak_distance)
        for peak in peaks:
            peak_coords.append((row_idx, peak))

    return peak_coords

def display_image_with_peaks(image, peak_coords):
    """
    Display an image with detected peaks overlaid.
    Args:
        image (np.array): 2D numpy array representing the image.
        peak_coords (list of tuples): List of (row, column) coordinates of detected peaks.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', origin='lower')
    peak_rows, peak_cols = zip(*peak_coords)  # Separate rows and columns
    plt.plot(peak_cols, peak_rows, 'r.', markersize=5, label='Detected Peaks')
    plt.title("Image with Detected Peaks")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend()
    plt.show()

# Main script
h5_file_path = "/home/buster/UOX123/deiced_UOX_merged.h5"  # Update with the path to your HDF5 file
image_index = 18581  # Index of the image to process

# Load the specified image from the HDF5 file
image = load_image_from_h5(h5_file_path, image_index=image_index)

# Detect peaks in the selected image
peak_coords = detect_peaks_in_image(image, intensity_threshold, min_peak_distance, local_bg_radius)

# Display the image with detected peaks
display_image_with_peaks(image, peak_coords)
