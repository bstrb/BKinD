import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters for peakfinding
intensity_threshold = 50
min_peak_distance = 10
local_bg_radius = 5  # Background radius can be adjusted for sensitivity

def load_image_from_h5(h5_file_path, image_index=0):
    with h5py.File(h5_file_path, 'r') as f:
        images = f['/entry/data/images']
        image = images[image_index]  # Select the image at the specified index
    return image

def load_mask(mask_path):
    with h5py.File(mask_path, 'r') as f:
        mask = f['/mask'][:]  # Load the mask from the /mask dataset
    return mask

def apply_mask(image, mask, mask_good=0x01):
    """
    Apply a mask to an image. Only include areas where mask equals mask_good.
    Args:
        image (np.array): 2D numpy array representing the image.
        mask (np.array): 2D numpy array of the same shape as the image.
        mask_good (int): Value in the mask that indicates valid pixels.
    Returns:
        np.array: Masked image with invalid areas set to zero.
    """
    return np.where(mask == mask_good, image, 0)

def detect_peaks_in_image(image, intensity_threshold, min_peak_distance, local_bg_radius):
    background_subtracted = np.copy(image)
    for row_idx in range(image.shape[0]):
        for col_idx in range(image.shape[1]):
            row_start = max(0, row_idx - local_bg_radius)
            row_end = min(image.shape[0], row_idx + local_bg_radius + 1)
            col_start = max(0, col_idx - local_bg_radius)
            col_end = min(image.shape[1], col_idx + local_bg_radius + 1)
            local_bg = np.mean(image[row_start:row_end, col_start:col_end])
            background_subtracted[row_idx, col_idx] -= local_bg

    peak_coords = []
    for row_idx, row in enumerate(background_subtracted):
        peaks, _ = find_peaks(row, height=intensity_threshold, distance=min_peak_distance)
        for peak in peaks:
            peak_coords.append((row_idx, peak))

    return peak_coords

def display_image_with_peaks(image, peak_coords):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', origin='lower')
    if peak_coords:
        peak_rows, peak_cols = zip(*peak_coords)
        plt.plot(peak_cols, peak_rows, 'r.', markersize=5, label='Detected Peaks')
    plt.title("Image with Detected Peaks")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend()
    plt.show()

# Main script
h5_file_path = "/home/buster/UOX123/deiced_UOX_merged.h5"
mask_path = "/home/buster/mask/pxmask.h5"
image_index = 18581

# Load the specified image and mask
image = load_image_from_h5(h5_file_path, image_index=image_index)
mask = load_mask(mask_path)

# Apply the mask to the image using the geometry file's specifications
masked_image = apply_mask(image, mask, mask_good=0x01)

# Detect peaks in the masked image
peak_coords = detect_peaks_in_image(masked_image, intensity_threshold, min_peak_distance, local_bg_radius)

# Display the masked image with detected peaks
display_image_with_peaks(masked_image, peak_coords)
