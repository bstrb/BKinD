import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_h5_dataset(file_path: str, dataset_path: str) -> np.ndarray:
    """
    Loads a dataset from an HDF5 file.

    Parameters:
        file_path (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the HDF5 file.

    Returns:
        np.ndarray: The loaded dataset as a NumPy array.
    """
    try:
        with h5py.File(file_path, 'r') as h5_file:
            if dataset_path not in h5_file:
                raise KeyError(f"Dataset '{dataset_path}' not found in '{file_path}'.")
            data = h5_file[dataset_path][()]
            logging.info(f"Loaded dataset '{dataset_path}' from '{file_path}'.")
    except Exception as e:
        logging.error(f"Error loading dataset '{dataset_path}' from '{file_path}': {e}")
        raise
    return data

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a mask to an image. Masked areas are set to NaN.

    Parameters:
        image (np.ndarray): The input image.
        mask (np.ndarray): The mask array (True for valid pixels, False for masked pixels).

    Returns:
        np.ndarray: The masked image.
    """
    if image.shape != mask.shape:
        error_msg = f"Image shape {image.shape} and mask shape {mask.shape} do not match."
        logging.error(error_msg)
        raise ValueError(error_msg)
    masked_image = np.where(mask, image, np.nan)
    logging.info("Applied mask to image.")
    return masked_image

def compute_radial_median_intensity(image: np.ndarray, center: Tuple[float, float], num_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the median intensity as a function of radial distance from a center point.

    Parameters:
        image (np.ndarray): The 2D image array.
        center (tuple): (x, y) coordinates of the center.
        num_bins (int): Number of radial bins.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Radial distances (bin centers) and corresponding median intensities.
    """
    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.flatten()
    intensity = image.flatten()

    # Remove NaN values
    valid = ~np.isnan(intensity)
    r = r[valid]
    intensity = intensity[valid]

    # Define radial bins
    r_max = r.max()
    bins = np.linspace(0, r_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute median intensity for each bin
    median_intensity = np.zeros(num_bins)
    for i in range(num_bins):
        bin_mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(bin_mask):
            median_intensity[i] = np.median(intensity[bin_mask])
        else:
            median_intensity[i] = np.nan  # Handle empty bins

    logging.info("Computed radial median intensity.")
    return bin_centers, median_intensity

def plot_radial_median_intensity(radii: np.ndarray, median_intensity: np.ndarray, image_index: int):
    """
    Plots the radial median intensity.

    Parameters:
        radii (np.ndarray): Radial distances (bin centers).
        median_intensity (np.ndarray): Median intensities corresponding to the radial bins.
        image_index (int): Index of the processed image.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(radii, median_intensity, marker='o', linestyle='-')
    plt.title(f'Radial Median Intensity for Image {image_index}')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Median Intensity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logging.info(f"Plotted radial median intensity for image {image_index}.")

def process_image(h5_file_path: str, images_dataset_path: str, mask_file_path: str,
                 mask_dataset_path: str, image_index: int, center: Optional[Tuple[float, float]] = None):
    """
    Processes a single image: applies mask, computes radial median intensity, and plots it.

    Parameters:
        h5_file_path (str): Path to the HDF5 file containing images.
        images_dataset_path (str): Dataset path within the HDF5 file for images.
        mask_file_path (str): Path to the mask HDF5 file.
        mask_dataset_path (str): Dataset path within the mask HDF5 file.
        image_index (int): Index of the image to process.
        center (tuple, optional): (x, y) coordinates for radial computations. Defaults to image center.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Radial distances and median intensities.
    """
    # Load mask
    mask = load_h5_dataset(mask_file_path, mask_dataset_path)

    # Load image
    images = load_h5_dataset(h5_file_path, images_dataset_path)
    if image_index >= images.shape[0] or image_index < 0:
        error_msg = f"Image index {image_index} is out of bounds. Total images: {images.shape[0]}."
        logging.error(error_msg)
        raise IndexError(error_msg)
    image = images[image_index, :, :]
    logging.info(f"Processing Image Index: {image_index}")

    # Apply mask
    masked_image = apply_mask(image, mask)

    # Determine center
    if center is None:
        # Default to image center
        center = (image.shape[1] / 2, image.shape[0] / 2)
        logging.info(f"No center provided. Using image center: {center}")
    else:
        logging.info(f"Using provided center: {center}")

    # Compute radial median intensity
    radii, median_intensity = compute_radial_median_intensity(masked_image, center)

    # Plot radial median intensity
    plot_radial_median_intensity(radii, median_intensity, image_index)

    return radii, median_intensity

def process_multiple_images(h5_file_path: str, images_dataset_path: str, mask_file_path: str,
                           mask_dataset_path: str, selected_indices: List[int], center: Optional[Tuple[float, float]] = None):
    """
    Processes multiple images: applies mask, computes radial median intensity, and plots them.

    Parameters:
        h5_file_path (str): Path to the HDF5 file containing images.
        images_dataset_path (str): Dataset path within the HDF5 file for images.
        mask_file_path (str): Path to the mask HDF5 file.
        mask_dataset_path (str): Dataset path within the mask HDF5 file.
        selected_indices (List[int]): List of image indices to process.
        center (tuple, optional): (x, y) coordinates for radial computations. Defaults to image center.

    Returns:
        None
    """
    for image_index in selected_indices:
        try:
            radii, median_intensity = process_image(
                h5_file_path=h5_file_path,
                images_dataset_path=images_dataset_path,
                mask_file_path=mask_file_path,
                mask_dataset_path=mask_dataset_path,
                image_index=image_index,
                center=center
            )
        except Exception as e:
            logging.error(f"Failed to process image {image_index}: {e}")

def main():
    """
    Main function to process selected images and plot radial median intensity.
    Modify the paths and selected_indices as per your data.
    """
    # Example Inputs (Modify these paths and indices as needed)
    h5_file_path = '/home/buster/UOX1/UOX1_min_10/UOX1_min_10.h5'
    images_dataset_path = '/entry/data/images'  # Modify if different
    mask_file_path = '/home/buster/mask/pxmask.h5'
    mask_dataset_path = '/mask'  # Modify if different
    selected_indices = [1300]  # Replace with your selected indices

    # Optional: Define a custom center (x, y). If None, image center is used.
    custom_center = None  # e.g., (512, 512) (or image center depending on size)

    # Process the selected images
    process_multiple_images(
        h5_file_path=h5_file_path,
        images_dataset_path=images_dataset_path,
        mask_file_path=mask_file_path,
        mask_dataset_path=mask_dataset_path,
        selected_indices=selected_indices,
        center=custom_center
    )

# Uncomment the following line to run the main function directly
# main()
