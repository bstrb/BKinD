import h5py
from PIL import Image
import numpy as np

def save_h5_image_as_tiff(h5_file_path, dataset_path, index, output_tiff_path):
    """
    Extracts an image from an HDF5 file at a given index and saves it as a TIFF file.

    Parameters:
    - h5_file_path: Path to the HDF5 file.
    - dataset_path: Path to the dataset inside the HDF5 file.
    - index: Index of the image to extract.
    - output_tiff_path: Path to save the output TIFF file.
    """
    try:
        # Open the HDF5 file
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Access the dataset
            dataset = h5_file[dataset_path]

            # Validate index
            if index < 0 or index >= dataset.shape[0]:
                raise IndexError("Index out of range for the dataset.")

            # Extract the image at the given index
            image_data = dataset[index]

            # Normalize data if it's not in an 8-bit range
            if image_data.dtype != np.uint8:
                image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
                image_data = image_data.astype(np.uint8)

            # Save as TIFF using PIL
            img = Image.fromarray(image_data)
            img.save(output_tiff_path)

        print(f"Image saved successfully to {output_tiff_path}")

    except Exception as e:
        print(f"Error: {e}")

# File details
h5_file_path = "/home/buster/UOX1/UOX1_original/UOX1.h5"  # Path to your HDF5 file
dataset_path = "/entry/data/images"  # Path to your dataset inside the HDF5 file
index = 20604  # Index of the image to extract
output_tiff_path = "/home/buster/UOX1/UOX1_original/output_image_20604.tiff"  # Path to save the output TIFF

save_h5_image_as_tiff(h5_file_path, dataset_path, index, output_tiff_path)
