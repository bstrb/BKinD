import h5py
import numpy as np
from PIL import Image
import sys
import os

def extract_and_save_image(h5_file_path, output_png_path, image_index=0):
    """
    Extracts a single image from an HDF5 file and saves it as a PNG.

    Parameters:
    - h5_file_path (str): Path to the input HDF5 file.
    - output_png_path (str): Path where the PNG image will be saved.
    - image_index (int): Index of the image to extract (default is 0).

    Raises:
    - FileNotFoundError: If the HDF5 file does not exist.
    - KeyError: If the specified dataset path does not exist in the HDF5 file.
    - IndexError: If the image_index is out of bounds.
    - ValueError: If the image data cannot be converted to a valid image format.
    """
    # Check if the HDF5 file exists
    if not os.path.isfile(h5_file_path):
        raise FileNotFoundError(f"The file '{h5_file_path}' does not exist.")

    # Open the HDF5 file in read mode
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Navigate to the dataset containing images
        dataset_path = 'entry/data/images'
        if dataset_path not in h5_file:
            raise KeyError(f"The dataset path '{dataset_path}' does not exist in the HDF5 file.")

        images_dataset = h5_file[dataset_path]

        # Check if the dataset is empty
        if images_dataset.size == 0:
            raise ValueError(f"The dataset '{dataset_path}' is empty.")

        # Check if the image_index is within bounds
        if image_index < 0 or image_index >= images_dataset.shape[0]:
            raise IndexError(f"Image index {image_index} is out of bounds for dataset with {images_dataset.shape[0]} images.")

        # Extract the specified image
        image_data = images_dataset[image_index]

        # Handle different image data shapes
        # Common shapes:
        # - (height, width) for grayscale
        # - (height, width, channels) for RGB/RGBA
        if image_data.ndim == 2:
            mode = 'L'  # Grayscale
        elif image_data.ndim == 3:
            if image_data.shape[2] == 3:
                mode = 'RGB'
            elif image_data.shape[2] == 4:
                mode = 'RGBA'
            else:
                raise ValueError(f"Unsupported number of channels: {image_data.shape[2]}")
        else:
            raise ValueError(f"Unsupported image data shape: {image_data.shape}")

        # Convert the image data to uint8 if it's not already
        if image_data.dtype != np.uint8:
            # Normalize the data to the range 0-255
            image_min = image_data.min()
            image_max = image_data.max()
            if image_max == image_min:
                # Avoid division by zero; set to mid-gray
                image_data = np.full_like(image_data, 128, dtype=np.uint8)
            else:
                image_data = 255 * (image_data - image_min) / (image_max - image_min)
                image_data = image_data.astype(np.uint8)

        # Create a PIL Image from the numpy array
        try:
            image = Image.fromarray(image_data, mode)
        except Exception as e:
            raise ValueError(f"Failed to convert image data to PIL Image: {e}")

        # Save the image as PNG
        image.save(output_png_path)
        print(f"Image {image_index} saved as '{output_png_path}'.")

if __name__ == "__main__":
    import argparse

    # Default HDF5 file path
    DEFAULT_H5_PATH = "/Users/xiaodong/Desktop/UOX1/deiced_UOX1_min_15_peak_backup.h5"

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract and save a single image from an HDF5 file as a PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--h5_file',
        type=str,
        default=DEFAULT_H5_PATH,
        help=f"Path to the input HDF5 (.h5) file. Default: {DEFAULT_H5_PATH}"
    )
    parser.add_argument(
        '--index',
        type=int,
        default=1010,
        help="Index of the image to extract (default: 0)."
    )

    args = parser.parse_args()

    # Determine the output directory and filename based on input HDF5 file and image index
    input_h5_path = args.h5_file
    image_index = args.index

    # Get the directory of the input HDF5 file
    input_dir = os.path.dirname(os.path.abspath(input_h5_path))

    # Define the output PNG filename based on image index
    output_png_filename = f"image{image_index}.png"

    # Full path for the output PNG
    output_png_path = os.path.join(input_dir, output_png_filename)

    try:
        extract_and_save_image(input_h5_path, output_png_path, image_index)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
