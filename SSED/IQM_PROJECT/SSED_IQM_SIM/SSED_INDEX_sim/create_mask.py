#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py

def create_binary_mask(image_path, mask_color):
    """
    Create a binary mask from the marked regions in the edited image.
    """
    # Load the image using PIL
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Define the mask based on a specific color
    # For example, assuming mask color is red (255, 0, 0)
    red, green, blue = mask_color
    mask = ((image_np[:, :, 0] == red) & 
            (image_np[:, :, 1] == green) & 
            (image_np[:, :, 2] == blue))

    # Convert mask to binary (0s and 1s)
    binary_mask = mask.astype(np.uint8)

    return binary_mask

def main():
    if len(sys.argv) != 4:
        print("Usage: ./create_binary_mask.py <edited_image.png> <h5_file_path> <image_index>")
        sys.exit(1)

    edited_image_path = sys.argv[1]
    h5_file_path = sys.argv[2]
    image_index = int(sys.argv[3])

    if not os.path.isfile(edited_image_path):
        print(f"Error: File '{edited_image_path}' not found.")
        sys.exit(1)

    # Define the color of the mask (e.g., red (255, 0, 0))
    mask_color = (255, 0, 0)

    # Create the binary mask
    binary_mask = create_binary_mask(edited_image_path, mask_color)

    # Save the mask to an HDF5 file in the same folder as the original H5 file
    output_mask_file = os.path.join(os.path.dirname(h5_file_path), "mask.h5")
    with h5py.File(output_mask_file, "w") as h5_mask:
        h5_mask.create_dataset(f"mask_{image_index}", data=binary_mask, compression="gzip")

    print(f"Mask saved to: {output_mask_file}")

if __name__ == "__main__":
    main()
