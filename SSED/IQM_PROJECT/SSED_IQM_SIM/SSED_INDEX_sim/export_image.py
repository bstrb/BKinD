#!/usr/bin/env python3

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 3:
        print("Usage: ./export_image.py <input_file.h5> <image_index>")
        sys.exit(1)

    input_file = sys.argv[1]
    image_index = int(sys.argv[2])

    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    # Open the HDF5 file
    with h5py.File(input_file, "r") as h5_file:
        dataset_path = "/entry/data/images"  # Update if necessary
        if dataset_path not in h5_file:
            print(f"Error: Dataset '{dataset_path}' not found in the file.")
            sys.exit(1)
        
        images_dataset = h5_file[dataset_path]
        if image_index < 0 or image_index >= images_dataset.shape[0]:
            print(f"Error: Index {image_index} out of bounds. Dataset contains {images_dataset.shape[0]} images.")
            sys.exit(1)
        
        image = images_dataset[image_index]

    # Define output file path
    output_file = os.path.join(os.path.dirname(input_file), f"exported_image_{image_index}.png")

    # Save the image as a .png file
    plt.imsave(output_file, image, cmap="gray")

    print(f"Image saved to: {output_file}")

if __name__ == "__main__":
    main()
