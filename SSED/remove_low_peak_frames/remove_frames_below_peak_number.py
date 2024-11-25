# remove_frames_below_peak_number.py

import os
import h5py
import numpy as np
from tqdm import tqdm

def remove_frames_below_peak_number(input_file, tresh=1):
    # Define the output file path
    input_file_dir = os.path.dirname(input_file)
    output_file = os.path.join(input_file_dir, f"min_{tresh}_peak.h5")

    # Ensure the output file does not already exist, or create it
    if not os.path.exists(output_file):
        with h5py.File(output_file, 'w') as out_file:
            pass

    with h5py.File(input_file, 'r') as in_file, h5py.File(output_file, 'w') as out_file:
        # Create the overall structure in the output file
        out_entry = out_file.create_group('entry')
        out_data_group = out_entry.create_group('data')

        # Read the original data
        in_data_group = in_file['entry/data']

        # Get the indices of frames with peaks
        nPeaks = in_data_group['nPeaks'][:]
        valid_indices = np.where(nPeaks >= tresh)[0]
        num_valid_frames = len(valid_indices)

        # Create the 'images' dataset with only frames that have peaks
        images_dataset = in_data_group['images']
        image_shape = (num_valid_frames,) + images_dataset.shape[1:]
        
        # Create the new images dataset in the output file with chunk size of 1000x1024x1024
        out_images_dataset = out_data_group.create_dataset(
            'images', shape=image_shape, maxshape=image_shape, dtype=images_dataset.dtype, chunks=(1000, 1024, 1024)
        )

        # Copy only valid frames into the new dataset
        for idx, valid_idx in enumerate(tqdm(valid_indices, desc=f"Removing frames with less than {tresh} peaks from images")):
            out_images_dataset[idx, ...] = images_dataset[valid_idx, ...]

        # Copy other datasets from input to output
        for dataset_name in in_data_group.keys():
            if dataset_name != 'images':  # We've already handled 'images'
                dataset = in_data_group[dataset_name]
                filtered_data = dataset[valid_indices] if len(dataset.shape) == 1 else dataset[valid_indices, ...]
                out_data_group.create_dataset(dataset_name, data=filtered_data, chunks=True)

    print(f"Filtered HDF5 file created: {output_file}")

# # Path to the input HDF5 file
# input_file = '/home/buster/UOX1/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5'

# # Threshold for minimum number of peaks
# threshold = 15

# # Call the function to remove frames with less than the threshold number of peaks
# remove_frames_below_peak_number(input_file, tresh=threshold)
