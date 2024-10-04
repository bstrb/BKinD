# remove_frames_with_no_peaks_v2.py

import h5py
import numpy as np
from tqdm import tqdm

def remove_frames_with_no_peaks(input_file, output_file):
    with h5py.File(input_file, 'r') as in_file, h5py.File(output_file, 'w') as out_file:
        # Create the overall structure in the output file
        out_entry = out_file.create_group('entry')
        out_data_group = out_entry.create_group('data')

        # Read the original data
        in_data_group = in_file['entry/data']

        # Get the indices of frames with peaks
        nPeaks = in_data_group['nPeaks'][:]
        valid_indices = np.where(nPeaks > 0)[0]
        num_valid_frames = len(valid_indices)

        # Create the 'images' dataset with only frames that have peaks
        images_dataset = in_data_group['images']
        image_shape = (num_valid_frames,) + images_dataset.shape[1:]
        
        # Create the new images dataset in the output file
        out_images_dataset = out_data_group.create_dataset(
            'images', shape=image_shape, maxshape=image_shape, dtype=images_dataset.dtype, chunks=True
        )

        # Copy only valid frames into the new dataset
        for idx, valid_idx in enumerate(tqdm(valid_indices, desc="Removing frames without peaks from images")):
            out_images_dataset[idx, ...] = images_dataset[valid_idx, ...]

        # Copy other datasets from input to output
        for dataset_name in in_data_group.keys():
            if dataset_name != 'images':  # We've already handled 'images'
                dataset = in_data_group[dataset_name]
                filtered_data = dataset[valid_indices] if len(dataset.shape) == 1 else dataset[valid_indices, ...]
                out_data_group.create_dataset(dataset_name, data=filtered_data, chunks=True)

    print(f"Filtered HDF5 file created: {output_file}")

# Usage example
input_file = '/home/buster/UOX1/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5'
output_file = '/home/buster/UOX1/UOX_filtered.h5'
remove_frames_with_no_peaks(input_file, output_file)
