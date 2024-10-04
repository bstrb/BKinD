# remove_frames_with_no_peaks.py

import h5py
import numpy as np
from tqdm import tqdm

def remove_frames_with_no_peaks(input_file, output_file, chunk_size=1000):
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

        # Create datasets in the output file, except for images
        for dataset_name in tqdm(in_data_group.keys(), desc="Initializing datasets"):
            dataset = in_data_group[dataset_name]
            if dataset_name == 'images':
                # Calculate a chunk size to stay within the HDF5 limits
                max_elements = (4 * 1024 * 1024 * 1024) // 2  # 4GB divided by 2 bytes per element
                max_frames_per_chunk = max_elements // (dataset.shape[1] * dataset.shape[2])
                adjusted_chunk_size = min(chunk_size, max_frames_per_chunk)

                # Create dataset for images with adjusted chunking
                shape = (num_valid_frames,) + dataset.shape[1:]
                out_data_group.create_dataset(dataset_name, shape=shape, maxshape=shape, dtype=dataset.dtype,
                                              chunks=(adjusted_chunk_size,) + dataset.shape[1:], compression="gzip")
            else:
                # Create other datasets without chunking
                filtered_data = dataset[valid_indices] if len(dataset.shape) == 1 else dataset[valid_indices, ...]
                out_data_group.create_dataset(dataset_name, data=filtered_data, chunks=True)

        # Process the images dataset in chunks sequentially
        images_dataset = in_data_group['images']
        out_images_dataset = out_data_group['images']

        for start in tqdm(range(0, num_valid_frames, chunk_size), desc="Processing images sequentially"):
            end = min(start + chunk_size, num_valid_frames)
            chunk_indices = valid_indices[start:end]
            out_images_dataset[start:end, ...] = images_dataset[chunk_indices, ...]

    print(f"Filtered HDF5 file created: {output_file}")


# Usage example
input_file = '/home/buster/UOX1/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5'
output_file = '/home/buster/UOX1/UOX_filtered.h5'
remove_frames_with_no_peaks(input_file, output_file)