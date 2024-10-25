import h5py
import numpy as np
import os
from tqdm import tqdm

def merge_h5_files(input_files, output_file):
    """
    Merge multiple HDF5 files with similar structures into one.
    """
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Please remove it or use a different name.")
        return

    try:
        with h5py.File(output_file, 'w') as h5out:
            # Create 'entry/data' group in the output file
            h5out.create_group('entry/data')
            data_group_out = h5out['entry/data']

            # Initialize datasets for concatenation
            datasets = {}
            images_list = []

            # Iterate over all input files
            for file_path in input_files:
                with h5py.File(file_path, 'r') as h5in:
                    # Check if 'entry/data' group exists in the input file
                    if 'entry/data' not in h5in:
                        print(f"'entry/data' group not found in input file: {file_path}")
                        return

                    data_group_in = h5in['entry/data']

                    # Collect data from each dataset except 'images'
                    for name, dataset in data_group_in.items():
                        if name != 'images':
                            if name not in datasets:
                                datasets[name] = []
                            datasets[name].append(dataset[:])

                    # Collect images dataset
                    images_list.append(data_group_in['images'][:])

            # Write merged datasets (excluding 'images') to output file
            for name, data_parts in datasets.items():
                merged_data = np.concatenate(data_parts)
                data_group_out.create_dataset(name, data=merged_data, dtype=merged_data.dtype, maxshape=(None,) + merged_data.shape[1:])
                # Copy attributes from the first dataset
                for attr_name, attr_value in data_parts[0].attrs.items():
                    data_group_out[name].attrs[attr_name] = attr_value

            # Merge and write 'images' dataset
            merged_images = np.concatenate(images_list)
            chunks = images_list[0].chunks if images_list[0].chunks else (1000,) + images_list[0].shape[1:]
            compression = images_list[0].compression
            compression_opts = images_list[0].compression_opts

            images_out = data_group_out.create_dataset(
                'images',
                data=merged_images,
                maxshape=(None,) + merged_images.shape[1:],
                dtype='float32',
                chunks=chunks,
                compression=compression,
                compression_opts=compression_opts
            )

            # Copy attributes from the original images dataset to the new one
            for attr_name, attr_value in h5py.File(input_files[0], 'r')['entry/data/images'].attrs.items():
                images_out.attrs[attr_name] = attr_value

    except Exception as e:
        print(f"Error merging files: {e}")

if __name__ == "__main__":
    input_files = [
        "/home/buster/UOX123/UOX1_minpeaks_15.h5",  # Example input file path 1
        "/home/buster/UOX123/UOX2_minpeaks_15.h5",  # Example input file path 2
        "/home/buster/UOX123/UOX3_minpeaks_15.h5"   # Example input file path 3
    ]
    output_file = "/home/buster/UOX123/UOX_merged.h5"  # Output file path
    merge_h5_files(input_files, output_file)
