import h5py
import numpy as np
import os
from tqdm import tqdm

# Function to merge multiple HDF5 files and maintain the same structure and layout, preserving dataset properties like chunking and data type, without logging
def merge_hdf5_files(input_files, output_file):
    try:
        with h5py.File(output_file, 'w') as h5out:
            for file_index, input_file in enumerate(tqdm(input_files, desc='Merging HDF5 files')):
                try:
                    with h5py.File(input_file, 'r') as h5in:
                        # Access the 'entry/data' group in each input file
                        input_group = h5in['entry/data']

                        if file_index == 0:
                            # For the first file, create the structure in the output file
                            output_group = h5out.create_group('entry/data')
                            for dataset_name in tqdm(input_group, desc=f'Processing datasets in {input_file}'):
                                data = input_group[dataset_name]
                                # Create datasets in the output file with appropriate shape and data type
                                maxshape = (None,) + data.shape[1:] if len(data.shape) > 1 else (None,)
                                chunks = (1000, 1024, 1024) if dataset_name == 'images' else None
                                output_group.create_dataset(
                                    dataset_name,
                                    data=data,
                                    maxshape=maxshape,
                                    chunks=chunks,
                                    dtype=data.dtype,
                                    compression=data.compression,
                                    fillvalue=data.fillvalue
                                )
                        else:
                            # For subsequent files, append the data to the existing datasets
                            output_group = h5out['entry/data']
                            for dataset_name in tqdm(input_group, desc=f'Appending datasets from {input_file}'):
                                data = input_group[dataset_name]
                                output_dataset = output_group[dataset_name]
                                output_dataset.resize(output_dataset.shape[0] + data.shape[0], axis=0)
                                output_dataset[-data.shape[0]:] = data[:]

                except Exception as e:
                    print(f"Error processing file {input_file}: {e}")
    except Exception as e:
        print(f"Error creating output file {output_file}: {e}")

if __name__ == "__main__":
    # Example usage
    input_folder = "/home/buster/UOX12"
    output_file = os.path.join(input_folder, "merged_output.h5")

    # Get a list of all HDF5 files in the input folder
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.h5')]

    # Merge the HDF5 files
    merge_hdf5_files(input_files, output_file)
