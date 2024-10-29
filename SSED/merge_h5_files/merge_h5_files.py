import h5py
import numpy as np
import os
import gc
from tqdm import tqdm  # Import progress bar

def merge_h5_files(input_files, output_file):
    """
    Merge multiple HDF5 files with similar structures into one.
    """
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Overwriting the file.")
        os.remove(output_file)

    try:
        with h5py.File(output_file, 'w') as h5out:
            # Create 'entry/data' group in the output file
            h5out.create_group('entry/data')
            data_group_out = h5out['entry/data']

            # Initialize datasets for concatenation
            datasets = {}
            

            # Iterate over all input files with progress bar
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
                    for i in tqdm(range(0, data_group_in['images'].shape[0], 1000), desc=f"Processing images from {file_path}"):
                        if 'images' not in data_group_out:
                            chunks = data_group_in['images'].chunks if data_group_in['images'].chunks else (1000,) + data_group_in['images'].shape[1:]
                            compression = data_group_in['images'].compression
                            compression_opts = data_group_in['images'].compression_opts
                            images_out = data_group_out.create_dataset(
                                'images',
                                shape=(0,) + data_group_in['images'].shape[1:],
                                maxshape=(None,) + data_group_in['images'].shape[1:],
                                dtype='float32',
                                chunks=chunks,
                                compression=compression,
                                compression_opts=compression_opts
                            )
                        images_chunk = data_group_in['images'][i:i + 1000]
                        images_out.resize(images_out.shape[0] + images_chunk.shape[0], axis=0)
                        images_out[-images_chunk.shape[0]:] = images_chunk
                        gc.collect()

                # Run garbage collection to free up memory
                gc.collect()

            # Write merged datasets (excluding 'images') to output file
            for name, data_parts in datasets.items():
                merged_data = np.concatenate(data_parts)
                data_group_out.create_dataset(name, data=merged_data, dtype=merged_data.dtype, maxshape=(None,) + merged_data.shape[1:])
                # Copy attributes from the first dataset
                if isinstance(data_parts[0], h5py.Dataset):
                    for attr_name, attr_value in data_parts[0].attrs.items():
                      data_group_out[name].attrs[attr_name] = attr_value

            # Merge and write 'images' dataset
            

            

            # Copy attributes from the original images dataset to the new one
            with h5py.File(input_files[0], 'r') as h5in:
                for attr_name, attr_value in h5in['entry/data/images'].attrs.items():
                  images_out.attrs[attr_name] = attr_value

    except Exception as e:
        print(f"Error merging files: {e}")

if __name__ == "__main__":
    input_files = [
        "/home/buster/leidata/hMTH1_TH287_Serial/0deg/0deg_merge/hMTH1_TH287_Serial_spot2_300nm_CL1350_015speed_0degree_20241019_1924.h5",  # Example input file path 1
        "/home/buster/leidata/hMTH1_TH287_Serial/0deg/0deg_merge/hMTH1_TH287_Serial_spot2_300nm_CL1350_015speed_0degree_20241019_1945.h5",  # Example input file path 2
        "/home/buster/leidata/hMTH1_TH287_Serial/0deg/0deg_merge/hMTH1_TH287_Serial_spot2_300nm_CL1350_015speed_0degree_20241019_2213.h5"   # Example input file path 3
    ]
    output_file = "/home/buster/leidata/hMTH1_TH287_Serial/0deg/0deg_merge/hMTH1_TH287_Serial_merged.h5"  # Output file path
    merge_h5_files(input_files, output_file)
