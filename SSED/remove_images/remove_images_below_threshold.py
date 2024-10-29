import h5py
import numpy as np
import os
from tqdm import tqdm

def remove_images_below_threshold(input_file, min_peaks):
    output_file = os.path.splitext(input_file)[0] + f"_minpeaks_{min_peaks}.h5"

    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Please remove it or use a different name.")
        return

    try:
        with h5py.File(input_file, 'r') as h5in:
            # Check if 'entry/data' group exists
            if 'entry/data' not in h5in:
                print(f"Group 'entry/data' not found in file {input_file}")
                return

            # Filter indices based on min_peaks
            nPeaks = h5in['entry/data/nPeaks'][:]
            valid_indices = np.where(nPeaks > min_peaks)[0]

        # Create the output file and copy necessary datasets
        with h5py.File(input_file, 'r') as h5in, h5py.File(output_file, 'w') as h5out:
            # Create 'entry/data' group in the output file
            h5out.create_group('entry/data')

            # Copy datasets except 'images', filtering based on valid indices
            for name, dataset in h5in['entry/data'].items():
                if name != 'images':
                    data_shape = dataset.shape
                    if len(data_shape) > 0 and data_shape[0] == nPeaks.shape[0]:
                        # Copy only valid indices
                        filtered_data = dataset[valid_indices]
                        h5out['entry/data'].create_dataset(name, data=filtered_data, maxshape=(None, *data_shape[1:]), dtype=dataset.dtype, chunks=None, compression=None, compression_opts=None)
                    else:
                        # Copy entire dataset if it doesn't match the nPeaks shape
                        h5out['entry/data'].create_dataset(name, data=dataset[:], dtype=dataset.dtype, chunks=None, compression=None, compression_opts=None)
                    # Copy attributes for each dataset
                    for attr_name, attr_value in dataset.attrs.items():
                        h5out[f'entry/data/{name}'].attrs[attr_name] = attr_value

            # Create the output dataset with the same structure, except converting images to float32
            images_dset = h5in['entry/data/images']
            chunks = images_dset.chunks if images_dset.chunks else (1000, 1024, 1024)
            compression = images_dset.compression if images_dset.compression else None
            compression_opts = images_dset.compression_opts if images_dset.compression_opts else None

            new_shape = (len(valid_indices),) + images_dset.shape[1:]
            images_out = h5out['entry/data'].create_dataset(
                'images',
                shape=new_shape,
                maxshape=new_shape,
                dtype='float32',
                chunks=chunks,
                compression=compression,
                compression_opts=compression_opts
            )

            # Create a progress bar for processing images in chunks
            chunk_size = 1000
            total_chunks = (len(valid_indices) + chunk_size - 1) // chunk_size
            progress_bar = tqdm(total=total_chunks, desc='Processing images', unit='chunk')

            # Copy valid images in chunks
            for i in range(0, len(valid_indices), chunk_size):
                chunk_indices = valid_indices[i:i + chunk_size]
                images_chunk = images_dset[chunk_indices].astype('float32')
                images_out[i:i + len(chunk_indices)] = images_chunk
                progress_bar.update(1)

            progress_bar.close()

            # Copy attributes from the original images dataset to the new one
            for attr_name, attr_value in images_dset.attrs.items():
                images_out.attrs[attr_name] = attr_value

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

if __name__ == "__main__":
    # input_file = "/home/buster/leidata/hMTH1_TH287_Serial/0deg/0deg_merge/hMTH1_TH287_Serial_spot2_300nm_CL1350_015speed_0degree_20241019_2213.h5"  # Example input file path
    # input_file = "/home/buster/leidata/hMTH1_TH287_Serial/0deg/0deg_merge/hMTH1_TH287_Serial_spot2_300nm_CL1350_015speed_0degree_20241019_1924.h5"  # Example input file path
    input_file = "/home/buster/leidata/hMTH1_TH287_Serial/0deg/0deg_merge/hMTH1_TH287_Serial_merged.h5"  # Example input file path
    min_peaks = 15  # Set the threshold for minimum number of peaks
    remove_images_below_threshold(input_file, min_peaks)
