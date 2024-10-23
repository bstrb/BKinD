import h5py
import numpy as np
import os
from tqdm import tqdm

def merge_h5_files(file1, file2, output_file):
    """
    Merge two HDF5 files with similar structures into one.
    """
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Please remove it or use a different name.")
        return

    try:
        with h5py.File(file1, 'r') as h5in1, h5py.File(file2, 'r') as h5in2, h5py.File(output_file, 'w') as h5out:
            # Check if 'entry/data' group exists in both input files
            if 'entry/data' not in h5in1 or 'entry/data' not in h5in2:
                print("'entry/data' group not found in one or both input files")
                return

            # Create 'entry/data' group in the output file
            h5out.create_group('entry/data')

            data_group_in1 = h5in1['entry/data']
            data_group_in2 = h5in2['entry/data']
            data_group_out = h5out['entry/data']

            # Copy datasets except 'images'
            for name, dataset in data_group_in1.items():
                if name != 'images':
                    data = np.concatenate([dataset[:], data_group_in2[name][:]])
                    data_group_out.create_dataset(name, data=data, dtype=dataset.dtype, maxshape=(None,) + dataset.shape[1:])
                    # Copy attributes
                    for attr_name, attr_value in dataset.attrs.items():
                        data_group_out[name].attrs[attr_name] = attr_value

            # Merge the 'images' dataset
            images_dset1 = data_group_in1['images']
            images_dset2 = data_group_in2['images']

            chunks = images_dset1.chunks if images_dset1.chunks else (1000,) + images_dset1.shape[1:]
            compression = images_dset1.compression
            compression_opts = images_dset1.compression_opts

            # Create the output dataset with the combined shape
            new_shape = (images_dset1.shape[0] + images_dset2.shape[0],) + images_dset1.shape[1:]
            images_out = data_group_out.create_dataset(
                'images',
                shape=new_shape,
                maxshape=(None,) + images_dset1.shape[1:],
                dtype='float32',
                chunks=chunks,
                compression=compression,
                compression_opts=compression_opts
            )

            # Copy images in chunks with progress bar
            chunk_size = 1000
            total_chunks_1 = (images_dset1.shape[0] + chunk_size - 1) // chunk_size
            total_chunks_2 = (images_dset2.shape[0] + chunk_size - 1) // chunk_size

            with tqdm(total=total_chunks_1 + total_chunks_2, desc='Merging images', unit='chunk') as pbar:
                # Copy images from the first dataset
                for i in range(0, images_dset1.shape[0], chunk_size):
                    chunk_end = min(i + chunk_size, images_dset1.shape[0])
                    images_chunk = images_dset1[i:chunk_end]
                    images_out[i:chunk_end] = images_chunk.astype('float32')
                    pbar.update(1)

                # Copy images from the second dataset
                offset = images_dset1.shape[0]
                for i in range(0, images_dset2.shape[0], chunk_size):
                    chunk_end = min(i + chunk_size, images_dset2.shape[0])
                    images_chunk = images_dset2[i:chunk_end]
                    images_out[offset + i:offset + chunk_end] = images_chunk.astype('float32')
                    pbar.update(1)

            # Copy attributes from the original images datasets to the new one
            for attr_name, attr_value in images_dset1.attrs.items():
                images_out.attrs[attr_name] = attr_value

    except Exception as e:
        print(f"Error merging files {file1} and {file2}: {e}")

if __name__ == "__main__":
    file1 = "/home/buster/UOX12/UOX1_minpeaks_15.h5"  # Example input file path 1
    file2 = "/home/buster/UOX12/UOX2_minpeaks_15.h5" # Example input file path 2
    output_file = "/home/buster/UOX12/UOX_merged.h5"  # Output file path
    merge_h5_files(file1, file2, output_file)
