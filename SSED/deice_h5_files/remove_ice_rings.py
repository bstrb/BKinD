import h5py
import numpy as np
import os
import gc  # Import garbage collector interface
from tqdm import tqdm  # Import progress bar

def remove_ice_rings(input_file, output_file, min_spots=10, inner_radius=100, outer_radius=150):
    # Check if output file already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    try:
        with h5py.File(input_file, 'r') as h5in:
            # Check if 'entry/data' group exists
            if 'entry/data' not in h5in:
                print(f"Group 'entry/data' not found in file {input_file}")
                return

            # Filter indices based on the number of spots within the annulus
            nPeaks = h5in['/entry/data/nPeaks'][:]
            centerX = h5in['/entry/data/center_x'][:]
            centerY = h5in['/entry/data/center_y'][:]
            peakXPosRaw = h5in['/entry/data/peakXPosRaw'][:]
            peakYPosRaw = h5in['/entry/data/peakYPosRaw'][:]

            valid_indices = []
            for i in range(nPeaks.shape[0]):
                distances = np.sqrt((peakXPosRaw[i, :nPeaks[i]] - centerX[i])**2 + (peakYPosRaw[i, :nPeaks[i]] - centerY[i])**2)
                spots_within_annulus = np.sum((distances >= inner_radius) & (distances <= outer_radius))
                if spots_within_annulus >= min_spots:
                    valid_indices.append(i)
                del distances
            valid_indices = np.array(valid_indices)

        # Create the output file and copy necessary datasets
        with h5py.File(input_file, 'r') as h5in, h5py.File(output_file, 'w') as h5out:
            # Create 'entry/data' group in the output file
            h5out.create_group('entry/data')

            # Copy datasets except 'images', filtering based on valid indices
            for name, dataset in h5in['/entry/data'].items():
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
            images_dset = h5in['/entry/data/images']
            chunks = images_dset.chunks if images_dset.chunks else (min(1000, len(valid_indices)), 1024, 1024)
            compression = images_dset.compression if images_dset.compression else None
            compression_opts = images_dset.compression_opts if images_dset.compression_opts else None

            new_shape = (len(valid_indices),) + images_dset.shape[1:]
            images_out = h5out['entry/data'].create_dataset(
                'images',
                shape=new_shape,
                maxshape=new_shape,
                dtype='float32',
                chunks=(min(1000, len(valid_indices)),) + images_dset.shape[1:],
                compression=compression,
                compression_opts=compression_opts
            )

            # Create a progress bar for processing images in chunks
            chunk_size = 1000
            total_chunks = max((len(valid_indices) + chunk_size - 1) // chunk_size, 1)
            progress_bar = tqdm(total=total_chunks, desc='Processing images', unit='chunk', miniters=1)

            # Copy valid images in chunks
            for i in range(0, len(valid_indices), chunk_size):
                chunk_indices = valid_indices[i:i + chunk_size]
                images_chunk = images_dset[chunk_indices].astype('float32')
                images_out[i:i + len(chunk_indices)] = images_chunk
                gc.collect()
                progress_bar.update(1)

            progress_bar.close()

            # Copy attributes from the original images dataset to the new one
            for attr_name, attr_value in images_dset.attrs.items():
                images_out.attrs[attr_name] = attr_value

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

if __name__ == "__main__":
    input_file = "/home/buster/UOX123/UOX3_minpeaks_15.h5"  # Example input file path
    output_file = "/home/buster/UOX123/deiced_UOX3_minpeaks_15.h5"  # Example output file path
    min_spots = 5  # Set the threshold for minimum number of spots within the annulus
    inner_radius = 60  # Set the inner radius of the annulus in pixels
    outer_radius = 155  # Set the outer radius of the annulus in pixels
    
    remove_ice_rings(input_file, output_file, min_spots, inner_radius, outer_radius)
