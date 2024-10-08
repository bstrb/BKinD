import h5py
import numpy as np
import os
from tqdm import tqdm

def convert_hdf5_images_to_floats(input_file, dataset_name='entry/data/images'):
    # Create output file name by adding '_32float' to the input file name
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_32float{ext}"

    # Open the input HDF5 file in read mode
    with h5py.File(input_file, 'r') as infile:
        # Prepare the output HDF5 file
        with h5py.File(output_file, 'w') as outfile:
            # Copy all groups and datasets except the target dataset
            infile.copy('entry', outfile)

            # Extract the dataset containing images
            if dataset_name not in infile:
                raise KeyError(f"Dataset '{dataset_name}' not found in file '{input_file}'")
            images = infile[dataset_name]

            # Overwrite the target dataset with the converted images
            del outfile[dataset_name]
            out_dataset = outfile.create_dataset(dataset_name, shape=images.shape, dtype=np.float32, chunks=(1000, 1024, 1024))

            # Process and convert images in chunks with progress bar
            for i in tqdm(range(0, images.shape[0], 1000), desc="Converting images to float"):
                chunk = images[i:i+1000]
                chunk_float = chunk.astype(np.float32)
                out_dataset[i:i+1000] = chunk_float

if __name__ == '__main__':
    input_hdf5_file = '/home/buster/UOX1/min_15_peak.h5' # Path to your input HDF5 file

    convert_hdf5_images_to_floats(input_hdf5_file)
    print(f"Conversion completed! Converted images saved to '{input_hdf5_file.replace('.h5', '_32float.h5')}'")