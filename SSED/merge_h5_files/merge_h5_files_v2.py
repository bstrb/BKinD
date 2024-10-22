import h5py
from tqdm import tqdm

def merge_h5_files(input_files, output_file, chunk_size=1000):
    try:
        total_chunks = 0
        # Calculate total number of chunks across all files for progress bar
        for input_file in input_files:
            with h5py.File(input_file, 'r') as in_file:
                num_images = in_file['entry/data']['images'].shape[0]
                total_chunks += (num_images + chunk_size - 1) // chunk_size  # Total chunks for this file

        with h5py.File(output_file, 'w') as out_file:
            # Create the overall structure in the output file
            out_entry = out_file.create_group('entry')
            out_data_group = out_entry.create_group('data')

            # Dynamically infer datasets from input file
            with h5py.File(input_files[0], 'r') as in_file:
                dataset_names = list(in_file['entry/data'].keys())

            data_types = {
                'center_x': 'float64', 'center_y': 'float64', 'det_shift_x_mm': 'float64', 'det_shift_y_mm': 'float64',
                'index': 'float64', 'nPeaks': 'float64', 'peakTotalIntensity': 'float64', 'peakXPosRaw': 'float64', 'peakYPosRaw': 'float64'
            }

            # Initialize datasets with appropriate shapes, data types, and storage layouts
            for dataset_name in dataset_names:
                dtype = data_types.get(dataset_name, 'float64')
                if dataset_name == 'images':
                    image_shape = in_file['entry/data']['images'].shape[1:]
                    out_data_group.create_dataset('images', shape=(0, *image_shape), maxshape=(None, *image_shape), dtype='int16', chunks=(1000, 1024, 1024))
                else:
                    out_data_group.create_dataset(dataset_name, shape=(0,), maxshape=(None,), dtype=dtype)

            current_index = 0

            # Process each input file
            with tqdm(total=total_chunks, desc="Merging HDF5 chunks", unit="chunk") as pbar:
                for input_file in input_files:
                    try:
                        with h5py.File(input_file, 'r') as in_file:
                            in_data_group = in_file['entry/data']

                            num_images = in_data_group['images'].shape[0]
                            # Resize the output dataset for 'images'
                            out_data_group['images'].resize(current_index + num_images, axis=0)

                            # Merge images in chunks
                            for start in range(0, num_images, chunk_size):
                                end = min(start + chunk_size, num_images)
                                out_data_group['images'][current_index:current_index + end - start] = in_data_group['images'][start:end]
                                current_index += end - start
                                
                                # Update the progress bar for each chunk processed
                                pbar.update(1)

                            # Now copy other datasets for this file
                            for dataset_name in dataset_names:
                                if dataset_name != 'images':
                                    dataset = in_data_group[dataset_name]
                                    dataset_length = dataset.shape[0]
                                    out_dataset = out_data_group[dataset_name]

                                    # Resize the output dataset to accommodate new data
                                    out_dataset.resize(out_dataset.shape[0] + dataset_length, axis=0)
                                    out_dataset[-dataset_length:] = dataset[:]
                    except OSError as e:
                        print(f"Error opening file {input_file}: {e}")
                        continue
    except Exception as e:
        print(f"An error occurred during merging: {e}")
