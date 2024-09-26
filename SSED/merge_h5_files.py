import h5py
from tqdm import tqdm

def merge_h5_files(input_files, output_file, chunk_size=5000):
    with h5py.File(output_file, 'w') as out_file:
        # Create the overall structure in the output file
        out_entry = out_file.create_group('entry')
        out_data_group = out_entry.create_group('data')

        # Initialize datasets with appropriate shapes and data types
        data_types = {
            'center_x': 'f', 'center_y': 'f', 'det_shift_x_mm': 'f', 'det_shift_y_mm': 'f', 
            'index': 'f', 'nPeaks': 'f', 'peakTotalIntensity': 'f', 'peakXPosRaw': 'f', 'peakYPosRaw': 'f'
        }

        for dataset_name, dtype in data_types.items():
            if dataset_name in ['peakTotalIntensity', 'peakXPosRaw', 'peakYPosRaw']:
                out_data_group.create_dataset(dataset_name, shape=(0, 500), maxshape=(None, 500), dtype=dtype, chunks=True)
            else:
                out_data_group.create_dataset(dataset_name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)

        # Merge 'images' dataset differently since it has more dimensions
        image_shape = None
        total_images = 0

        # Calculate total images and determine image shape from all files first
        for input_file in input_files:
            with h5py.File(input_file, 'r') as in_file:
                in_data_group = in_file['entry/data']
                images_dataset = in_data_group['images']
                total_images += images_dataset.shape[0]
                if image_shape is None:
                    image_shape = images_dataset.shape[1:]

        # images_out_dataset = out_data_group.create_dataset(
        #     'images', shape=(0, *image_shape), maxshape=(None, *image_shape), dtype='int16', chunks=True
        # )

        images_out_dataset = out_data_group.create_dataset(
            'images', shape=(0, *image_shape), maxshape=(None, *image_shape), dtype='int16', chunks=(1000, 1024, 1024)
        )

        current_index = 0
        
        for input_file in input_files:
            with h5py.File(input_file, 'r') as in_file:
                in_data_group = in_file['entry/data']

                num_images = in_data_group['images'].shape[0]
                # Resize the output dataset for 'images'
                images_out_dataset.resize(current_index + num_images, axis=0)
                
                for start in tqdm(range(0, num_images, chunk_size), desc=f"Merging images from {input_file}"):
                    end = min(start + chunk_size, num_images)
                    images_out_dataset[current_index:current_index + end - start] = in_data_group['images'][start:end]
                    current_index += end - start

                # Now copy other datasets for this file
                for dataset_name in ['center_x', 'center_y', 'det_shift_x_mm', 'det_shift_y_mm', 'index', 
                                     'nPeaks', 'peakTotalIntensity', 'peakXPosRaw', 'peakYPosRaw']:
                    dataset = in_data_group[dataset_name]
                    dataset_length = dataset.shape[0]
                    out_dataset = out_data_group[dataset_name]
                    
                    # Resize the output dataset to accommodate new data
                    if len(dataset.shape) == 1:
                        out_dataset.resize(out_dataset.shape[0] + dataset_length, axis=0)
                        out_dataset[-dataset_length:] = dataset[:]
                    else:  # For multi-dimensional data (e.g., 27025 x 500)
                        out_dataset.resize(out_dataset.shape[0] + dataset_length, axis=0)
                        for i in tqdm(range(0, dataset_length, chunk_size), desc=f"Merging {dataset_name} from {input_file}"):
                            end_idx = min(i + chunk_size, dataset_length)
                            out_dataset[-(dataset_length - i):-(dataset_length - end_idx)] = dataset[i:end_idx]

path1 = '/home/buster/UOX1/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5'
path2 = '/home/buster/UOX2/UOX_His_MUA_450nm_spot4_ON_20240311_1014.h5'
path3 = '/home/buster/UOX3/UOX_His_MUA_450nm_spot4_ON_20240311_0737.h5'

# Usage example
input_files = [path1, path2, path3]
output_file = '/home/buster/merged_UOX.h5'
merge_h5_files(input_files, output_file)
