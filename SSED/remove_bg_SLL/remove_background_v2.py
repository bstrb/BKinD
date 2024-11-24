import numpy as np
import h5py
from process_image import process_image
import multiprocessing

def copy_without_dataset(source_group, dest_group, exclude_paths=[]):
    for name, item in source_group.items():
        source_path = item.name
        if source_path in exclude_paths:
            continue
        if isinstance(item, h5py.Group):
            dest_subgroup = dest_group.create_group(name)
            # Copy attributes
            for attr_name, attr_value in item.attrs.items():
                dest_subgroup.attrs[attr_name] = attr_value
            # Recursively copy subgroup
            copy_without_dataset(item, dest_subgroup, exclude_paths)
        elif isinstance(item, h5py.Dataset):
            # Copy dataset
            dest_dataset = dest_group.create_dataset(name, data=item[...], dtype=item.dtype)
            # Copy attributes
            for attr_name, attr_value in item.attrs.items():
                dest_dataset.attrs[attr_name] = attr_value

def process_image_worker(args):
    image_index, h5_file_path = args
    try:
        # Each worker opens its own HDF5 file handle
        with h5py.File(h5_file_path, 'r') as h5_file:
            images_dataset = h5_file['/entry/data/images']
            center_x_dataset = h5_file['/entry/data/center_x']
            center_y_dataset = h5_file['/entry/data/center_y']

            image = images_dataset[image_index, :, :].astype(np.float32)
            center_x = center_x_dataset[image_index]
            center_y = center_y_dataset[image_index]

        # Process the image
        corrected_image = process_image(image, center_x, center_y)

        return (image_index, corrected_image)
    except Exception as e:
        # Return the exception to handle it in the main process
        return (image_index, e)

def remove_background(h5_file_path, new_h5_file_path, selected_indices=None):
    # Open the original HDF5 file to get the number of images
    with h5py.File(h5_file_path, 'r') as h5_file:
        num_images = h5_file['/entry/data/images'].shape[0]
        image_shape = h5_file['/entry/data/images'].shape[1:]  # Assuming images are 2D
        if selected_indices is None:
            selected_indices = range(num_images)
        else:
            # Ensure selected_indices are within the valid range
            selected_indices = [idx for idx in selected_indices if 0 <= idx < num_images]

    print(f"Processing {len(selected_indices)} selected images.")

    # Determine chunk size (adjust as needed)
    chunk_size = min(1000, len(selected_indices))

    # Open the new HDF5 file and create the datasets
    with h5py.File(new_h5_file_path, 'w') as new_h5_file:
        # Exclude the 'images' dataset
        exclude_paths = ['/entry/data/images']

        # Copy all datasets and groups except excluded paths
        with h5py.File(h5_file_path, 'r') as h5_file:
            copy_without_dataset(h5_file, new_h5_file, exclude_paths=exclude_paths)

        # Create the 'images' dataset in the new file and save the processed images
        images_group = new_h5_file['/entry/data']
        corrected_images_dataset = images_group.create_dataset(
            'images',
            shape=(len(selected_indices), 1024, 1024),
            dtype='float32',
            chunks=(chunk_size, 1024, 1024)
        )

        # Process images in chunks
        num_images = len(selected_indices)
        for chunk_start in range(0, num_images, chunk_size):
            chunk_indices = selected_indices[chunk_start:chunk_start + chunk_size]

            print(f"Processing images {chunk_start + 1} to {chunk_start + len(chunk_indices)}...")

            # Prepare arguments for worker processes
            args_list = [(image_index, h5_file_path) for image_index in chunk_indices]

            # Use multiprocessing Pool
            with multiprocessing.Pool() as pool:
                results = pool.map(process_image_worker, args_list)

            # Collect and write results
            for image_index, result in results:
                idx_in_dataset = selected_indices.index(image_index)
                if isinstance(result, Exception):
                    print(f"Error processing image {image_index}: {result}")
                    continue  # Handle or log the error as needed
                corrected_images_dataset[idx_in_dataset, :, :] = result

    print("Processing completed.")

# Example usage
if __name__ == '__main__':
    h5_file_path = 'input.h5'
    new_h5_file_path = 'output.h5'
    remove_background(h5_file_path, new_h5_file_path)
