import numpy as np
import h5py
from process_image_v5 import process_image

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

def remove_background(h5_file_path, new_h5_file_path, selected_indices=None):
    # Open the original HDF5 file to get the number of images
    with h5py.File(h5_file_path, 'r') as h5_file:
        num_images = h5_file['/entry/data/images'].shape[0]
        if selected_indices is None:
            selected_indices = range(num_images)
        else:
            # Ensure selected_indices are within the valid range
            selected_indices = [idx for idx in selected_indices if 0 <= idx < num_images]

    print(f"Processing {len(selected_indices)} selected images.")

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
            dtype='float32'
        )

        # Process each selected image
        with h5py.File(h5_file_path, 'r') as h5_file:
            images_dataset = h5_file['/entry/data/images']
            center_x_dataset = h5_file['/entry/data/center_x']
            center_y_dataset = h5_file['/entry/data/center_y']

            for i, image_index in enumerate(selected_indices):
                image = images_dataset[image_index, :, :].astype(np.float32)
                center_x = center_x_dataset[image_index]
                center_y = center_y_dataset[image_index]

                print(f"Processing image {image_index + 1} with center coordinates: center_x = {center_x}, center_y = {center_y}")

                # Process the image
                corrected_image = process_image(image, center_x, center_y)
                corrected_images_dataset[i, :, :] = corrected_image

    print("Processing completed.")
