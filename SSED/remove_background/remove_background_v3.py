import numpy as np
import h5py
from damped_sinusoid_v2 import process_image

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

def process_and_write_batch(start_index, end_index, images_dataset, center_x_dataset, center_y_dataset, corrected_images_dataset):
    for image_index in range(start_index, end_index):
        image = images_dataset[image_index, :, :].astype(np.float32)
        center_x = center_x_dataset[image_index]
        center_y = center_y_dataset[image_index]

        print(f"Processing image {image_index+1}/{images_dataset.shape[0]} with center coordinates: center_x = {center_x}, center_y = {center_y}")

        # Process the image
        corrected_image = process_image(image, center_x, center_y)

        # Write the corrected image to the new dataset
        corrected_images_dataset[image_index, :, :] = corrected_image

def remove_background(h5_file_path, new_h5_file_path, batch_size=1000):

    # Open the original HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Open the new HDF5 file
        with h5py.File(new_h5_file_path, 'w') as new_h5_file:
            # Exclude the 'images' dataset
            exclude_paths = ['/entry/data/images']

            # Copy all datasets and groups except excluded paths
            copy_without_dataset(h5_file, new_h5_file, exclude_paths=exclude_paths)

            # Now process the images and save to the new file
            images_dataset = h5_file['/entry/data/images']
            center_x_dataset = h5_file['/entry/data/center_x']
            center_y_dataset = h5_file['/entry/data/center_y']

            num_images = images_dataset.shape[0]

            print(f"The dataset contains {num_images} images.")

            # Create the 'images' dataset in the new file
            images_group = new_h5_file['/entry/data']
            corrected_images_dataset = images_group.create_dataset(
                'images', 
                shape=images_dataset.shape, 
                dtype='float32', 
                chunks=(1000, 1024, 1024)
            )

            # Process images sequentially
            for start_index in range(0, num_images, batch_size):
                end_index = min(start_index + batch_size, num_images)
                process_and_write_batch(
                    start_index,
                    end_index,
                    images_dataset,
                    center_x_dataset,
                    center_y_dataset,
                    corrected_images_dataset
                )
