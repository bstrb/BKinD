import numpy as np
import h5py
from process_image import process_image

def load_mask(mask_file_path):
    with h5py.File(mask_file_path, 'r') as h5_file:
        mask = h5_file['/mask'][()]
        print(f"Loaded mask with shape {mask.shape} and dtype {mask.dtype}", flush=True)
    return mask

def find_center(h5_file_path, new_h5_file_path, mask_path, selected_indices=None, plot=False, verbose=True):
    # Load the mask once
    mask = load_mask(mask_path)

    # Open the HDF5 file to get images
    with h5py.File(h5_file_path, 'r') as h5_file:
        images_dataset = h5_file['/entry/data/images']
        num_images = images_dataset.shape[0]

        if selected_indices is None:
            selected_indices = range(num_images)
        else:
            selected_indices = [idx for idx in selected_indices if 0 <= idx < num_images]

        print(f"Processing {len(selected_indices)} selected images.", flush=True)

        total_images = len(selected_indices)

        # Create the output HDF5 file and dataset
        with h5py.File(new_h5_file_path, 'w') as new_h5_file:
            beam_centers_dataset = new_h5_file.create_dataset(
                'beam_centers',
                shape=(total_images, 2),
                dtype='int16'
            )

            # Process images
            for idx_in_dataset, image_index in enumerate(selected_indices):
                try:
                    image = images_dataset[image_index, :, :].astype(np.float32)
                    beam_center = process_image(image, image_index, mask, plot=plot, verbose=verbose)
                    print(f"Center for image {image_index} found at {beam_center}")
                    if beam_center is None:
                        print(f"Skipping image {image_index} due to processing error.", flush=True)
                        continue

                    beam_centers_dataset[idx_in_dataset, :] = beam_center
                except Exception as e:
                    print(f"Error processing image {image_index}: {e}", flush=True)
                    continue

        print("Processing completed.", flush=True)

# Example usage
if __name__ == '__main__':
    h5_file_path = '/home/buster/UOX1/UOX1_min_10/UOX1_min_10.h5'
    mask_path = '/home/buster/mask/pxmask.h5'
    selected_indices = [1300]  # Change to desired indices

    new_h5_file_path = f'/home/buster/UOX1/UOX1_min_10/UOX1_min_10_CF_{selected_indices}.h5'
    # Set plot=True to visualize the results
    find_center(h5_file_path, new_h5_file_path, mask_path, selected_indices=selected_indices, plot=True, verbose=True)
