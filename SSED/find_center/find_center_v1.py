import numpy as np
import h5py
from tqdm import tqdm

def load_mask(mask_file_path):
    with h5py.File(mask_file_path, 'r') as h5_file:
        # Corrected dataset path
        mask = h5_file['/mask'][()]
        print(f"Loaded mask with shape {mask.shape} and dtype {mask.dtype}")
    return mask

def process_image(image, mask):
    try:
        # Apply the mask to the image
        masked_image = image * mask

        # Compute the center of mass of the masked image
        y_indices, x_indices = np.indices(masked_image.shape)
        total_intensity = np.sum(masked_image)

        if total_intensity == 0:
            # No intensity in the image
            return None, None

        center_x = np.sum(x_indices * masked_image) / total_intensity
        center_y = np.sum(y_indices * masked_image) / total_intensity

        print(f"Computed center: (x={center_x:.2f}, y={center_y:.2f})")

        # Return the computed center coordinates
        return (center_x, center_y)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def find_center(h5_file_path, new_h5_file_path, mask_path, selected_indices=None):
    # Load the mask once
    mask = load_mask(mask_path)

    # Open the original HDF5 file to get the number of images and image shape
    with h5py.File(h5_file_path, 'r') as h5_file:
        images_dataset = h5_file['/entry/data/images']
        num_images = images_dataset.shape[0]
        image_shape = images_dataset.shape[1:]  # Assuming images are 2D

        if selected_indices is None:
            selected_indices = range(num_images)
        else:
            # Ensure selected_indices are within the valid range
            selected_indices = [idx for idx in selected_indices if 0 <= idx < num_images]

        print(f"Processing {len(selected_indices)} selected images.")

        total_images = len(selected_indices)

        # Open the new HDF5 file and create the beam_centers dataset
        with h5py.File(new_h5_file_path, 'w') as new_h5_file:
            # Create a dataset for beam centers
            beam_centers_dataset = new_h5_file.create_dataset(
                'beam_centers',
                shape=(total_images, 2),
                dtype='float64'
            )

            # Process images sequentially
            with tqdm(total=total_images, desc='Processing images') as pbar:
                for idx_in_dataset, image_index in enumerate(selected_indices):
                    try:
                        image = images_dataset[image_index, :, :].astype(np.float32)

                        # Process the image to find the beam center
                        beam_center = process_image(image, mask)

                        if beam_center is None:
                            print(f"Skipping image {image_index} due to processing error.")
                            continue

                        # Store the beam center coordinates
                        beam_centers_dataset[idx_in_dataset, :] = beam_center
                    except Exception as e:
                        print(f"Error processing image {image_index}: {e}")
                        continue

                    pbar.update(1)  # Update progress bar

    print("Processing completed.")

# Example usage
if __name__ == '__main__':
    h5_file_path = '/home/buster/UOX1/UOX1_min_50/UOX1_min_50_peak.h5'
    new_h5_file_path = '/home/buster/UOX1/UOX1_min_50/UOX1_min_50_peak_centerfound.h5'
    mask_path = '/home/buster/mask/pxmask.h5'
    selected_indices = [1800] # list(range(1500, 1501)) 

    find_center(h5_file_path, new_h5_file_path, mask_path, selected_indices=selected_indices)