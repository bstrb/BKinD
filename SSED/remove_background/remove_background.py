import numpy as np 
import h5py

# from read_mask_file import read_mask_file
from super_lorentzian import process_image

# Main processing function

def remove_background(h5_file_path):#, mask_file_path):
    # Read the mask
    # mask = read_mask_file(mask_file_path)

    # # Convert mask to binary values (assuming 1 is valid and 2 is masked)
    # mask = np.where(mask == 2, 0, mask)  # Convert 2 to 0 (masked), retain 1 as valid
    # mask = mask.astype(bool)  # Ensure the mask is boolean

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Access the datasets
        images_dataset = h5_file['/entry/data/images']
        center_x_dataset = h5_file['/entry/data/center_x']
        center_y_dataset = h5_file['/entry/data/center_y']

        # Get the number of images
        num_images = images_dataset.shape[0]
        print(f"The dataset contains {num_images} images.")

        # Prompt the user to select an image index
        image_index = int(input(f"Enter the index of the image to process (0 to {num_images - 1}): "))
        if image_index < 0 or image_index >= num_images:
            raise ValueError("Invalid image index.")

        # Extract the image and center coordinates
        image = images_dataset[image_index, :, :].astype(np.float64)
        center_x = center_x_dataset[image_index]
        center_y = center_y_dataset[image_index]

        print(f"Using center coordinates: center_x = {center_x}, center_y = {center_y}")

        # # Apply the mask from the mask file
        # if mask.shape != image.shape:
        #     raise ValueError("Mask shape does not match image shape.")

        # Process the image
        corrected_image = process_image(
            image,
            # mask,
            center_x,
            center_y
        )

        # Unpack processed data
        # corrected_image, corrected_image_no_mask, radial_distances, radial_means, popt_pv = processed_data


if __name__ == "__main__":
    # Paths to the files
    h5_file_path = '/Users/xiaodong/Desktop/deiced_UOX1_min_15_peak.h5'
    #h5_file_path = '/home/buster/UOX1/deiced_UOX1_min_15_peak.h5'

    mask_file_path = '/home/buster/mask/pxmask.h5'

    # Call the main processing function
    remove_background(h5_file_path)#, mask_file_path)
