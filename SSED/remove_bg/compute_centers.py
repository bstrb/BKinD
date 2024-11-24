import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
import multiprocessing
import logging
from scipy.optimize import differential_evolution
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')  # Use 'TkAgg' or 'Agg' depending on your environment
import matplotlib.pyplot as plt
def asymmetry_error(center, image, lower_threshold=10):
    center_x, center_y = center

    # Compute shifted indices based on the center
    y_indices, x_indices = np.indices(image.shape)
    x_shifted = x_indices - center_x
    y_shifted = y_indices - center_y

    # Mirror the image across both axes
    image_mirrored = np.flip(np.flip(image, axis=0), axis=1)

    # Compute the difference between the original and mirrored images
    diff = image - image_mirrored

    # Compute radial distance from the center
    distance = np.sqrt(x_shifted**2 + y_shifted**2)

    # Define maximum distance based on the center position
    max_distance = min(center_x, image.shape[1] - center_x, center_y, image.shape[0] - center_y)

    # Create radial mask
    radial_mask = distance <= max_distance

    # Create intensity mask
    median_intensity = np.median(image)
    upper_threshold = median_intensity * 1.2  # Adjust as needed

    intensity_mask = (image >= lower_threshold) & (image <= upper_threshold)

    # Combine masks
    mask = radial_mask & intensity_mask

    # Ensure mask is not empty
    if np.sum(mask) == 0:
        return np.inf  # Return a large error if mask is empty

    # Apply the mask to the difference
    error = np.sum((diff[mask])**2)

    # Debugging: Print the error and number of masked pixels
    num_masked_pixels = np.sum(mask)
    print(f"Center: ({center[0]:.2f}, {center[1]:.2f}), Error: {error:.2f}, Masked Pixels: {num_masked_pixels}")

    return error


def find_center(image, initial_center=None, visualize=False, index=None):
    try:
        # Preprocess the image
        smoothed_image = gaussian_filter(image, sigma=2)

        # Initial guess for the center
        if initial_center is None:
            initial_center = (image.shape[1] / 2, image.shape[0] / 2)

        # Bounds around the initial center
        bounds = [
            (initial_center[0] - 50, initial_center[0] + 50),
            (initial_center[1] - 50, initial_center[1] + 50)
        ]

        # Objective function to minimize
        def objective(center):
            error = asymmetry_error(center, smoothed_image, lower_threshold=10)
            print(f"Evaluating center at ({center[0]:.2f}, {center[1]:.2f}), Error: {error:.2f}")
            return error

        print("Starting optimization")
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            disp=True
        )
        print("Optimization completed")
        print(f"Optimization result: {result}")

        center_x, center_y = result.x

        if visualize:
            visualize_center(image, center_x, center_y, index)
            
        return center_x, center_y
    except Exception as e:
        logging.error(f"Error finding center: {e}", exc_info=True)
        return None, None

def visualize_center(image, center_x, center_y, index=None):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image, cmap='gray', origin='lower')
    ax.scatter(center_x, center_y, color='red', marker='x', s=100, label='Estimated Center')
    ax.legend()
    if index is not None:
        ax.set_title(f'Detected Center for Image Index {index}')
    else:
        ax.set_title('Detected Center')
    plt.show()
    plt.close(fig)

def compute_centers(h5_file_path, updated_h5_file_path, selected_indices=None, visualize=False):
    # Set up logging
    logging.basicConfig(filename='center_finding_errors.log', level=logging.ERROR)

    # Open the original HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        num_images = h5_file['/entry/data/images'].shape[0]

    if selected_indices is None:
        selected_indices = list(range(num_images))
    else:
        selected_indices = [idx for idx in selected_indices if 0 <= idx < num_images]

    total_images = len(selected_indices)

    with h5py.File(updated_h5_file_path, 'w') as new_h5_file:
        new_h5_file['/entry/data/images'] = h5py.ExternalLink(h5_file_path, '/entry/data/images')
        center_x_dataset = new_h5_file.create_dataset('/entry/data/center_x', shape=(num_images,), dtype='float32')
        center_y_dataset = new_h5_file.create_dataset('/entry/data/center_y', shape=(num_images,), dtype='float32')

        def process_image_center(index):
            with h5py.File(h5_file_path, 'r') as h5_file:
                images_dataset = h5_file['/entry/data/images']
                image = images_dataset[index, :, :]
                print(f"Processing image {index}, shape: {image.shape}")
                center_x, center_y = find_center(image, initial_center=(508, 515), visualize=visualize, index=index)
                if center_x is None or center_y is None:
                    logging.error(f"Center not found for image index {index}")
                return index, center_x, center_y

        args_list = selected_indices

        if not visualize:
            with multiprocessing.Pool() as pool:
                with tqdm(total=total_images, desc='Computing centers') as pbar:
                    for index, center_x, center_y in pool.imap_unordered(process_image_center, args_list):
                        if center_x is not None and center_y is not None:
                            center_x_dataset[index] = center_x
                            center_y_dataset[index] = center_y
                        else:
                            center_x_dataset[index] = np.nan
                            center_y_dataset[index] = np.nan
                        pbar.update(1)
        else:
            for index in tqdm(args_list, desc='Computing centers'):
                index, center_x, center_y = process_image_center(index)
                if center_x is not None and center_y is not None:
                    center_x_dataset[index] = center_x
                    center_y_dataset[index] = center_y
                else:
                    center_x_dataset[index] = np.nan
                    center_y_dataset[index] = np.nan

    print("Center computation completed.")

if __name__ == '__main__':
    h5_file_path = '/home/buster/UOX1/UOX1.h5'
    updated_h5_file_path = '/home/buster/UOX1/UOX1_center.h5'

    # Process a specific image index
    selected_indices = [1500]

    # Set visualize=True to display the estimated center
    compute_centers(h5_file_path, updated_h5_file_path, selected_indices=selected_indices, visualize=True)
