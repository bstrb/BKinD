import numpy as np
import h5py
import multiprocessing
from scipy.optimize import curve_fit

# Include your process_image function here
def damped_sinusoid(r, A_ds, k, phi, tau):
    return A_ds * np.sin(k * r + phi) * np.exp(-r / tau)

def sum_damped_sinusoids(r, *params):
    N = len(params) // 4  # Number of damped sinusoids
    result = np.zeros_like(r)
    for i in range(N):
        A_ds = params[4 * i]
        k = params[4 * i + 1]
        phi = params[4 * i + 2]
        tau = params[4 * i + 3]
        result += damped_sinusoid(r, A_ds, k, phi, tau)
    return result

def compute_radial_statistics(radii_filtered, image_filtered, bins):
    num_bins = len(bins)
    radial_medians = []
    radial_stds = []
    radial_distances = []
    for i in range(1, num_bins):
        bin_mask = (radii_filtered >= bins[i - 1]) & (radii_filtered < bins[i])
        if np.any(bin_mask):
            median_intensity = np.median(image_filtered[bin_mask])
            std_intensity = np.std(image_filtered[bin_mask])
            radial_medians.append(median_intensity)
            radial_stds.append(std_intensity)
            radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center
    return np.array(radial_medians), np.array(radial_stds), np.array(radial_distances)

def process_image(image, center_x, center_y):

    # Compute radial distances from the center
    y_indices, x_indices = np.indices(image.shape)
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # Flatten the image and radii
    image_flat = image.flatten()
    radii_flat = radii.flatten()

    # Define max_radius as maximum radius to consider
    max_radius = int(np.min(image.shape) / np.sqrt(2) - 10)

    # Filter data within max_radius
    within_limit_mask = radii_flat <= max_radius
    radii_filtered = radii_flat[within_limit_mask]
    image_filtered = image_flat[within_limit_mask]

    # Bin the data to compute median intensity in each bin
    num_bins = int(max_radius)
    bins = np.linspace(0, max_radius, num_bins)

    # Compute radial statistics
    radial_medians, radial_stds, radial_distances = compute_radial_statistics(radii_filtered, image_filtered, bins)

    # Reverse arrays to start from max_radius moving towards the center
    radial_medians_rev = radial_medians[::-1]
    radial_distances_rev = radial_distances[::-1]
    radial_stds_rev = radial_stds[::-1]

    # Compute gradient to detect sharp intensity drop due to beam stopper
    gradient_rev = np.gradient(radial_medians_rev)

    # Dynamic drop threshold based on data statistics
    gradient_median = np.median(gradient_rev)
    gradient_std = np.std(gradient_rev)
    drop_threshold = gradient_median - 1 * gradient_std  # Threshold at 1 standard deviation below median

    # Find indices where gradient drops below the threshold
    drop_indices = np.where(gradient_rev <= drop_threshold)[0]

    if len(drop_indices) > 0:
        # Exclude points where intensity drops sharply
        exclude_index = drop_indices[0]
        radial_medians_rev = radial_medians_rev[:exclude_index]
        radial_distances_rev = radial_distances_rev[:exclude_index]
        radial_stds_rev = radial_stds_rev[:exclude_index]

    # Reverse back to correct order
    radial_medians_filtered = radial_medians_rev[::-1]
    radial_distances_filtered = radial_distances_rev[::-1]
    radial_stds_filtered = radial_stds_rev[::-1]

    # Exclude additional points with lowest radius (closest to the center)
    ex_points = 2  # Adjust this number as needed
    radial_medians_filtered = radial_medians_filtered[ex_points:]
    radial_distances_filtered = radial_distances_filtered[ex_points:]
    radial_stds_filtered = radial_stds_filtered[ex_points:]

    # Fit a sum of damped sinusoids to the radial medians
    N_sinusoids = 6 # Number of damped sinusoids to sum
    initial_guess_ds = []
    for i in range(N_sinusoids):
        # Estimate initial parameters based on radial medians
        A_ds = (np.max(radial_medians_filtered) - np.min(radial_medians_filtered)) / N_sinusoids
        k = 2 * np.pi / (50 * (i + 1))  # Adjust the periodicity
        phi = np.pi * i / N_sinusoids
        tau = np.max(radial_medians_filtered)  # Adjust the damping factor
        initial_guess_ds.extend([A_ds, k, phi, tau])

    # Fit the sum of damped sinusoids to the radial medians
    try:
        popt_ds, _ = curve_fit(sum_damped_sinusoids, radial_distances_filtered, radial_medians_filtered, p0=initial_guess_ds, sigma=radial_stds_filtered, absolute_sigma=True, maxfev=100000)
        print("Damped sinusoids fitting successful.")
    except RuntimeError as e:
        print("Damped sinusoids curve fitting failed:", e)
        # popt_ds = initial_guess_ds  # Use initial guess if fitting fails

    # Evaluate the damped sinusoids model over the entire image
    background_ds = sum_damped_sinusoids(radii, *popt_ds)


    # Subtract the background from the original image
    corrected_image = image - background_ds  # Version without masking applied

    # Compute residuals after damped sinusoids fit
    residuals_ds = radial_medians_filtered - sum_damped_sinusoids(radial_distances_filtered, *popt_ds)

    # Plotting results if requested
    # plot_results(image, corrected_image, radial_distances_filtered, radial_medians_filtered, popt_ds, residuals_ds, N_sinusoids)

    return corrected_image


    return corrected_image

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
        with h5py.File(h5_file_path, 'r') as h5_file:
            images_dataset = h5_file['/entry/data/images']
            center_x_dataset = h5_file['/entry/data/center_x']
            center_y_dataset = h5_file['/entry/data/center_y']

            image = images_dataset[image_index, :, :].astype(np.float32)
            center_x = center_x_dataset[image_index]
            center_y = center_y_dataset[image_index]

        print(f"Processing image {image_index+1} with center coordinates: center_x = {center_x}, center_y = {center_y}")

        # Process the image
        corrected_image = process_image(image, center_x, center_y)

        return image_index, corrected_image
    except Exception as e:
        print(f"Error processing image {image_index}: {e}")
        return image_index, None

def remove_background(h5_file_path, new_h5_file_path):
    # Open the original HDF5 file to get the number of images
    with h5py.File(h5_file_path, 'r') as h5_file:
        num_images = h5_file['/entry/data/images'].shape[0]
        print(f"The dataset contains {num_images} images.")

    # Open the new HDF5 file and create the datasets
    with h5py.File(new_h5_file_path, 'w') as new_h5_file:
        # Exclude the 'images' dataset
        exclude_paths = ['/entry/data/images']

        # Copy all datasets and groups except excluded paths
        with h5py.File(h5_file_path, 'r') as h5_file:
            copy_without_dataset(h5_file, new_h5_file, exclude_paths=exclude_paths)

        # Create the 'images' dataset in the new file
        images_group = new_h5_file['/entry/data']
        corrected_images_dataset = images_group.create_dataset(
            'images', 
            shape=(num_images, 1024, 1024), 
            dtype='float32', 
            chunks=(1, 1024, 1024),
            compression="gzip"
        )

        # Prepare arguments for multiprocessing
        args_list = [(image_index, h5_file_path) for image_index in range(num_images)]

        # Use multiprocessing Pool
        with multiprocessing.Pool() as pool:
            # Process images in parallel
            for image_index, corrected_image in pool.imap_unordered(process_image_worker, args_list):
                if corrected_image is not None:
                    # Write the corrected image to the new dataset
                    corrected_images_dataset[image_index, :, :] = corrected_image
                else:
                    print(f"Image {image_index} was not processed due to an error.")

    print("Processing completed.")

if __name__ == '__main__':
    h5_file_path = '/path/to/your/input_file.h5'
    new_h5_file_path = '/path/to/your/output_file.h5'

    remove_background(h5_file_path, new_h5_file_path)
