import numpy as np
import h5py
import multiprocessing
from scipy.optimize import curve_fit

def damped_sinusoid(r, A_ds, k, phi, tau):
    # To prevent overflow, clip the exponent to a minimum value
    exponent = -r / tau
    exponent = np.clip(exponent, -700, 0)  # np.exp(-700) is about 5e-305
    return A_ds * np.sin(k * r + phi) * np.exp(exponent)

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
    radial_medians, radial_stds, radial_distances = compute_radial_statistics(
        radii_filtered, image_filtered, bins)

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

    # Define acceptable residual threshold
    residual_threshold = 0.5  # Adjust this value based on acceptable residual

    # Set minimum and maximum number of sinusoids
    min_N_sinusoids = 3
    max_N_sinusoids = 7

    # Initialize N_sinusoids
    N_sinusoids = min_N_sinusoids

    # Initialize variables
    fit_success = False

    best_residual_norm = np.inf
    best_popt_ds = None

    while N_sinusoids <= max_N_sinusoids:
        # Prepare initial guesses and bounds for current N_sinusoids
        initial_guess_ds = []
        lower_bounds = []
        upper_bounds = []

        for i in range(N_sinusoids):
            # Estimate initial parameters based on radial medians
            A_ds = (np.max(radial_medians_filtered) - np.min(radial_medians_filtered)) / N_sinusoids
            k = 2 * np.pi / (50 * (i + 1))  # Adjust the periodicity
            phi = np.pi * i / N_sinusoids
            tau = np.max(radial_distances_filtered)  # Adjust the damping factor

            initial_guess_ds.extend([A_ds, k, phi, tau])

            # Set bounds for each parameter
            # A_ds: Amplitude between negative and positive reasonable values
            amplitude_min = -np.abs(A_ds) * 10
            amplitude_max = np.abs(A_ds) * 10

            lower_bounds.append(amplitude_min)
            upper_bounds.append(amplitude_max)

            # k: Wave number, positive
            k_min = 0
            k_max = 2 * np.pi / 10  # Adjust based on expected frequency
            lower_bounds.append(k_min)
            upper_bounds.append(k_max)

            # phi: Phase, between -2*pi and 2*pi
            phi_min = -2 * np.pi
            phi_max = 2 * np.pi
            lower_bounds.append(phi_min)
            upper_bounds.append(phi_max)

            # tau: Damping factor, positive
            tau_min = 1e-3  # Avoid zero or negative tau
            tau_max = np.max(radial_distances_filtered) * 10
            lower_bounds.append(tau_min)
            upper_bounds.append(tau_max)

        bounds = (lower_bounds, upper_bounds)

        # Fit the sum of damped sinusoids to the radial medians
        try:
            popt_ds, _ = curve_fit(
                sum_damped_sinusoids,
                radial_distances_filtered,
                radial_medians_filtered,
                p0=initial_guess_ds,
                sigma=radial_stds_filtered,
                absolute_sigma=True,
                maxfev=10000,
                bounds=bounds
            )
            # Compute residuals
            residuals_ds = radial_medians_filtered - sum_damped_sinusoids(radial_distances_filtered, *popt_ds)
            residual_norm = np.linalg.norm(residuals_ds) / len(residuals_ds)

            print(f"Damped sinusoids fitting successful with N_sinusoids = {N_sinusoids}. Residual: {residual_norm}")

            if residual_norm < best_residual_norm:
                best_residual_norm = residual_norm
                best_popt_ds = popt_ds

            if residual_norm <= residual_threshold:
                fit_success = True
                break  # Acceptable fit achieved
            else:
                N_sinusoids += 1  # Increase the number of sinusoids
        except RuntimeError as e:
            print(f"Damped sinusoids curve fitting failed with N_sinusoids = {N_sinusoids}: {e}")
            N_sinusoids += 1

    if not fit_success and best_popt_ds is not None:
        # Use the best fit obtained
        popt_ds = best_popt_ds
        print(f"Using best fit with residual {best_residual_norm} and N_sinusoids = {N_sinusoids - 1}")
    elif not fit_success:
        # No successful fit; use initial guesses
        popt_ds = initial_guess_ds
        print(f"Fitting did not converge within acceptable residual after trying up to {max_N_sinusoids} sinusoids.")

    # Evaluate the damped sinusoids model over the entire image
    background_ds = sum_damped_sinusoids(radii.flatten(), *popt_ds)
    background_ds = background_ds.reshape(image.shape)

    # Subtract the background from the original image
    corrected_image = image - background_ds

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

def init_worker(h5_file_path):
    global h5_file
    h5_file = h5py.File(h5_file_path, 'r')

def process_image_worker(image_index):
    try:
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

def remove_background(h5_file_path, new_h5_file_path, selected_indices=None):
    # Open the original HDF5 file to get the number of images
    with h5py.File(h5_file_path, 'r') as h5_file:
        num_images = h5_file['/entry/data/images'].shape[0]
        print(f"The dataset contains {num_images} images.")

    # If selected_indices is None, process all images
    if selected_indices is None:
        selected_indices = range(num_images)
    else:
        # Ensure selected_indices are within the valid range
        selected_indices = [idx for idx in selected_indices if 0 <= idx < num_images]

    print(f"Processing {len(selected_indices)} selected images.")

    # Determine chunk size
    chunk_size = min(1000, num_images)

    if selected_indices:
        chunk_size = min(chunk_size, len(selected_indices))

    print(f"Chunk size set to: {chunk_size}")

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
            chunks=(chunk_size, 1024, 1024),
            compression="gzip"
        )

        # Prepare arguments for multiprocessing
        args_list = [(image_index) for image_index in selected_indices]

        # Use multiprocessing Pool with initializer
        with multiprocessing.Pool(initializer=init_worker, initargs=(h5_file_path,)) as pool:
            # Process images in parallel
            for image_index, corrected_image in pool.imap_unordered(process_image_worker, args_list):
                if corrected_image is not None:
                    # Write the corrected image to the new dataset
                    corrected_images_dataset[image_index, :, :] = corrected_image
                else:
                    print(f"Image {image_index} was not processed due to an error.")

    print("Processing completed.")

if __name__ == '__main__':

    # Paths to the files
    h5_file_path = '/home/buster/UOX1/UOX1_background/deiced_UOX1_min_15_peak.h5'
    new_h5_file_path = '/home/buster/UOX1/deiced_UOX1_min_15_peak_corrected_ds_v5.h5'

    # Process all images
    selected_indices = [70, 700, 1500, 1800]  # Set to None to process all images

    # Call the main processing function
    remove_background(h5_file_path, new_h5_file_path, selected_indices=selected_indices)