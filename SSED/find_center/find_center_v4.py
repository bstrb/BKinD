import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.optimize import minimize

def load_mask(mask_file_path):
    with h5py.File(mask_file_path, 'r') as h5_file:
        mask = h5_file['/mask'][()]
        print(f"Loaded mask with shape {mask.shape} and dtype {mask.dtype}", flush=True)
    return mask

def get_radial_profile_slice(data, center, angle_range, mask, max_radius, radial_bins):
    y, x = np.indices(data.shape)
    x = x - center[0]
    y = y - center[1]
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)  # Angle in radians

    # Adjust angles to be in [0, 2*pi]
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    # Define the angular slice
    angle_min, angle_max = angle_range
    # Handle angle wrapping
    if angle_min < 0:
        angle_min += 2 * np.pi
    if angle_max > 2 * np.pi:
        angle_max -= 2 * np.pi

    if angle_min <= angle_max:
        angular_mask = (theta >= angle_min) & (theta <= angle_max)
    else:
        angular_mask = (theta >= angle_min) | (theta <= angle_max)

    # Combine masks
    combined_mask = angular_mask & (mask > 0)

    # Exclude masked points
    valid = combined_mask

    # Get data within the slice
    r_valid = r[valid]
    data_valid = data[valid]

    if len(r_valid) == 0:
        # No valid data in this slice
        return np.full(len(radial_bins) - 1, np.nan)

    # Bin data by radius
    radial_median = np.zeros(len(radial_bins) - 1)

    # Digitize radii into bins
    r_indices = np.digitize(r_valid, radial_bins) - 1  # Subtract 1 to get 0-based indices

    for i in range(len(radial_bins) - 1):
        bin_mask = r_indices == i
        if np.any(bin_mask):
            radial_median[i] = np.median(data_valid[bin_mask])
        else:
            radial_median[i] = np.nan  # Handle empty bins

    return radial_median

def process_image(image, image_index, mask, plot=False, convergence_threshold=0.5, max_iterations=10, verbose=True):
    try:
        if verbose:
            print(f"Processing image {image_index}...", flush=True)

        # Apply median filter to reduce noise
        filtered_image = median_filter(image, size=3)

        # Exclude high-intensity pixels (e.g., top 1% of intensities)
        nonzero_pixels = filtered_image[mask > 0]
        if nonzero_pixels.size == 0:
            print(f"No non-zero pixels in image {image_index} after filtering.", flush=True)
            return None, None

        intensity_threshold = np.percentile(nonzero_pixels, 99)
        filtered_image[filtered_image > intensity_threshold] = intensity_threshold

        # Initial guess for the center (image center)
        center = np.array([filtered_image.shape[1] / 2, filtered_image.shape[0] / 2])
        center_shift = np.inf
        iteration = 0

        # Parameters for radial profile
        max_radius = int(np.hypot(filtered_image.shape[1], filtered_image.shape[0]) / 2)
        radial_bins = np.linspace(0, max_radius, max_radius + 1)

        # Define the angle range for the slice (in radians)
        slice_angle = np.deg2rad(20)  # 20 degrees
        num_slices = 8  # Number of slices to use (adjustable)

        # Angles for the slices
        slice_angles = np.linspace(0, 2 * np.pi, num_slices, endpoint=False)

        while center_shift > convergence_threshold and iteration < max_iterations:
            previous_center = center.copy()

            # Objective function
            def objective_function(center):
                total_difference = 0
                for angle in slice_angles:
                    angle_min = angle - slice_angle / 2
                    angle_max = angle + slice_angle / 2

                    # Get radial profile for the slice
                    radial_profile1 = get_radial_profile_slice(
                        filtered_image, center, (angle_min, angle_max), mask, max_radius, radial_bins
                    )

                    # Get radial profile for the opposite slice
                    opposite_angle = (angle + np.pi) % (2 * np.pi)
                    angle_min_opp = opposite_angle - slice_angle / 2
                    angle_max_opp = opposite_angle + slice_angle / 2

                    radial_profile2 = get_radial_profile_slice(
                        filtered_image, center, (angle_min_opp, angle_max_opp), mask, max_radius, radial_bins
                    )

                    # Exclude NaN values (from empty bins)
                    valid_bins = (~np.isnan(radial_profile1)) & (~np.isnan(radial_profile2))

                    if np.sum(valid_bins) == 0:
                        continue  # Skip if no valid bins

                    # Smooth the radial profiles
                    radial_profile1_smooth = gaussian_filter1d(radial_profile1[valid_bins], sigma=2)
                    radial_profile2_smooth = gaussian_filter1d(radial_profile2[valid_bins], sigma=2)

                    # Compute the difference
                    difference = radial_profile1_smooth - radial_profile2_smooth

                    # Sum of squared differences
                    total_difference += np.sum(difference ** 2)

                # Debugging output for the objective function value
                if verbose:
                    print(f"Objective function value: {total_difference:.4f}", flush=True)
                return total_difference

            # Optimization
            result = minimize(
                objective_function,
                center,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-2, 'fatol': 1e-2, 'disp': verbose}
            )

            center = result.x
            center_shift = np.linalg.norm(center - previous_center)
            iteration += 1
            if verbose:
                print(f"Iteration {iteration}: Computed center at (x={center[0]:.2f}, y={center[1]:.2f}), shift={center_shift:.4f}", flush=True)

            if center_shift <= convergence_threshold:
                if verbose:
                    print(f"Convergence reached after {iteration} iterations.", flush=True)
                break

        if plot:
            # Plot the filtered image with the computed center
            plt.figure(figsize=(8, 8))
            plt.imshow(filtered_image * mask, cmap='gray', origin='lower')
            plt.scatter(center[0], center[1], color='red', marker='x', s=100, label='Computed Center')
            plt.title(f"Image with Computed Center (Index {image_index})")
            plt.legend()
            plt.show()

        # Return the computed center coordinates
        return center[0], center[1]
    except Exception as e:
        print(f"Error processing image {image_index}: {e}", flush=True)
        return None, None

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
                dtype='float64'
            )

            # Process images
            with tqdm(total=total_images, desc='Processing images') as pbar:
                for idx_in_dataset, image_index in enumerate(selected_indices):
                    try:
                        image = images_dataset[image_index, :, :].astype(np.float32)
                        beam_center = process_image(image, image_index, mask, plot=plot, verbose=verbose)

                        if beam_center is None:
                            print(f"Skipping image {image_index} due to processing error.", flush=True)
                            continue

                        beam_centers_dataset[idx_in_dataset, :] = beam_center
                    except Exception as e:
                        print(f"Error processing image {image_index}: {e}", flush=True)
                        continue

                    pbar.update(1)

        print("Processing completed.", flush=True)

# Example usage
if __name__ == '__main__':
    h5_file_path = '/home/buster/UOX1/UOX1_min_50/UOX1_min_50_peak.h5'
    new_h5_file_path = '/home/buster/UOX1/UOX1_min_50/UOX1_min_50_peak_centerfound.h5'
    mask_path = '/home/buster/mask/pxmask.h5'
    selected_indices = [1800]  # Change to desired indices

    # Set plot=True to visualize the results
    find_center(h5_file_path, new_h5_file_path, mask_path, selected_indices=selected_indices, plot=True, verbose=True)
