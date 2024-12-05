import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d, zoom
from scipy.optimize import minimize

def get_radial_profile_slice(data, center, angle_range, mask, radial_bins):
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
    angle_min %= 2 * np.pi
    angle_max %= 2 * np.pi

    if angle_min <= angle_max:
        angular_mask = (theta >= angle_min) & (theta <= angle_max)
    else:
        angular_mask = (theta >= angle_min) | (theta <= angle_max)

    # Combine masks: angular slice, provided mask, and non-NaN data
    combined_mask = angular_mask & (mask > 0) & (~np.isnan(data))

    # Extract valid pixel coordinates relative to center
    x_valid = x[combined_mask]
    y_valid = y[combined_mask]
    r_valid = r[combined_mask]
    data_valid = data[combined_mask]

    # Compute symmetric coordinates about the center
    symmetric_x = -x_valid
    symmetric_y = -y_valid

    # Convert back to absolute indices
    symmetric_x_idx = np.round(symmetric_x + center[0]).astype(int)
    symmetric_y_idx = np.round(symmetric_y + center[1]).astype(int)

    # Ensure symmetric indices are within image boundaries
    within_bounds = (
        (symmetric_x_idx >= 0) & (symmetric_x_idx < data.shape[1]) &
        (symmetric_y_idx >= 0) & (symmetric_y_idx < data.shape[0])
    )

    # Filter out pixels whose symmetric counterparts are out of bounds
    x_valid = x_valid[within_bounds]
    y_valid = y_valid[within_bounds]
    r_valid = r_valid[within_bounds]
    data_valid = data_valid[within_bounds]
    symmetric_x_idx = symmetric_x_idx[within_bounds]
    symmetric_y_idx = symmetric_y_idx[within_bounds]

    # Check if symmetric counterparts are valid (mask > 0 and not NaN)
    mask_sym = (mask[symmetric_y_idx, symmetric_x_idx] > 0) & (~np.isnan(data[symmetric_y_idx, symmetric_x_idx]))

    # Further filter to include only symmetric valid pixels
    valid_final = mask_sym

    # Apply the final mask
    r_valid = r_valid[valid_final]
    data_valid = data_valid[valid_final]

    if len(r_valid) == 0:
        # No valid data in this slice after symmetry check
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

def process_image(image, image_index, mask, plot=False, verbose=True):
    try:
        if verbose:
            print(f"Processing image {image_index}...\n", flush=True)


        # Apply median filter to reduce noise
        median_filter_size = 5
        if median_filter_size != 1:
            filtered_image = median_filter(image, size = median_filter_size)
        else:
            filtered_image = image

        # Exclude high-intensity pixels (e.g., top 1% of intensities)
        top_intensity_exc = 0
        if top_intensity_exc != 0:
            nonzero_pixels = filtered_image[mask > 0]
            if nonzero_pixels.size == 0:
                print(f"No non-zero pixels in image {image_index} after filtering.", flush=True)
                return None, None

            intensity_threshold = np.percentile(nonzero_pixels, 100 - top_intensity_exc)  # Adjusted to 99th percentile
            filtered_image[filtered_image > intensity_threshold] = intensity_threshold
        else:
            filtered_image = filtered_image

        # Downsample the image and mask (optional)
        downsample_factor = 0.5  # Adjust as needed
        
        if downsample_factor != 1:
            downsampled_image = zoom(filtered_image, downsample_factor)
            downsampled_mask = zoom(mask, downsample_factor, order=0)
        else:
            downsampled_image = filtered_image
            downsampled_mask = mask


        center_initial = [downsampled_image.shape[1] / 2, downsampled_image.shape[0] / 2]  # Center of the downsampled image
        if verbose:
            real_center_initial =  [image.shape[1] / 2, image.shape[0] / 2]  # Center of the image
            print(f"Initial center: {real_center_initial}")

        # Parameters for radial profile
        max_radius = min(downsampled_image.shape[1] / 2, downsampled_image.shape[0] / 2)
        radial_bins = np.linspace(0, max_radius, 101)  # 50 bins

        # Define the angle range for the slice (in radians)
        num_slices = 3  # You can change this as needed
        slice_angle = np.pi / num_slices  # Each slice covers π / num_slices radians

        # Angles for the slices (centered within their angular range)
        slice_angles = np.linspace(0, np.pi - slice_angle, num_slices) + (slice_angle / 2)
        if verbose:
            print("Slice centers (degrees):", np.degrees(slice_angles))

        # Objective function
        def objective_function(center):
            total_difference = 0
            for angle in slice_angles:
                # Calculate angular boundaries of the slice
                angle_min = (angle - slice_angle / 2) % (2 * np.pi)
                angle_max = (angle + slice_angle / 2) % (2 * np.pi)

                # Get radial profile for the slice
                radial_profile1 = get_radial_profile_slice(
                    downsampled_image, center, (angle_min, angle_max), downsampled_mask, radial_bins
                )

                # Get radial profile for the opposite slice
                opposite_angle = (angle + np.pi) % (2 * np.pi)
                angle_min_opp = (opposite_angle - slice_angle / 2) % (2 * np.pi)
                angle_max_opp = (opposite_angle + slice_angle / 2) % (2 * np.pi)

                radial_profile2 = get_radial_profile_slice(
                    downsampled_image, center, (angle_min_opp, angle_max_opp), downsampled_mask, radial_bins
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

            return total_difference

        # Optimization
        result = minimize(
            objective_function,
            center_initial,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-2, 'fatol': 1e-2, 'disp': verbose}
        )

        computed_center_downsampled = result.x
        computed_center = computed_center_downsampled / downsample_factor  # Adjust center back to original scale
        if verbose:
            print(f"Computed center at (x={computed_center[0]:.2f}, y={computed_center[1]:.2f})", flush=True)

        if plot:
            # Plot the original image with the computed center
            plt.figure(figsize=(8, 8))
            plt.imshow(image * mask, cmap='gray', origin='lower')
            plt.scatter(
                computed_center[0],
                computed_center[1],
                color='red',
                marker='x',
                s=100,
                label='Computed Center'
            )
            plt.title(f"Image with Computed Center (Index {image_index})")
            plt.legend()
            plt.axis('equal')
            plt.show()

            # Plot the downsampled image with angular slices
            plt.figure(figsize=(8, 8))
            plt.imshow(downsampled_image * downsampled_mask, cmap='gray', origin='lower')
            plt.scatter(
                computed_center_downsampled[0],
                computed_center_downsampled[1],
                color='red',
                marker='x',
                s=100,
                label='Computed Center'
            )
            for angle in slice_angles:
                angle_min = (angle - slice_angle / 2) % (2 * np.pi)
                angle_max = (angle + slice_angle / 2) % (2 * np.pi)
                angles = [angle_min, angle_max]
                for a in angles:
                    x_end = computed_center_downsampled[0] + max_radius * np.cos(a)
                    y_end = computed_center_downsampled[1] + max_radius * np.sin(a)
                    plt.plot(
                        [computed_center_downsampled[0], x_end],
                        [computed_center_downsampled[1], y_end],
                        color='yellow',
                        linestyle='--',
                        linewidth=1
                    )
            plt.title(f"Downsampled Image with Angular Slices (Index {image_index})")
            plt.legend()
            plt.axis('equal')
            plt.show()

            # Recompute radial profiles using the computed center for plotting
            radial_profiles = []
            opposite_profiles = []
            valid_slice_indices = []
            for angle in slice_angles:
                angle_min = (angle - slice_angle / 2) % (2 * np.pi)
                angle_max = (angle + slice_angle / 2) % (2 * np.pi)

                # Radial profile for the slice
                radial_profile1 = get_radial_profile_slice(
                    downsampled_image, computed_center_downsampled, (angle_min, angle_max), downsampled_mask, radial_bins
                )

                # Radial profile for the opposite slice
                opposite_angle = (angle + np.pi) % (2 * np.pi)
                angle_min_opp = (opposite_angle - slice_angle / 2) % (2 * np.pi)
                angle_max_opp = (opposite_angle + slice_angle / 2) % (2 * np.pi)

                radial_profile2 = get_radial_profile_slice(
                    downsampled_image, computed_center_downsampled, (angle_min_opp, angle_max_opp), downsampled_mask, radial_bins
                )

                # Exclude NaN values
                valid_bins = (~np.isnan(radial_profile1)) & (~np.isnan(radial_profile2))

                if np.sum(valid_bins) == 0:
                    continue  # Skip if no valid bins

                # Smooth the radial profiles
                radial_profile1_smooth = gaussian_filter1d(radial_profile1[valid_bins], sigma=2)
                radial_profile2_smooth = gaussian_filter1d(radial_profile2[valid_bins], sigma=2)

                radial_profiles.append(radial_profile1_smooth)
                opposite_profiles.append(radial_profile2_smooth)
                valid_slice_indices.append(np.where(valid_bins)[0])

            # Plot radial profiles
            plt.figure(figsize=(10, 6))
            for i, (profile1, profile2) in enumerate(zip(radial_profiles, opposite_profiles)):
                bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
                valid_indices = valid_slice_indices[i]
                plt.plot(bin_centers[valid_indices], profile1, label=f'Slice {i+1} +', alpha=0.7)
                plt.plot(bin_centers[valid_indices], profile2, label=f'Slice {i+1} -', alpha=0.7)
            plt.xlabel('Radius (pixels)')
            plt.ylabel('Median Intensity')
            plt.title(f'Radial Profiles for Image {image_index}')
            plt.legend(loc='upper right', fontsize='small', ncol=2)
            plt.grid(True)
            plt.show()

        # Return the computed center coordinates
        return computed_center[0], computed_center[1]
    except Exception as e:
        print(f"Error processing image {image_index}: {e}", flush=True)
        return None, None
