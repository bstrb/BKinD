import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import CircleModel, ransac
from sklearn.cluster import DBSCAN
import os
from typing import Tuple, List, Optional

def fit_circle_ransac(
    x: np.ndarray,
    y: np.ndarray,
    residual_threshold: float = 1.5,
    min_samples: int = 5,
    max_trials: int = 1000
) -> Tuple[float, float, float, np.ndarray]:
    """
    Fit a circle robustly using RANSAC.

    Parameters:
        x (np.ndarray): X-coordinates of the points.
        y (np.ndarray): Y-coordinates of the points.
        residual_threshold (float): Maximum distance for a data point to be classified as an inlier.
        min_samples (int): Minimum number of data points to fit a model.
        max_trials (int): Maximum number of iterations for random sample selection.

    Returns:
        Tuple containing:
            - xc (float): X-coordinate of the circle center.
            - yc (float): Y-coordinate of the circle center.
            - R (float): Radius of the circle.
            - inliers (np.ndarray): Boolean array indicating inliers.
    """
    points = np.column_stack((x, y))
    try:
        model_robust, inliers = ransac(
            points, 
            CircleModel, 
            min_samples=min_samples, 
            residual_threshold=residual_threshold, 
            max_trials=max_trials
        )
        if model_robust is None:
            raise ValueError("RANSAC failed to find a valid model.")
        xc, yc, R = model_robust.params
        return xc, yc, R, inliers
    except Exception as e:
        raise RuntimeError(f"RANSAC fitting failed: {e}")

def dbscan_subsample(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 10.0,
    min_samples: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample points using DBSCAN clustering by selecting the cluster centroids.

    Parameters:
        x (np.ndarray): X-coordinates of the points.
        y (np.ndarray): Y-coordinates of the points.
        eps (float): The maximum distance between two samples for them to be considered in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        Tuple containing:
            - x_sub (np.ndarray): Subsampled X-coordinates.
            - y_sub (np.ndarray): Subsampled Y-coordinates.
    """
    points = np.column_stack((x, y))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    x_sub, y_sub = [], []

    for label in unique_labels:
        if label == -1:
            # Noise points are skipped or handled separately
            continue
        cluster_points = points[labels == label]
        centroid = cluster_points.mean(axis=0)
        x_sub.append(centroid[0])
        y_sub.append(centroid[1])

    return np.array(x_sub), np.array(y_sub)

def process_image(
    image: np.ndarray, 
    image_index: int, 
    mask: Optional[np.ndarray] = None, 
    plot: bool = True, 
    verbose: bool = True,
    intensity_levels: List[float] = [0.3, 0.5, 0.7, 0.9],  # Multiple intensity levels
    bin_widths: List[float] = [0.05, 0.05, 0.05, 0.05],    # Corresponding bin widths
    residual_threshold: float = 1.0,
    min_samples: int = 10,
    max_trials: int = 3000,
    eps: float = 15.0,            # DBSCAN parameter
    dbscan_min_samples: int = 5,  # DBSCAN parameter
    center_history: Optional[List[Tuple[float, float]]] = None,
    max_center_diff: float = 5.0,
    moving_mean_window: int = 5
) -> Tuple[Optional[float], Optional[float], List[Tuple[float, float]]]:
    """
    Processes the image to fit circles at specified intensity levels using DBSCAN for subsampling 
    and applies moving mean if center shifts exceed a threshold.

    Parameters:
        image (np.ndarray): 2D array representing the image.
        image_index (int): Index of the image being processed.
        mask (Optional[np.ndarray]): Optional boolean mask array of the same shape as image.
        plot (bool): Whether to display plots.
        verbose (bool): Whether to print detailed processing information.
        intensity_levels (List[float]): List of intensity levels to fit circles.
        bin_widths (List[float]): List of bin widths corresponding to intensity levels.
        residual_threshold (float): RANSAC residual threshold.
        min_samples (int): RANSAC minimum samples.
        max_trials (int): RANSAC maximum trials.
        eps (float): DBSCAN epsilon parameter.
        dbscan_min_samples (int): DBSCAN min_samples parameter.
        center_history (Optional[List[Tuple[float, float]]]): List of previous centers [(x1, y1), (x2, y2), ...].
        max_center_diff (float): Maximum allowed difference in pixels before applying moving mean.
        moving_mean_window (int): Number of previous centers to include in moving mean.

    Returns:
        Tuple containing:
            - adjusted_center_x (Optional[float]): X-coordinate of the adjusted center.
            - adjusted_center_y (Optional[float]): Y-coordinate of the adjusted center.
            - center_history (List[Tuple[float, float]]): Updated list of previous centers.
    """
    if center_history is None:
        center_history = []

    if verbose:
        print(f"\nProcessing image {image_index}...", flush=True)

    # Validate and apply mask
    if mask is not None:
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.shape != image.shape:
            raise ValueError("Mask shape does not match image shape.")
    else:
        mask = np.ones_like(image, dtype=bool)

    # Extract unmasked pixels
    unmasked_pixels = image[mask]
    if unmasked_pixels.size == 0:
        raise ValueError("No unmasked pixels in the image.")

    # Normalize the image based on unmasked pixels
    min_intensity = unmasked_pixels.min()
    max_intensity = unmasked_pixels.max()
    if max_intensity == min_intensity:
        raise ValueError("Image has constant intensity in unmasked region.")

    norm_image = np.full_like(image, np.nan, dtype=np.float64)
    norm_image[mask] = (image[mask] - min_intensity) / (max_intensity - min_intensity)

    # Extract valid pixels after normalization
    valid_pixels = norm_image[~np.isnan(norm_image)]
    if valid_pixels.size == 0:
        raise ValueError("No valid pixels after normalization.")

    # Ensure intensity_levels and bin_widths have the same length
    if len(intensity_levels) != len(bin_widths):
        raise ValueError("Intensity levels and bin widths must have the same length.")

    centers, radii, intensity_levels_used, x_positions, y_positions = [], [], [], [], []

    for intensity_level, bin_width in zip(intensity_levels, bin_widths):
        # Define the intensity bin range
        level_min = max(0.0, intensity_level - bin_width / 2)
        level_max = min(1.0, intensity_level + bin_width / 2)

        # Get indices of unmasked pixels within the intensity bin
        bin_mask = (norm_image >= level_min) & (norm_image < level_max)
        y_coords, x_coords = np.where(bin_mask)

        if len(x_coords) == 0:
            if verbose:
                print(f"No pixels found in intensity bin {intensity_level:.2f} ± {bin_width / 2:.2f}", flush=True)
            continue

        # Subsample using DBSCAN
        x_sub, y_sub = dbscan_subsample(x_coords, y_coords, eps=eps, min_samples=dbscan_min_samples)
        if len(x_sub) == 0:
            if verbose:
                print(f"No clusters found in intensity bin {intensity_level:.2f} ± {bin_width / 2:.2f}", flush=True)
            continue

        # Determine appropriate min_samples for RANSAC
        actual_min_samples = min_samples if len(x_sub) >= min_samples else max(3, len(x_sub))
        if len(x_sub) < 3:
            if verbose:
                print(f"Not enough points ({len(x_sub)}) after subsampling for RANSAC in bin {intensity_level:.2f} ± {bin_width / 2:.2f}", flush=True)
            continue

        # Fit the circle robustly using RANSAC
        try:
            xc, yc, R, inliers = fit_circle_ransac(
                x_sub, 
                y_sub,
                residual_threshold=residual_threshold,
                min_samples=actual_min_samples,
                max_trials=max_trials
            )
        except Exception as e:
            if verbose:
                print(f"Could not fit circle in bin {intensity_level:.2f} ± {bin_width / 2:.2f}: {e}", flush=True)
            continue

        # Extract inlier points
        x_inliers = x_sub[inliers]
        y_inliers = y_sub[inliers]

        # Validate radius
        min_radius, max_radius = 10, 1000  # Adjust based on expected data
        if not (min_radius < R < max_radius):
            if verbose:
                print(f"Rejected circle with radius {R:.2f} outside [{min_radius}, {max_radius}].", flush=True)
            continue

        centers.append((xc, yc))
        radii.append(R)
        intensity_levels_used.append(intensity_level)
        x_positions.append(xc)
        y_positions.append(yc)

        if verbose:
            print(f"Fitted circle at intensity level {intensity_level:.2f} ± {bin_width / 2:.2f}: "
                    f"center=({xc:.2f}, {yc:.2f}), radius={R:.2f}", flush=True)

        if plot:
            # Plot the subsampled points and fitted circle
            plt.figure(figsize=(6, 6))
            plt.title(f'Intensity bin {intensity_level:.2f} ± {bin_width / 2:.2f}')
            plt.imshow(image * mask, cmap='gray', origin='lower')
            plt.scatter(x_sub, y_sub, c='blue', s=10, alpha=0.5, label='DBSCAN Subsampled Points')
            plt.scatter(x_inliers, y_inliers, c='green', s=20, label='RANSAC Inliers')
            circle_patch = plt.Circle((xc, yc), R, color='red', fill=False, linewidth=2, label='Fitted Circle')
            plt.gca().add_patch(circle_patch)
            plt.scatter(xc, yc, c='yellow', marker='x', s=100, label='Circle Center')
            plt.legend(loc='upper right', fontsize='small')
            plt.axis('equal')
            plt.xlabel('X Pixels')
            plt.ylabel('Y Pixels')
            plt.tight_layout()
            plt.show()

    if not centers:
        if verbose:
            print("No circles were fitted for the image.", flush=True)
        return None, None, center_history

    # Compute the combined center as the median of the centers
    centers_array = np.array(centers)
    combined_center_x, combined_center_y = np.median(centers_array, axis=0)

    if verbose:
        print(f"Combined center: ({combined_center_x:.2f}, {combined_center_y:.2f})", flush=True)

    # Compare with previous center and apply moving mean if necessary
    if center_history:
        prev_center_x, prev_center_y = center_history[-1]
        center_diff = np.hypot(combined_center_x - prev_center_x, combined_center_y - prev_center_y)
        if center_diff > max_center_diff:
            if verbose:
                print(f"Center shift {center_diff:.2f} exceeds {max_center_diff} pixels. Applying moving mean.", flush=True)
            # Determine the window for moving mean
            window = min(len(center_history), moving_mean_window)
            # Get the last 'window' centers
            recent_centers = center_history[-window:]
            recent_centers.append((combined_center_x, combined_center_y))
            recent_centers_array = np.array(recent_centers)
            adjusted_center_x = np.mean(recent_centers_array[:, 0])
            adjusted_center_y = np.mean(recent_centers_array[:, 1])
            if verbose:
                print(f"Adjusted center using moving mean: ({adjusted_center_x:.2f}, {adjusted_center_y:.2f})", flush=True)
        else:
            adjusted_center_x, adjusted_center_y = combined_center_x, combined_center_y
    else:
        adjusted_center_x, adjusted_center_y = combined_center_x, combined_center_y

    # Update the center history
    center_history.append((adjusted_center_x, adjusted_center_y))
    if len(center_history) > moving_mean_window:
        center_history.pop(0)

    if plot:
        # Plot all fitted circles and the adjusted center
        plt.figure(figsize=(12, 6))

        # Left subplot: image with fitted circles
        plt.subplot(1, 2, 1)
        plt.imshow(image * mask, cmap='gray', origin='lower')
        ax = plt.gca()

        for (xc, yc), R in zip(centers, radii):
            circle_patch = plt.Circle((xc, yc), R, color='cyan', fill=False, linewidth=2)
            ax.add_patch(circle_patch)
            plt.scatter(xc, yc, color='magenta', marker='x', s=50)

        plt.scatter(adjusted_center_x, adjusted_center_y, color='yellow', marker='o', 
                    s=100, label='Adjusted Center')
        plt.title("Image with Fitted Circles")
        plt.legend(loc='upper right')

        # Right subplot: center positions vs intensity levels
        plt.subplot(1, 2, 2)
        plt.scatter(intensity_levels_used, x_positions, label='X Position', color='green', marker='o')
        plt.scatter(intensity_levels_used, y_positions, label='Y Position', color='magenta', marker='o')
        plt.xlabel('Intensity Level')
        plt.ylabel('Position (pixels)')
        plt.title('Center Positions vs. Intensity Levels')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return adjusted_center_x, adjusted_center_y, center_history

def generate_synthetic_image(
    image_size: Tuple[int, int],
    true_centers: List[Tuple[float, float]],
    true_radii: List[float],
    frame_idx: int,
    noise_std: float = 0.05,
    num_outliers: int = 500,
    max_shift: float = 50.0  # Maximum total shift to keep the circle within frame
) -> np.ndarray:
    """
    Generates a synthetic image with shifted circles, Gaussian noise, and random outliers.

    Parameters:
        image_size (Tuple[int, int]): Size of the image (rows, cols).
        true_centers (List[Tuple[float, float]]): List of true circle centers.
        true_radii (List[float]): List of circle radii.
        frame_idx (int): Current frame index for shifting the centers.
        noise_std (float): Standard deviation of the Gaussian noise.
        num_outliers (int): Number of random high-intensity outliers.
        max_shift (float): Maximum total shift to prevent circle from moving out of frame.

    Returns:
        np.ndarray: Generated synthetic image.
    """
    image = np.zeros(image_size)
    # Calculate shift per frame but cap the total shift
    total_shift = min(frame_idx * 0.5, max_shift)
    shifted_centers = [(xc + total_shift, yc + total_shift) for xc, yc in true_centers]

    for (xc, yc), R in zip(shifted_centers, true_radii):
        Y, X = np.ogrid[:image_size[0], :image_size[1]]
        dist_from_center = np.sqrt((X - xc)**2 + (Y - yc)**2)
        mask_circle = (dist_from_center >= R - 2) & (dist_from_center <= R + 2)
        image[mask_circle] = 1.0  # Set circle intensity

    # Add Gaussian noise
    np.random.seed(frame_idx)  # Different seed for each frame for reproducibility
    noise = np.random.normal(0, noise_std, image_size)
    image += noise

    # Add random outliers
    outlier_indices = (
        np.random.randint(0, image_size[0], num_outliers),
        np.random.randint(0, image_size[1], num_outliers)
    )
    image[outlier_indices] = 1.0  # Set outliers to high intensity

    # Clip the image to [0, 1] range
    image = np.clip(image, 0, 1)

    return image

def main():
    # Parameters
    num_frames = 2000  # Adjusted to include frames up to 1800
    image_size = (500, 500)
    true_centers = [(250, 250)]
    true_radii = [100]

    # Initialize center history
    center_history: List[Tuple[float, float]] = []

    # Process each frame
    for frame_idx in range(1, num_frames + 1):
        # Generate synthetic image
        image = generate_synthetic_image(
            image_size=image_size,
            true_centers=true_centers,
            true_radii=true_radii,
            frame_idx=frame_idx,
            noise_std=0.05,
            num_outliers=500,
            max_shift=50.0  # Ensure the circle stays within frame
        )

        # Create a mask (all True in this example)
        mask = np.ones_like(image, dtype=bool)

        # Determine if plotting is needed for specific frames
        plot_flag = frame_idx in [1200, 1800]  # Example: visualize problematic frames

        # Process the image
        combined_center_x, combined_center_y, center_history = process_image(
            image=image, 
            image_index=frame_idx, 
            mask=mask, 
            plot=plot_flag,  # Enable plotting for frames 1200 and 1800
            verbose=True,
            intensity_levels=[0.5, 0.7, 0.9],    # Multiple intensity levels for more rings
            bin_widths=[0.05, 0.05, 0.05],      # Corresponding bin widths
            residual_threshold=1.0,
            min_samples=10,
            max_trials=3000,
            eps=15.0,                            # DBSCAN epsilon parameter
            dbscan_min_samples=5,                # DBSCAN min_samples parameter
            center_history=center_history,
            max_center_diff=5.0,
            moving_mean_window=5
        )

        if combined_center_x is not None and combined_center_y is not None:
            print(f"Frame {frame_idx}: Computed Combined Center: x={combined_center_x:.2f}, y={combined_center_y:.2f}")
        else:
            print(f"Frame {frame_idx}: Center could not be determined.")

if __name__ == "__main__":
    main()
