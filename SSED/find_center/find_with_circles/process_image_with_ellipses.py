import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.measure import CircleModel, ransac
from sklearn.cluster import DBSCAN
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_mask(mask_file_path: str) -> np.ndarray:
    """
    Loads the mask from the specified HDF5 file.

    Parameters:
        mask_file_path (str): Path to the mask HDF5 file.

    Returns:
        np.ndarray: The mask array.
    """
    with h5py.File(mask_file_path, 'r') as h5_file:
        mask = h5_file['/mask'][()]
    return mask

def dbscan_subsample(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 20.0,
    min_samples: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsamples points using DBSCAN clustering by selecting the cluster centroids.

    Parameters:
        x (np.ndarray): X-coordinates of the points.
        y (np.ndarray): Y-coordinates of the points.
        eps (float): The maximum distance between two samples for them to be considered in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Subsampled X and Y coordinates.
    """
    points = np.column_stack((x, y))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    x_sub, y_sub = [], []

    for label in unique_labels:
        if label == -1:
            # Noise points are skipped
            continue
        cluster_points = points[labels == label]
        centroid = cluster_points.mean(axis=0)
        x_sub.append(centroid[0])
        y_sub.append(centroid[1])

    return np.array(x_sub), np.array(y_sub)

def fit_circle(
    x: np.ndarray,
    y: np.ndarray,
    residual_threshold: float = 1.5,
    min_samples: int = 3,
    max_trials: int = 1000
) -> Optional[Tuple[float, float, float]]:
    """
    Fits a circle to the given points using RANSAC for robustness.

    Parameters:
        x (np.ndarray): X-coordinates of the points.
        y (np.ndarray): Y-coordinates of the points.
        residual_threshold (float): Maximum distance for a data point to be classified as an inlier.
        min_samples (int): Minimum number of data points to fit the model.
        max_trials (int): Maximum number of iterations for random sample selection.

    Returns:
        Optional[Tuple[float, float, float]]: (xc, yc, radius) of the fitted circle, or None if fitting fails.
    """
    if len(x) < 3:
        logging.warning("Not enough points to fit a circle.")
        return None

    points = np.column_stack((x, y))
    model = CircleModel()
    try:
        model_robust, inliers = ransac(
            points,
            model,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials
        )
        if model_robust is None:
            logging.error("RANSAC failed to find a valid circle model.")
            return None
        xc, yc, radius = model_robust.params
        logging.info(f"Fitted circle: Center=({xc:.2f}, {yc:.2f}), Radius={radius:.2f}")
        return (xc, yc, radius)
    except Exception as e:
        logging.error(f"Exception during circle fitting: {e}")
        return None

def process_image(
    image: np.ndarray,
    mask: np.ndarray,
    plot: bool = False
) -> Optional[Tuple[float, float]]:
    """
    Processes the image to find the beam center by sampling points in valid intensity intervals and fitting circles.

    Parameters:
        image (np.ndarray): The 2D image array.
        mask (np.ndarray): The mask array to apply to the image.
        plot (bool): Whether to display plots for debugging.

    Returns:
        Optional[Tuple[float, float]]: (X, Y) coordinates of the beam center, or None if not found.
    """
    # Apply mask
    masked_image = np.where(mask, image, np.nan)

    # Normalize the image based on unmasked pixels
    unmasked_pixels = masked_image[~np.isnan(masked_image)]
    if unmasked_pixels.size == 0:
        logging.error("All pixels are masked. Cannot process the image.")
        return None

    min_intensity = unmasked_pixels.min()
    max_intensity = unmasked_pixels.max()
    if max_intensity == min_intensity:
        logging.error("Image has constant intensity. Cannot process the image.")
        return None

    norm_image = (masked_image - min_intensity) / (max_intensity - min_intensity)

    # Define intensity bins
    intensity_levels = [0.3, 0.5, 0.7, 0.9]  # Adjust as needed
    bin_width = 0.05  # Width around each intensity level

    centers = []

    for level in intensity_levels:
        lower = max(0.0, level - bin_width / 2)
        upper = min(1.0, level + bin_width / 2)
        bin_mask = (norm_image >= lower) & (norm_image < upper)
        y_coords, x_coords = np.where(bin_mask)

        if len(x_coords) == 0:
            logging.info(f"No pixels found in intensity bin {level:.2f} ± {bin_width / 2:.2f}")
            continue

        # Subsample points using DBSCAN
        x_sub, y_sub = dbscan_subsample(x_coords, y_coords, eps=20.0, min_samples=3)
        logging.info(f"Intensity bin {level:.2f} ± {bin_width / 2:.2f}: {len(x_coords)} points before subsampling, {len(x_sub)} after DBSCAN")

        if len(x_sub) == 0:
            logging.info(f"No clusters found in intensity bin {level:.2f} ± {bin_width / 2:.2f}")
            continue

        # Fit circle to the subsampled points
        circle = fit_circle(x_sub, y_sub)
        if circle is not None:
            xc, yc, radius = circle
            centers.append((xc, yc))
            if plot:
                plt.figure(figsize=(6, 6))
                plt.imshow(norm_image, cmap='gray', origin='lower')
                plt.scatter(x_sub, y_sub, c='blue', label='DBSCAN Subsampled Points')
                plt.scatter(xc, yc, c='red', label='Fitted Circle Center')
                circle_patch = plt.Circle((xc, yc), radius, color='green', fill=False, linewidth=2, label='Fitted Circle')
                plt.gca().add_patch(circle_patch)
                plt.legend()
                plt.title(f'Intensity Bin {level:.2f} ± {bin_width / 2:.2f}')
                plt.xlabel('X Pixels')
                plt.ylabel('Y Pixels')
                plt.axis('equal')
                plt.show()

    if not centers:
        logging.warning("No circles were fitted in any intensity bins. Beam center could not be determined.")
        return None

    # Aggregate the centers to determine the beam center
    centers_array = np.array(centers)
    beam_center_x = np.median(centers_array[:, 0])
    beam_center_y = np.median(centers_array[:, 1])
    logging.info(f"Aggregated Beam Center: ({beam_center_x:.2f}, {beam_center_y:.2f})")

    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(norm_image, cmap='gray', origin='lower')
        plt.scatter(centers_array[:, 0], centers_array[:, 1], c='yellow', label='Fitted Circle Centers')
        plt.scatter(beam_center_x, beam_center_y, c='red', marker='x', s=100, label='Beam Center')
        plt.legend()
        plt.title('Aggregated Beam Center')
        plt.xlabel('X Pixels')
        plt.ylabel('Y Pixels')
        plt.axis('equal')
        plt.show()

    return (beam_center_x, beam_center_y)

# Example usage
if __name__ == "__main__":
    # Paths to the HDF5 image and mask files
    h5_file_path = '/home/buster/UOX1/UOX1_min_10/UOX1_min_10.h5'
    mask_path = '/home/buster/mask/pxmask.h5'

    # Load the mask
    mask = load_mask(mask_path)

    # Specify the image index to process (e.g., 1800)
    image_index = 1800

    # Open the HDF5 file and load the specified image
    with h5py.File(h5_file_path, 'r') as h5_file:
        images_dataset = h5_file['/entry/data/images']
        if image_index < 0 or image_index >= images_dataset.shape[0]:
            logging.error(f"Image index {image_index} is out of bounds.")
        else:
            image = images_dataset[image_index, :, :].astype(np.float32)
            logging.info(f"Loaded image {image_index} with shape {image.shape}")

            # Find the beam center
            beam_center = find_beam_center(image, mask, plot=True)

            if beam_center:
                logging.info(f"Beam center for image {image_index}: {beam_center}")
            else:
                logging.warning(f"Beam center for image {image_index} could not be determined.")
