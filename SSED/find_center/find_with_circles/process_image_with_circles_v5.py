import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter

def fit_circle_least_squares(x, y):
    """
    Fit a circle to the given x, y coordinates using least squares.
    """
    def calc_R(c):
        xc, yc = c
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def residuals(c):
        Ri = calc_R(c)
        return Ri - Ri.mean()

    x_m, y_m = np.mean(x), np.mean(y)
    center_estimate = np.array([x_m, y_m])
    result = least_squares(residuals, center_estimate)
    center = result.x
    Ri = calc_R(center)
    R = Ri.mean()
    return center[0], center[1], R

def process_image(image, image_index, mask=None, plot=True, verbose=True):
    """
    Processes the image to fit circles at specified intensity levels.
    """
    try:
        if verbose:
            print(f"Processing image {image_index}...", flush=True)

        # Ensure mask is boolean and matches image dimensions
        if mask is None:
            mask = np.ones_like(image, dtype=bool)
        else:
            if mask.dtype != np.bool_:
                mask = mask > 0
            if mask.shape != image.shape:
                print("Mask shape does not match image shape.", flush=True)
                return None, None

        # Smooth the image to reduce noise
        smoothed_image = gaussian_filter(image, sigma=1)

        # Extract unmasked pixels
        unmasked_pixels = smoothed_image[mask]

        if unmasked_pixels.size == 0:
            print(f"No unmasked pixels in the image.", flush=True)
            return None, None

        # Normalize the image based on unmasked pixels
        min_intensity, max_intensity = unmasked_pixels.min(), unmasked_pixels.max()
        if max_intensity == min_intensity:
            print(f"Image has constant intensity in unmasked region.", flush=True)
            return None, None

        # Create a normalized image, setting masked regions to NaN
        norm_image = np.full_like(smoothed_image, np.nan, dtype=np.float64)
        norm_image[mask] = (smoothed_image[mask] - min_intensity) / (max_intensity - min_intensity)

        # Apply intensity thresholding to exclude diffraction spots
        norm_image[(norm_image >= 0.8) | (norm_image <= 0.1)] = np.nan

        # Calculate mean and standard deviation for intensity binning
        mean, std = np.nanmean(norm_image), np.nanstd(norm_image)

        # Define intensity levels for fitting circles
        intensity_levels = np.arange(mean, mean + 3*std, 0.0010)  # Smaller bin size for better resolution

        centers = []
        radii = []

        for intensity_level in intensity_levels:
            bin_mask = (norm_image >= intensity_level - 0.0002) & (norm_image < intensity_level + 0.0002)
            indices = np.argwhere(bin_mask)

            if len(indices) < 15:
                continue

            y_coords, x_coords = indices[:, 0], indices[:, 1]

            # Initial circle fitting
            xc, yc, R = fit_circle_least_squares(x_coords, y_coords)

            # Compute residuals and exclude outliers
            Ri = np.sqrt((x_coords - xc) ** 2 + (y_coords - yc) ** 2)
            residuals = Ri - R
            inlier_mask = np.abs(residuals) <= 2 * np.std(residuals)

            x_inliers, y_inliers = x_coords[inlier_mask], y_coords[inlier_mask]
            if len(x_inliers) < 15:
                continue

            # Refit circle with inliers only
            xc, yc, R = fit_circle_least_squares(x_inliers, y_inliers)

            centers.append((xc, yc))
            radii.append(R)

            if verbose:
                print(f"Fitted circle at intensity level {intensity_level:.4f}: "
                      f"center=({xc:.2f}, {yc:.2f}), radius={R:.2f}", flush=True)

        if not centers:
            print(f"No circles were fitted for the image.", flush=True)
            return None, None

        # Compute the combined center as the mean of the centers
        centers_array = np.array(centers)
        combined_center_x, combined_center_y = centers_array.mean(axis=0)

        if verbose:
            print(f"Combined center: ({combined_center_x:.2f}, {combined_center_y:.2f})", flush=True)

        if plot:
            # Plot the original image with all fitted circles and centers
            plt.figure(figsize=(12, 6))
            plt.imshow(image * mask, cmap='gray', origin='lower')
            ax = plt.gca()

            for (xc, yc), R in zip(centers, radii):
                circle = plt.Circle((xc, yc), R, color='red', fill=False, linewidth=1)
                ax.add_patch(circle)
                plt.scatter(xc, yc, color='blue', marker='x', s=50)

            plt.scatter(combined_center_x, combined_center_y, color='yellow', marker='x', s=5, label='Combined Center')
            plt.title("Image with Fitted Circles")
            plt.legend()
            plt.show()

        return combined_center_x, combined_center_y

    except Exception as e:
        print(f"Error processing the image: {e}", flush=True)
        return None, None
