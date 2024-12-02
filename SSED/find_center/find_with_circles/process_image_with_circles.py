import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def fit_circle_least_squares(x, y):
    """
    Fit a circle to the given x, y coordinates using least squares.

    Returns:
        xc, yc: Center coordinates of the circle.
        R: Radius of the circle.
    """
    def calc_R(c):
        xc, yc = c
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        Ri = calc_R(c)
        return Ri - Ri.mean()

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, _ = leastsq(f_2, center_estimate)
    Ri = calc_R(center)
    R = Ri.mean()
    return center[0], center[1], R

def process_image(image, image_index, mask=None, plot=True, verbose=True):
    """
    Processes the image to fit circles at specified intensity levels.

    Parameters:
        image (ndarray): 2D array representing the image.
        mask (ndarray): Optional boolean mask array of the same shape as image.
        plot (bool): Whether to display plots.
        verbose (bool): Whether to print detailed processing information.

    Returns:
        combined_center_x (float): X-coordinate of the combined center.
        combined_center_y (float): Y-coordinate of the combined center.
    """
    try:
        if verbose:
            print(f"Processing image {image_index}...", flush=True)

        # Ensure mask is boolean and matches image dimensions
        if mask is not None:
            if mask.dtype != np.bool_:
                mask = mask > 0
            if mask.shape != image.shape:
                print("Mask shape does not match image shape.", flush=True)
                return None, None
        else:
            mask = np.ones_like(image, dtype=bool)

        # Extract unmasked pixels
        unmasked_pixels = image[mask]

        if unmasked_pixels.size == 0:
            print(f"No unmasked pixels in the image.", flush=True)
            return None, None

        # Normalize the image based on unmasked pixels
        min_intensity = unmasked_pixels.min()
        max_intensity = unmasked_pixels.max()
        if max_intensity == min_intensity:
            print(f"Image has constant intensity in unmasked region.", flush=True)
            return None, None

        # Create a normalized image, setting masked regions to NaN
        norm_image = np.full_like(image, np.nan, dtype=np.float64)
        norm_image[mask] = (image[mask] - min_intensity) / (max_intensity - min_intensity)
        bin_size = 1
        # Define multiple intensity levels and bin widths (adjustable within the function)
        intensity_levels_percent = list(range(0, 15+bin_size, bin_size*2)) #[10, 15, 20, 25, 30]  # List of intensity levels in percentages
        bin_widths_percent = [bin_size] * len(intensity_levels_percent) #[1, 1, 1, 1, 1]           # Corresponding bin widths in percentages

        if len(intensity_levels_percent) != len(bin_widths_percent):
            print("The number of intensity levels and bin widths must be the same.", flush=True)
            return None, None

        centers = []
        radii = []
        intensity_levels_used = []
        x_positions = []
        y_positions = []

        for intensity_level_percent, bin_width_percent in zip(intensity_levels_percent, bin_widths_percent):
            # Convert percentages to normalized intensity values (between 0 and 1)
            intensity_level = intensity_level_percent / 100.0
            bin_width = bin_width_percent / 100.0

            # Define the intensity bin range
            level_min = intensity_level - bin_width / 2
            level_max = intensity_level + bin_width / 2

            # Ensure the intensity levels are within [0, 1]
            level_min = max(0.0, level_min)
            level_max = min(1.0, level_max)

            # Get indices of unmasked pixels within the intensity bin
            bin_mask = (norm_image >= level_min) & (norm_image < level_max)

            indices = np.argwhere(bin_mask)

            if len(indices) == 0:
                if verbose:
                    print(f"No pixels found in intensity bin {level_min:.3f} - {level_max:.3f}", flush=True)
                continue

            y_coords, x_coords = indices[:, 0], indices[:, 1]

            # Fit a circle to these points
            xc, yc, R = fit_circle_least_squares(x_coords, y_coords)

            centers.append((xc, yc))
            radii.append(R)
            intensity_levels_used.append(intensity_level_percent)
            x_positions.append(xc)
            y_positions.append(yc)

            if verbose:
                print(f"Fitted circle at intensity level {intensity_level_percent}% ± {bin_width_percent / 2}%: "
                      f"center=({xc:.2f}, {yc:.2f}), radius={R:.2f}", flush=True)

            if plot:
                # Plot the points and the fitted circle
                plt.figure(figsize=(6, 6))
                plt.title(f'Intensity bin {level_min:.3f} - {level_max:.3f}')
                plt.imshow(image * mask, cmap='gray', origin='lower')
                plt.plot(x_coords, y_coords, 'b.', label='Points in bin')
                circle = plt.Circle((xc, yc), R, color='red', fill=False, linewidth=1, label='Fitted Circle')
                ax = plt.gca()
                ax.add_patch(circle)
                plt.scatter(xc, yc, color='yellow', marker='x', s=10, label='Circle Center')
                plt.legend()
                plt.show()

        if len(centers) == 0:
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

            # Left subplot: image with circles
            plt.subplot(1, 2, 1)
            plt.imshow(image * mask, cmap='gray', origin='lower')
            ax = plt.gca()

            # Plot each fitted circle and center
            for (xc, yc), R in zip(centers, radii):
                circle = plt.Circle((xc, yc), R, color='red', fill=False, linewidth=0.5)
                ax.add_patch(circle)
                plt.scatter(xc, yc, color='blue', marker='x', s=50, linewidth=0.5)

            # Plot the combined center
            plt.scatter(combined_center_x, combined_center_y, color='yellow', marker='o', s=100, label='Combined Center')

            plt.title("Image with Fitted Circles")
            plt.legend()

            # Right subplot: center positions vs intensity levels
            plt.subplot(1, 2, 2)
            plt.plot(intensity_levels_used, x_positions, label='X Position', color='green', marker='o')
            plt.plot(intensity_levels_used, y_positions, label='Y Position', color='magenta', marker='o')
            plt.xlabel('Intensity Level (%)')
            plt.ylabel('Position (pixels)')
            plt.title('Center Positions vs. Intensity Levels')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        # Return the combined center coordinates
        return combined_center_x, combined_center_y

    except Exception as e:
        print(f"Error processing the image: {e}", flush=True)
        return None, None
