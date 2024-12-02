import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage import measure
from scipy.optimize import leastsq

def fit_circle_least_squares(x, y):
    """
    Fit a circle to the given x, y coordinates using least squares.

    Returns:
        xc, yc: Center coordinates of the circle.
        R: Radius of the circle.
    """
    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = leastsq(f_2, center_estimate)
    Ri = calc_R(*center)
    R = Ri.mean()
    return center[0], center[1], R

def process_image_with_circles(image, image_index, mask, plot=True, verbose=True):
    try:
        if verbose:
            print(f"Processing image {image_index}...", flush=True)

        # Apply median filter to reduce noise
        filtered_image = median_filter(image, size=3)

        # Exclude high-intensity pixels (e.g., top 0.01% of intensities)
        nonzero_pixels = filtered_image[mask > 0]
        if nonzero_pixels.size == 0:
            print(f"No non-zero pixels in image {image_index} after filtering.", flush=True)
            return None, None

        intensity_threshold = np.percentile(nonzero_pixels, 99.99)
        filtered_image[filtered_image > intensity_threshold] = intensity_threshold

        # Normalize the image to [0, 1]
        norm_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())

        # Define intensity levels (percentages of maximum intensity)
        intensity_levels = list(np.arange(0.1, 0.6, 0.01))

        centers = []
        radii = []
        intensity_levels_used = []
        x_positions = []
        y_positions = []

        for level in intensity_levels:
            # Find contours at the given intensity level
            contours = measure.find_contours(norm_image * mask, level)

            if len(contours) == 0:
                if verbose:
                    print(f"No contours found at level {level:.3f}", flush=True)
                continue

            # Choose the largest contour (assuming it's the main ring)
            contour_lengths = [len(contour) for contour in contours]
            max_length_index = np.argmax(contour_lengths)
            contour = contours[max_length_index]

            # Extract x and y coordinates
            y_coords, x_coords = contour[:, 0], contour[:, 1]

            # Fit a circle to the contour points
            xc, yc, R = fit_circle_least_squares(x_coords, y_coords)

            centers.append((xc, yc))
            radii.append(R)
            intensity_levels_used.append(level)
            x_positions.append(xc)
            y_positions.append(yc)

            if verbose:
                print(f"Fitted circle at level {level:.3f}: center=({xc:.2f}, {yc:.2f}), radius={R:.2f}", flush=True)

        if len(centers) == 0:
            print(f"No circles were fitted for image {image_index}.", flush=True)
            return None, None

        # Compute the combined center as the mean of the centers
        centers_array = np.array(centers)
        combined_center = centers_array.mean(axis=0)

        if verbose:
            print(f"Combined center: ({combined_center[0]:.2f}, {combined_center[1]:.2f})", flush=True)

        if plot:
            # Plot the original image with the fitted circles and centers
            plt.figure(figsize=(12, 6))

            # Plotting the image with circles
            plt.subplot(1, 2, 1)
            plt.imshow(image * mask, cmap='gray', origin='lower')
            ax = plt.gca()

            # Plot each fitted circle and center
            for (xc, yc), R in zip(centers, radii):
                circle = plt.Circle((xc, yc), R, color='red', fill=False, linewidth=0.5)
                ax.add_patch(circle)
                plt.scatter(xc, yc, color='blue', marker='x', s=50, linewidth=0.5)

            # Plot the combined center
            plt.scatter(combined_center[0], combined_center[1], color='yellow', marker='o', s=100, label='Combined Center')

            plt.title(f"Image with Fitted Circles (Index {image_index})")
            plt.legend()

            # Plotting the x and y positions vs intensity levels
            plt.subplot(1, 2, 2)
            plt.plot(intensity_levels_used, x_positions, label='X Position', color='green')
            plt.plot(intensity_levels_used, y_positions, label='Y Position', color='magenta')
            plt.xlabel('Intensity Level')
            plt.ylabel('Position (pixels)')
            plt.title('Center Positions vs. Intensity Levels')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        # Return the combined center coordinates
        return combined_center[0], combined_center[1]

    except Exception as e:
        print(f"Error processing image {image_index}: {e}", flush=True)
        return None, None
