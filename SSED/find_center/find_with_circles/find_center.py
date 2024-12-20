import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN

def circle_residuals(params, x, y):
    # params = (a, b, r) where (a,b) is center and r is radius
    a, b, r = params
    return ((x - a)**2 + (y - b)**2 - r**2)

def fit_circle(x, y):
    """
    Fit a circle to a set of points (x,y) using least squares.
    Returns (a, b, r) as the center and radius of the fitted circle, or None if fail.
    """
    if len(x) < 3:
        # Not enough points to fit a circle
        return None

    # Initial guess: center = mean of points, radius = mean distance to center
    x_m = np.mean(x)
    y_m = np.mean(y)
    r_m = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))

    res = least_squares(circle_residuals, x0=[x_m, y_m, r_m], args=(x, y), method='lm', max_nfev=1000)
    if res.success:
        a, b, r = res.x
        return a, b, abs(r)
    else:
        return None

def fit_ring_with_outlier_rejection(xs, ys, residual_threshold=2.0, max_iterations=5):
    """
    Iteratively fit a circle to the given points (xs, ys) and remove outliers.
    Outliers are points whose distance from the fitted circle is greater than residual_threshold.

    Returns:
        a, b, r: fitted circle parameters
        inlier_xs, inlier_ys: the points used in the final fit (after outlier rejection)
    or None if a fit cannot be obtained.
    """
    if len(xs) < 3:
        return None

    inlier_xs = xs.copy()
    inlier_ys = ys.copy()

    for _ in range(max_iterations):
        fit = fit_circle(inlier_xs, inlier_ys)
        if fit is None:
            return None  # Cannot fit a circle at all
        a, b, r = fit
        # Compute residuals (distance from circle)
        dist = np.abs((inlier_xs - a)**2 + (inlier_ys - b)**2 - r**2)
        dist = np.sqrt(dist + r**2) - r  # Another way: residual as difference in radius
        # Actually, the residual can be computed simply as:
        # distance of point from center: sqrt((x - a)^2 + (y - b)^2)
        # residual = |distance_from_center - r|
        distance_from_center = np.sqrt((inlier_xs - a)**2 + (inlier_ys - b)**2)
        residuals = np.abs(distance_from_center - r)

        good_mask = residuals < residual_threshold
        new_inlier_xs = inlier_xs[good_mask]
        new_inlier_ys = inlier_ys[good_mask]

        if len(new_inlier_xs) < 3:
            return None  # Too few points remain

        # If no change in inliers, we have converged
        if len(new_inlier_xs) == len(inlier_xs):
            # No outliers removed this iteration, done
            return a, b, r, new_inlier_xs, new_inlier_ys

        inlier_xs = new_inlier_xs
        inlier_ys = new_inlier_ys

    # After max_iterations, return the last fit
    fit = fit_circle(inlier_xs, inlier_ys)
    if fit is None:
        return None
    a, b, r = fit
    return a, b, r, inlier_xs, inlier_ys

def get_radial_intensities(img, center, num_levels):
    """
    Get median intensities at different radii from the center.
    """
    y_idx, x_idx = np.indices(img.shape)
    max_radius = min(center[0], center[1], img.shape[0]-center[0], img.shape[1]-center[1])
    radii = np.linspace(max_radius*0.1, max_radius*0.5, num_levels)
    
    intensities = []
    for r in radii:
        # Create an annulus mask
        distances = np.sqrt((y_idx - center[0])**2 + (x_idx - center[1])**2)
        ring_mask = (distances > (r-0.5)) & (distances < (r+0.5))
        ring_values = img[ring_mask]
        if len(ring_values) > 0:
            intensities.append(np.nanmedian(ring_values))
    
    return np.array(intensities)

def find_ring_points(img_masked, intensity_level, delta, eps=3, min_samples=5):
    """
    Find points belonging to a ring using DBSCAN clustering.
    
    Parameters:
    -----------
    eps : float
        DBSCAN parameter for neighborhood size
    min_samples : int
        Minimum points to form a cluster
    """
    y_idx, x_idx = np.indices(img_masked.shape)
    
    # Extract points close to this intensity level
    mask_ring = (img_masked > (intensity_level - delta)) & (img_masked < (intensity_level + delta))
    ys = y_idx[mask_ring]
    xs = x_idx[mask_ring]
    
    if len(xs) < min_samples:
        return None, None
    
    # Apply DBSCAN clustering
    points = np.column_stack((xs, ys))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # Find the largest cluster
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:  # Only noise points found
        return None, None
    
    largest_cluster = None
    max_size = 0
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_size = np.sum(labels == label)
        if cluster_size > max_size:
            max_size = cluster_size
            largest_cluster = label
    
    if largest_cluster is None:
        return None, None
    
    # Return points from largest cluster
    cluster_mask = labels == largest_cluster
    return xs[cluster_mask], ys[cluster_mask]

def main(image_file, mask_file, output_dir='output', num_levels=50, initial_center=(512,512), 
         residual_threshold=2.0, max_iterations=5, dbscan_eps=3, dbscan_min_samples=5):
    """
    Parameters updated to include initial_center and remove low/high intensity params
    ----------
    image_file : str
        Path to the HDF5 file containing images under 'entry/data/images'.
    mask_file : str
        Path to the HDF5 file containing the mask under '/mask'.
    output_dir : str
        Directory to store output results.
    num_levels : int
        Number of intensity levels (rings) to fit between low_intensity and high_intensity fractions of max intensity.
    initial_center : tuple
        Initial guess for the center position as (y, x), defaults to (512, 512)
    residual_threshold : float
        Threshold in pixels for pixel-level outlier rejection per ring.
    max_iterations : int
        Maximum iterations for the outlier rejection process.
    dbscan_eps : float
        DBSCAN parameter for neighborhood size
    dbscan_min_samples : int
        Minimum points to form a cluster
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load images
    with h5py.File(image_file, 'r') as f:
        images = f['entry/data/images'][...]

    # Load mask (0 = masked, 1=good)
    with h5py.File(mask_file, 'r') as f:
        mask = f['/mask'][...]

    logfile = os.path.join(output_dir, 'center_fitting_results.txt')
    with open(logfile, 'w') as log:
        log.write("ImageIndex,CenterY,CenterX,NumFittedRings\n")

        for idx, img in enumerate(images):
            # Apply mask: set masked pixels to NaN where mask is False
            img_masked = np.where(mask, img, np.nan)

            # Determine intensity range
            img_max = np.nanmax(img_masked)
            if np.isnan(img_max) or img_max == 0:
                print(f"Image {idx}: No valid data or max intensity is zero, skipping.")
                log.write(f"{idx},NaN,NaN,0\n")
                continue

            # Replace the intensity level calculation with radial intensities
            intensity_levels = get_radial_intensities(img_masked, initial_center, num_levels)
            if len(intensity_levels) == 0:
                print(f"Image {idx}: Could not determine intensity levels, skipping.")
                log.write(f"{idx},NaN,NaN,0\n")
                continue

            # For plotting, we want to store the final used points of each ring
            all_ring_centers = []
            all_ring_pixels = []  # list of (xs, ys) for each ring
            radii = []

            # Define a narrow band around each level
            delta = np.nanstd(img_masked) * 0.1  # Adaptive delta based on image statistics
            y_idx, x_idx = np.indices(img_masked.shape)

            for lev in intensity_levels:
                # Use DBSCAN to find ring points
                xs, ys = find_ring_points(img_masked, lev, delta, 
                                        eps=dbscan_eps, 
                                        min_samples=dbscan_min_samples)
                
                if xs is None or len(xs) < 3:
                    continue

                # Attempt iterative outlier rejection circle fit
                result = fit_ring_with_outlier_rejection(xs, ys, 
                                                       residual_threshold=residual_threshold, 
                                                       max_iterations=max_iterations)
                if result is not None:
                    a, b, r, inlier_xs, inlier_ys = result
                    # Note: fit returns (a,b) as (x_center,y_center)
                    # We want final center in (y,x)
                    all_ring_centers.append((b, a))
                    radii.append(r)
                    all_ring_pixels.append((inlier_xs, inlier_ys))

            if len(all_ring_centers) == 0:
                print(f"Image {idx}: No rings fitted successfully.")
                log.write(f"{idx},NaN,NaN,0\n")
                continue

            # Before computing final center, filter out outlier ring centers
            centers_array = np.array(all_ring_centers)
            if len(centers_array) > 3:  # Only filter if we have enough rings
                # Calculate distances from median center
                median_center = np.median(centers_array, axis=0)
                distances = np.sqrt(np.sum((centers_array - median_center)**2, axis=1))
                
                # Use median absolute deviation (MAD) for robust outlier detection
                mad = np.median(np.abs(distances - np.median(distances)))
                threshold = np.median(distances) +  0.1*mad  # 5 is a tunable parameter
                
                # Filter centers and corresponding data
                good_indices = distances < threshold
                centers_array = centers_array[good_indices]
                all_ring_pixels = [all_ring_pixels[i] for i in range(len(all_ring_pixels)) if good_indices[i]]
                radii = [radii[i] for i in range(len(radii)) if good_indices[i]]

            # Now compute final center from filtered centers
            final_center = np.mean(centers_array, axis=0)
            final_y, final_x = final_center
            num_good = len(centers_array)

            print(f"Image {idx}: Final center = ({final_y:.2f}, {final_x:.2f}) from {num_good} rings.")
            log.write(f"{idx},{final_y:.2f},{final_x:.2f},{num_good}\n")

            # Plot diagnostics
            fig, ax = plt.subplots(1,2, figsize=(10,5))

            # Left: image with final center
            ax[0].imshow(img_masked, cmap='inferno', origin='lower')
            ax[0].plot(final_x, final_y, 'g+', markersize=10, markeredgewidth=2)
            ax[0].set_title(f"Image {idx} with final center")
            ax[0].set_xlabel("X (pixels)")
            ax[0].set_ylabel("Y (pixels)")

            # Right: show ring pixels and ring centers
            # Assign each ring a color
            colors = plt.colormaps['tab10'](np.linspace(0, 1, num_good))
            for i, ((inlier_xs, inlier_ys), (cy, cx)) in enumerate(zip(all_ring_pixels, all_ring_centers)):
                ax[1].plot(inlier_xs, inlier_ys, '.', color=colors[i], alpha=0.5, label=f'Ring {i}')
                ax[1].plot(cx, cy, 'x', color=colors[i], markersize=10)

            ax[1].plot(final_x, final_y, 'ko', markersize=8, label='Final center')
            ax[1].set_title("Rings used for fitting")
            ax[1].set_xlabel("X (pixels)")
            ax[1].set_ylabel("Y (pixels)")
            ax[1].legend(loc='best', fontsize='small')

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'image_{idx}_center_fit.png'))
            plt.close(fig)


if __name__ == '__main__':
    import sys
    
    # Default values
    image_file = "/home/bubl3932/files/UOX1/UOXs_find_center/UOXs.h5"
    mask_file = "/home/bubl3932/mask/pxmask.h5"
    output_dir = '/home/bubl3932/files/UOX1/UOXs_find_center/output'
    num_levels = 20
    initial_center = (512, 512)  # New default
    residual_threshold = 0.1
    max_iterations = 10
    dbscan_eps = 3
    dbscan_min_samples = 5

    # Override defaults with command line arguments if provided
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    if len(sys.argv) > 2:
        mask_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    if len(sys.argv) > 4:
        num_levels = int(sys.argv[4])
    if len(sys.argv) > 5:
        initial_center = tuple(map(int, sys.argv[5].split(',')))  # Expect format "y,x"
    if len(sys.argv) > 6:
        residual_threshold = float(sys.argv[6])
    if len(sys.argv) > 7:
        max_iterations = int(sys.argv[7])
    if len(sys.argv) > 8:
        dbscan_eps = float(sys.argv[8])
    if len(sys.argv) > 9:
        dbscan_min_samples = int(sys.argv[9])

    main(image_file, mask_file, output_dir, num_levels, initial_center, 
         residual_threshold, max_iterations, dbscan_eps, dbscan_min_samples)
