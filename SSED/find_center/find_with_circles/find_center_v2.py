import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN

def circle_residuals(params, x, y):
    # Geometric residual: distance from center minus radius
    a, b, r = params
    dist = np.sqrt((x - a)**2 + (y - b)**2)
    return dist - r

def fit_circle(x, y):
    """Fit a circle (a,b,r) to points (x,y). Returns None if fails."""
    if len(x) < 3:
        return None
    a0, b0 = np.mean(x), np.mean(y)
    r0 = np.median(np.sqrt((x - a0)**2 + (y - b0)**2))
    if r0 == 0:
        return None
    res = least_squares(circle_residuals, [a0,b0,r0], args=(x,y), method='trf', max_nfev=1000)
    if res.success:
        a,b,r = res.x
        return a,b,abs(r)
    return None

def fit_ring_with_outlier_rejection(xs, ys, residual_threshold=1.0, max_iterations=5):
    """
    Iteratively fit a circle and remove pixel outliers.
    Outliers are points whose |distance_from_center - r| > residual_threshold.
    Returns (a,b,r,inlier_xs,inlier_ys) or None.
    """
    inlier_xs, inlier_ys = xs.copy(), ys.copy()
    for _ in range(max_iterations):
        fit = fit_circle(inlier_xs, inlier_ys)
        if fit is None:
            return None
        a, b, r = fit
        dist = np.sqrt((inlier_xs - a)**2 + (inlier_ys - b)**2)
        residuals = np.abs(dist - r)
        good = residuals < residual_threshold
        new_inlier_xs, new_inlier_ys = inlier_xs[good], inlier_ys[good]
        if len(new_inlier_xs) < 3:
            return None
        if len(new_inlier_xs) == len(inlier_xs): # converged
            return a, b, r, inlier_xs, inlier_ys
        inlier_xs, inlier_ys = new_inlier_xs, new_inlier_ys

    fit = fit_circle(inlier_xs, inlier_ys)
    if fit is None:
        return None
    a,b,r = fit
    return a,b,r,inlier_xs,inlier_ys

def get_radial_intensities(img, center, num_levels=20):
    """
    Compute median intensities at radii between 10% and 50% of max_radius.
    """
    y0, x0 = center
    y_idx, x_idx = np.indices(img.shape)
    max_radius = min(y0, x0, img.shape[0]-y0, img.shape[1]-x0)
    radii = np.linspace(max_radius*0.1, max_radius*0.5, num_levels)
    intensities = []
    for r in radii:
        dist = np.sqrt((y_idx - y0)**2 + (x_idx - x0)**2)
        mask = (dist > (r-0.5)) & (dist < (r+0.5))
        vals = img[mask]
        if len(vals) > 0:
            intensities.append(np.nanmedian(vals))
        else:
            intensities.append(np.nan)
    intensities = np.array(intensities)
    valid = ~np.isnan(intensities)
    return radii[valid], intensities[valid]

def find_ring_points(img, intensity_level, delta=10.0, dbscan_eps=3, dbscan_min_samples=5):
    """
    Extract and cluster pixels near a given intensity level using DBSCAN.
    Returns (xs, ys) of largest cluster.
    """
    y_idx, x_idx = np.indices(img.shape)
    mask = (img > (intensity_level - delta)) & (img < (intensity_level + delta))
    xs, ys = x_idx[mask], y_idx[mask]
    if len(xs) < dbscan_min_samples:
        return None, None

    points = np.column_stack((xs, ys))
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points)
    labels = clustering.labels_
    if len(labels) == 0:
        return None, None

    # Find largest non-noise cluster
    unique_labels = np.unique(labels)
    largest_cluster = None
    max_size = 0
    for l in unique_labels:
        if l == -1:
            continue
        size = np.sum(labels == l)
        if size > max_size:
            max_size = size
            largest_cluster = l

    if largest_cluster is None:
        return None, None

    mask_cluster = (labels == largest_cluster)
    return xs[mask_cluster], ys[mask_cluster]

def main(image_file, mask_file, output_dir='output',
         num_levels=20, initial_center=(512,512), intensity_delta_factor=0.1,
         residual_threshold=0.5, max_iterations=10, final_outlier_factor=2.0,
         dbscan_eps=3, dbscan_min_samples=5, plot_results=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(image_file, 'r') as f:
        images = f['entry/data/images'][...]
    with h5py.File(mask_file, 'r') as f:
        mask = f['/mask'][...]

    logfile = os.path.join(output_dir, 'center_fitting_results.txt')
    with open(logfile, 'w') as log:
        log.write("ImageIndex,Y,X,NumFittedRings\n")

        try:
            for idx, img in enumerate(images):
                img_masked = np.where(mask, img, np.nan)
                img_max = np.nanmax(img_masked)
                if np.isnan(img_max) or img_max == 0:
                    print(f"Image {idx}: No valid data, skipping.")
                    log.write(f"{idx},NaN,NaN,0\n")
                    continue

                radii, intensity_levels = get_radial_intensities(img_masked, initial_center, num_levels)
                if len(intensity_levels) == 0:
                    print(f"Image {idx}: No radial intensity levels, skipping.")
                    log.write(f"{idx},NaN,NaN,0\n")
                    continue

                delta = np.nanstd(img_masked) * intensity_delta_factor
                all_centers = []
                all_pixels = []

                for lev in intensity_levels:
                    xs, ys = find_ring_points(img_masked, lev, delta=delta, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples)
                    if xs is None or len(xs)<3:
                        continue
                    result = fit_ring_with_outlier_rejection(xs, ys, residual_threshold=residual_threshold, max_iterations=max_iterations)
                    if result is not None:
                        a, b, r, inlier_xs, inlier_ys = result
                        all_centers.append((a, b))
                        all_pixels.append((inlier_xs, inlier_ys))

                if len(all_centers) == 0:
                    print(f"Image {idx}: No rings fitted.")
                    log.write(f"{idx},NaN,NaN,0\n")
                    continue

                # Ring-level outlier rejection
                centers_array = np.array(all_centers)
                median_center = np.median(centers_array, axis=0)
                dist = np.sqrt(np.sum((centers_array - median_center)**2, axis=1))
                med_dist = np.median(dist)
                good_mask = dist < (med_dist * final_outlier_factor)
                centers_array = centers_array[good_mask]
                all_pixels = [all_pixels[i] for i,m in enumerate(good_mask) if m]

                if len(centers_array) == 0:
                    print(f"Image {idx}: All ring centers outliers.")
                    log.write(f"{idx},NaN,NaN,0\n")
                    continue

                final_center = np.mean(centers_array, axis=0)
                final_x, final_y = final_center
                num_good = len(centers_array)
                print(f"Image {idx}: Final center = (x={final_x:.2f}, y={final_y:.2f}) from {num_good} rings.")
                log.write(f"{idx},{final_y:.2f},{final_x:.2f},{num_good}\n")

                if plot_results:
                    fig, ax = plt.subplots(1,2, figsize=(10,5))
                    ax[0].imshow(img_masked, cmap='inferno', origin='lower')
                    ax[0].plot(final_x, final_y, 'g+', markersize=10, markeredgewidth=2)
                    ax[0].set_title(f"Image {idx} with final center")

                    colors = plt.cm.tab10(np.linspace(0,1,num_good))
                    for i, ((inlier_xs, inlier_ys), (cx,cy)) in enumerate(zip(all_pixels, centers_array)):
                        ax[1].plot(inlier_xs, inlier_ys, '.', color=colors[i], alpha=0.5)
                        ax[1].plot(cx, cy, 'x', color=colors[i], markersize=8)
                    ax[1].plot(final_x, final_y, 'ko', markersize=8, label='Final')
                    ax[1].set_title("Rings used for fitting")
                    fig.tight_layout()
                    fig.savefig(os.path.join(output_dir, f'image_{idx}_center_fit.png'))
                    plt.close(fig)
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving progress...")
            return

if __name__ == '__main__':
    import sys
    # Default values (as original)
    image_file = "/home/bubl3932/files/UOX1/UOXs_find_center/UOXs.h5"
    mask_file = "/home/bubl3932/mask/pxmask.h5"
    output_dir = '/home/bubl3932/files/UOX1/UOXs_find_center/output'
    num_levels = 50
    initial_center = (508, 515)
    intensity_delta_factor = 0.1
    residual_threshold = 0.5
    max_iterations = 10
    final_outlier_factor = 0.5
    dbscan_eps = 3
    dbscan_min_samples = 5
    plot_results = True

    # Command line overrides
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    if len(sys.argv) > 2:
        mask_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    if len(sys.argv) > 4:
        num_levels = int(sys.argv[4])
    if len(sys.argv) > 5:
        initial_center = tuple(map(int, sys.argv[5].split(',')))
    if len(sys.argv) > 6:
        intensity_delta_factor = float(sys.argv[6])
    if len(sys.argv) > 7:
        residual_threshold = float(sys.argv[7])
    if len(sys.argv) > 8:
        max_iterations = int(sys.argv[8])
    if len(sys.argv) > 9:
        final_outlier_factor = float(sys.argv[9])
    if len(sys.argv) > 10:
        dbscan_eps = float(sys.argv[10])
    if len(sys.argv) > 11:
        dbscan_min_samples = int(sys.argv[11])
    if len(sys.argv) > 12:
        plot_results = bool(int(sys.argv[12]))

    main(image_file, mask_file, output_dir, num_levels, initial_center, intensity_delta_factor,
         residual_threshold, max_iterations, final_outlier_factor, dbscan_eps, dbscan_min_samples,
         plot_results)
