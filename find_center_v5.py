#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

###############################################################################
# Utility Functions
###############################################################################
import numpy as np

def radial_profile(image, center, nbins=200, rmax=None):
    """
    Compute median radial profile of `image` around `center = (cx, cy)`.
    Returns (bin_centers, median_vals).
    
    Parameters
    ----------
    image : 2D numpy array
        The input image (NaN for invalid pixels).
    center : tuple (cx, cy)
        Center coordinates in (x, y) order.
    nbins : int
        Number of radial bins.
    rmax : float or None
        Maximum radius to consider. If None, use half the image diagonal.
    """
    cy, cx = center  # note: typical image indexing is (y, x)
    
    # Grid of coordinates
    height, width = image.shape
    Y, X = np.indices((height, width))
    
    # Distances from the center
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Valid (non-NaN) pixels
    valid_mask = ~np.isnan(image)
    r_valid = r[valid_mask]
    vals_valid = image[valid_mask]
    
    # Decide on maximum radius
    if rmax is None:
        rmax = r_valid.max()
    
    # Create radial bins
    r_edges = np.linspace(0, rmax, nbins+1)
    bin_indices = np.digitize(r_valid, r_edges) - 1  # which bin each pixel belongs to
    
    median_vals = []
    bin_centers = []
    for i in range(nbins):
        in_bin = (bin_indices == i)
        if not np.any(in_bin):
            median_vals.append(np.nan)
        else:
            median_vals.append(np.nanmedian(vals_valid[in_bin]))
        # center of bin
        bin_centers.append(0.5*(r_edges[i] + r_edges[i+1]))
    
    return np.array(bin_centers), np.array(median_vals)
from scipy.ndimage import gaussian_filter1d

def smooth_profile(profile, sigma=5):
    """
    Smooth the 1D profile array using a Gaussian filter.
    """
    return gaussian_filter1d(profile, sigma=sigma, mode="nearest")

def subtract_radial_background(image, center, r_centers, bg_profile):
    """
    Subtract `bg_profile` (which is defined at r_centers radii) from image.
    Returns background-subtracted image (NaN for invalid pixels).
    
    image : 2D array
    center : (cy, cx)
    r_centers : 1D array of radial bin centers
    bg_profile : 1D array of background values at each radial bin
    """
    cy, cx = center
    height, width = image.shape
    Y, X = np.indices((height, width))
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Interpolate background for each radius
    # Outside the range of r_centers, np.interp will clamp to endpoints by default
    bg_values = np.interp(r, r_centers, bg_profile, left=np.nan, right=np.nan)
    
    # Subtract; preserve NaN where image was invalid
    result = image - bg_values
    
    # Enforce that original NaNs remain NaN
    result[np.isnan(image)] = np.nan
    return result

def compute_circle_from_3points(p1, p2, p3):
    """
    Return (cx, cy, r) of the circle passing through 3 points,
    or None if degenerate/collinear.
    """
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2.0 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    if abs(d) < 1e-12:
        return None
    x1_sq = x1**2 + y1**2
    x2_sq = x2**2 + y2**2
    x3_sq = x3**2 + y3**2
    
    ux = (x1_sq*(y2 - y3) + x2_sq*(y3 - y1) + x3_sq*(y1 - y2)) / d
    uy = (x1_sq*(x3 - x2) + x2_sq*(x1 - x3) + x3_sq*(x2 - x1)) / d
    r = np.sqrt((x1 - ux)**2 + (y1 - uy)**2)
    return (ux, uy, r)

def circle_residuals(params, x_data, y_data):
    """
    Residual function for least-squares circle fitting:
    params = (cx, cy, r)
    Returns array of residuals for each point:
      residual_i = distance(point_i, center) - r
    """
    cx, cy, r = params
    distances = np.sqrt((x_data - cx)**2 + (y_data - cy)**2)
    return distances - r

def refine_circle_leastsq(x_data, y_data, initial_guess):
    """
    Least-squares refinement of circle params (cx, cy, r).
    Returns (cx, cy, r) or None if fit fails.
    """
    result = least_squares(
        fun=circle_residuals,
        x0=initial_guess,
        args=(x_data, y_data),
        method='lm'
    )
    if not result.success:
        return None
    return result.x  # (cx, cy, r)

def ransac_circle_fit(
    x_data, y_data,
    max_iterations=2000,
    r_min=100.0,
    r_max=300.0,
    n_initial_points=7
):
    """
    Simple RANSAC-like circle fit with NO distance threshold.
    
    Steps:
      1) Randomly pick n_initial_points -> fit initial circle using least squares
      2) Skip if radius not in [r_min, r_max]
      3) Refine with ALL points (least-squares)
      4) Keep best circle (lowest sum of squared residuals)
    
    Returns (cx, cy, r) or None if no valid circle found.
    """
    points = np.column_stack((x_data, y_data))
    n_points = len(points)
    if n_points < n_initial_points:
        return None

    best_circle = None
    best_resid_sum = np.inf

    rng = np.random.default_rng(seed=42)
    for _ in range(max_iterations):
        # Pick n_initial_points distinct points at random
        sample_idxs = rng.choice(n_points, size=n_initial_points, replace=False)
        sample_points = points[sample_idxs]
        
        # Get initial guess from first 3 points
        initial_circle = compute_circle_from_3points(
            sample_points[0], sample_points[1], sample_points[2]
        )
        if initial_circle is None:
            continue

        # Refine with all n_initial_points
        circle_candidate = refine_circle_leastsq(
            sample_points[:,0], 
            sample_points[:,1],
            initial_circle
        )
        if circle_candidate is None:
            continue
        
        cx_cand, cy_cand, r_cand = circle_candidate

        # Enforce radius constraints
        if not (r_min <= r_cand <= r_max):
            continue

        # Refine with ALL points (since there's no distance threshold)
        refined = refine_circle_leastsq(points[:,0], points[:,1],
                                      (cx_cand, cy_cand, r_cand))
        if refined is None:
            continue
        
        cx_ref, cy_ref, r_ref = refined

        # Compute sum of squared residuals
        resid = circle_residuals((cx_ref, cy_ref, r_ref),
                               points[:,0], points[:,1])
        resid_sum = np.sum(resid**2)
        
        # Keep track of best
        if resid_sum < best_resid_sum:
            best_resid_sum = resid_sum
            best_circle = (cx_ref, cy_ref, r_ref)

    return best_circle

###############################################################################
# Main Script
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Find beam center using a simplified RANSAC circle fit in successive intensity intervals."
    )
    parser.add_argument(
        "--datafile", type=str, default="/Users/xiaodong/Desktop/UOXs/UOX_subset.h5",
        help="Path to HDF5 file with diffraction images at entry/data/images"
    )
    parser.add_argument(
        "--maskfile", type=str, default="/Users/xiaodong/mask/pxmask.h5",
        help="Path to HDF5 file with a mask at /mask (1=valid, 0=invalid)"
    )
    parser.add_argument(
        "--image_index", type=int, default=10,
        help="Which image in the dataset"
    )
    parser.add_argument(
        "--threshold_percentile", type=float, default=99,
        help="Lower percentile for starting intensity threshold (e.g., 99%)"
    )
    parser.add_argument(
        "--exclude_top_percentile", type=float, default=99.1,
        help="Exclude pixels above this upper percentile (e.g., 99.5%)"
    )
    parser.add_argument(
        "--ransac_iterations", type=int, default=2000,
        help="Max RANSAC iterations"
    )
    parser.add_argument(
        "--min_radius", type=float, default=50.0,
        help="Minimum allowed circle radius"
    )
    parser.add_argument(
        "--max_radius", type=float, default=500.0,
        help="Maximum allowed circle radius"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------
    # 1) Load Image & Mask (1=valid, 0=invalid)
    # -------------------------------------------------------------
    with h5py.File(args.datafile, "r") as hf:
        dset = hf["entry"]["data"]["images"]
        img_full = dset[args.image_index].astype(float)
        print(f"Loaded image shape: {img_full.shape}")

    with h5py.File(args.maskfile, "r") as hf_mask:
        raw_mask = hf_mask["/mask"][:].astype(float)
        if raw_mask.shape != img_full.shape:
            raise ValueError("Mask shape does not match image shape!")
        print(f"Loaded mask shape: {raw_mask.shape}")

    # -------------------------------------------------------------
    # 2) Apply mask (invalid => 0 => NaN)
    # -------------------------------------------------------------
    img_masked = img_full * raw_mask
    img_masked[img_masked == 0] = np.nan

    # -------------------------------------------------------------
    # 3) Determine global lower and upper intensity thresholds
    # -------------------------------------------------------------
    valid_vals = img_masked[~np.isnan(img_masked)]
    if valid_vals.size < 10:
        print("Not enough valid pixels after masking!")
        return

    # Lower threshold from percentile
    global_low_val = np.nanpercentile(valid_vals, args.threshold_percentile)
    # Upper threshold from percentile
    global_high_val = np.nanpercentile(valid_vals, args.exclude_top_percentile)

    if global_low_val >= global_high_val:
        print(
            f"Warning: lower threshold ({global_low_val:.2f}) "
            f">= upper threshold ({global_high_val:.2f})!"
        )
        print("Adjust your percentiles. Exiting.")
        return

    print(f"Global intensity range for rings: [{global_low_val:.2f}, {global_high_val:.2f}]")

    # -------------------------------------------------------------
    # 4) Successive ring fitting over expanded intensity intervals
    #    (increment upper bound by 1 until we have enough points).
    # -------------------------------------------------------------
    all_fitted_circles = []  # Will store (cx, cy, r, lower_bound, upper_bound)

    current_lower = global_low_val
    epsilon = 1e-6  # small offset so we don't reuse the same boundary
    step = 1.0      # We'll increment in steps of 1

    while current_lower < global_high_val:
        current_upper = current_lower + step

        while True:
            if current_upper > global_high_val:
                current_upper = global_high_val
            
            # Identify pixels in [current_lower, current_upper]
            ring_mask = (img_masked >= current_lower) & (img_masked <= current_upper)
            y_coords, x_coords = np.nonzero(ring_mask)
            n_pixels = len(x_coords)

            # Stop expanding if we have at least 3 pixels or we've reached global_high_val
            if n_pixels >= 3 or current_upper >= global_high_val:
                break
            else:
                current_upper += step

        # If still fewer than 3 pixels, break
        if len(x_coords) < 3:
            print(f"Not enough pixels ({len(x_coords)}) in final interval [{current_lower:.2f}, {current_upper:.2f}]. Stopping.")
            break

        print(f"\nFitting ring in intensity range [{current_lower:.2f}, {current_upper:.2f}] with {len(x_coords)} pixels.")

        # RANSAC fit (all inliers => no distance threshold)
        fit_result = ransac_circle_fit(
            x_coords, y_coords,
            max_iterations=args.ransac_iterations,
            r_min=args.min_radius,
            r_max=args.max_radius
        )
        if fit_result is not None:
            cx, cy, r = fit_result
            all_fitted_circles.append((cx, cy, r, current_lower, current_upper))
            print(f"  => Found circle: center=({cx:.2f}, {cy:.2f}), radius={r:.2f}")
        else:
            print("  => No valid circle found in this interval.")

        if current_upper >= global_high_val:
            break
        current_lower = current_upper + epsilon

    # -------------------------------------------------------------
    # 5) Combine results from all rings -> mean circle
    # -------------------------------------------------------------
    if len(all_fitted_circles) == 0:
        print("No rings were successfully fitted.")
        return

    # Average the circles
    cx_vals = [c[0] for c in all_fitted_circles]
    cy_vals = [c[1] for c in all_fitted_circles]
    r_vals  = [c[2] for c in all_fitted_circles]

    mean_cx = np.mean(cx_vals)
    mean_cy = np.mean(cy_vals)
    mean_r  = np.mean(r_vals)

    median_cx = np.median(cx_vals)
    median_cy = np.median(cy_vals)
    median_r  = np.median(r_vals)

    print("\nSummary of all fitted rings:")
    for idx, (cx, cy, r, lb, ub) in enumerate(all_fitted_circles):
        print(f"  Ring {idx+1}: range=[{lb:.2f}, {ub:.2f}], center=({cx:.2f},{cy:.2f}), r={r:.2f}")

    print(f"\nMean circle from all rings => center=({mean_cx:.2f}, {mean_cy:.2f}), radius={mean_r:.2f}")

    print(f"\nMedian circle from all rings => center=({median_cx:.2f}, {median_cy:.2f}), radius={median_r:.2f}")

    # -------------------------------------------------------------
    # 6) Plot the result (example: show last ring’s data + mean circle)
    # -------------------------------------------------------------
    last_cx, last_cy, last_r, lb, ub = all_fitted_circles[-1]
    ring_mask = (img_masked >= lb) & (img_masked <= ub)
    y_coords, x_coords = np.nonzero(ring_mask)

    plt.figure(figsize=(8, 8))
    plt.title(f"Last Fitted Ring in [{lb:.2f}, {ub:.2f}] (magenta = median circle, yellow = mean circle)")

    plt.imshow(
        img_masked,
        origin='lower',
        cmap='gray',
        vmin=np.nanpercentile(img_masked, 2),
        vmax=np.nanpercentile(img_masked, 98)
    )
    plt.colorbar(label="Intensity")

    # # Plot ring candidate points
    # plt.scatter(x_coords, y_coords, s=5, c='r', alpha=0.5, label="Ring Pixels")

    # # Draw last circle
    theta = np.linspace(0, 2*np.pi, 360)
    # circle_x = last_cx + last_r * np.cos(theta)
    # circle_y = last_cy + last_r * np.sin(theta)
    # plt.plot(circle_x, circle_y, 'y--', lw=2, label="Last Circle")
    # plt.scatter([last_cx], [last_cy], s=120, c='y', edgecolors='k', label="Last Center")

    # Draw mean circle
    mean_circle_x = mean_cx + mean_r * np.cos(theta)
    mean_circle_y = mean_cy + mean_r * np.sin(theta)
    median_circle_x = median_cx + median_r * np.cos(theta)
    median_circle_y = median_cy + median_r * np.sin(theta)
    plt.plot(median_circle_x, median_circle_y, 'm:', lw=2, label="Median Circle")
    plt.scatter([median_cx], [median_cy], s=120, c='m', edgecolors='k', label="Median Center")
    plt.plot(mean_circle_x, mean_circle_y, 'y:', lw=2, label="Mean Circle")
    plt.scatter([mean_cx], [mean_cy], s=120, c='y', edgecolors='k', label="Mean Center")

    plt.legend(loc='best')
    plt.show()
    
    # -------------------------------------------------------------
    # 7) Radial background removal using the median circle center
    # -------------------------------------------------------------
    from scipy.ndimage import gaussian_filter1d

    # Suppose we decided to use the median center
    center_yx = (median_cy, median_cx)

    # 7.1) Compute radial profile
    nbins = 300  # for example
    r_bins, rad_median = radial_profile(img_masked, center_yx, nbins=nbins)

    # 7.2) Smooth the radial profile
    sigma = 1 # adjust as needed
    rad_median_smooth = gaussian_filter1d(rad_median, sigma=sigma, mode='nearest')

    # 7.3) Subtract radial background
    img_subtracted = subtract_radial_background(img_masked, center_yx, r_bins, rad_median_smooth)

    # -------------------------------------------------------------
    # 8) Plot the radial profile and background-subtracted image
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original masked image
    ax0 = axes[0]
    im0 = ax0.imshow(
        img_masked,
        origin='lower',
        cmap='gray',
        vmin=np.nanpercentile(img_masked, 2),
        vmax=np.nanpercentile(img_masked, 98)
    )
    ax0.set_title("Original Masked Image")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # Radial profile
    ax1 = axes[1]
    ax1.set_title("Median Radial Profile")
    ax1.plot(r_bins, rad_median, 'b-', label='Median')
    ax1.plot(r_bins, rad_median_smooth, 'r-', label='Smoothed')
    ax1.set_xlabel("Radius (pixels)")
    ax1.set_ylabel("Intensity")
    ax1.legend()

    # Background-subtracted image
    ax2 = axes[2]
    im2 = ax2.imshow(
        img_subtracted,
        origin='lower',
        cmap='gray',
        vmin=np.nanpercentile(img_subtracted, 2),
        vmax=np.nanpercentile(img_subtracted, 98)
    )
    ax2.set_title("Background-Subtracted Image")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
