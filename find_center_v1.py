#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

###############################################################################
# Utility Functions
###############################################################################

def bin_image(img, bin_factor=1):
    """
    Simple block-mean binning by an integer factor.
    If bin_factor=1, returns img unchanged.
    """
    if bin_factor <= 1:
        return img
    
    h, w = img.shape
    h2, w2 = h // bin_factor, w // bin_factor
    # Truncate rows/cols to be multiples of bin_factor
    img = img[:h2 * bin_factor, :w2 * bin_factor]
    # Reshape and mean over blocks
    img_binned = img.reshape(h2, bin_factor, w2, bin_factor).mean(axis=(1,3))
    return img_binned

def compute_circle_from_3points(p1, p2, p3):
    """
    Return (cx, cy, r) of the circle passing through 3 points
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
    """
    cx, cy, r = params
    return np.sqrt((x_data - cx)**2 + (y_data - cy)**2) - r

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
        distance_threshold=2.0,
        min_inliers=50,
        r_min=100.0,
        r_max=300.0
    ):
    """
    Simple RANSAC-based circle fit:
      1) Randomly pick 3 points -> define circle
      2) Reject if circle radius not in [r_min, r_max]
      3) Count inliers (dist < distance_threshold)
      4) Keep best -> refine with least-squares
    Returns (cx, cy, r, inlier_mask) or None if unsuccessful.
    """
    points = np.column_stack((x_data, y_data))
    n_points = len(points)
    if n_points < 3:
        return None

    best_inlier_count = 0
    best_circle = None
    best_inlier_mask = None

    rng = np.random.default_rng(seed=42)
    for _ in range(max_iterations):
        # Pick 3 distinct points at random
        sample_idxs = rng.choice(n_points, size=3, replace=False)
        p1, p2, p3 = points[sample_idxs]
        circle_candidate = compute_circle_from_3points(p1, p2, p3)
        if circle_candidate is None:
            continue

        cx_cand, cy_cand, r_cand = circle_candidate
        # Enforce radius constraints
        if not (r_min <= r_cand <= r_max):
            continue

        # Compute inliers
        dist = np.abs(
            np.sqrt((points[:,0] - cx_cand)**2 + (points[:,1] - cy_cand)**2) - r_cand
        )
        inliers = (dist < distance_threshold)
        count_inliers = np.count_nonzero(inliers)

        if count_inliers > best_inlier_count:
            best_inlier_count = count_inliers
            best_circle = (cx_cand, cy_cand, r_cand)
            best_inlier_mask = inliers

    if best_circle is None or best_inlier_count < min_inliers:
        return None

    # Refine with least squares on the inliers
    cx_init, cy_init, r_init = best_circle
    x_inliers = x_data[best_inlier_mask]
    y_inliers = y_data[best_inlier_mask]

    refined = refine_circle_leastsq(x_inliers, y_inliers, (cx_init, cy_init, r_init))
    if refined is None:
        return None
    cx_fin, cy_fin, r_fin = refined
    return (cx_fin, cy_fin, r_fin, best_inlier_mask)

###############################################################################
# Main Script
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Find beam center using RANSAC circle fit, with a mask where 1=valid, 0=invalid."
    )
    parser.add_argument(
        "--datafile", type=str, default="/Users/xiaodong/Desktop/UOX1/deiced_UOX1_min_15_peak_backup.h5",
        help="Path to HDF5 file with diffraction images at entry/data/images"
    )
    parser.add_argument(
        "--maskfile", type=str, default="/Users/xiaodong/mask/pxmask.h5",
        help="Path to HDF5 file with a mask at /mask (1=valid, 0=invalid)"
    )
    parser.add_argument(
        "--image_index", type=int, default=1010,
        help="Which image in the dataset"
    )
    parser.add_argument(
        "--bin_factor", type=int, default=1,
        help="Integer factor to bin the image and mask"
    )
    parser.add_argument(
        "--median_kernel", type=int, default=3,
        help="Size of median filter (0 => no median filter)"
    )
    parser.add_argument(
        "--threshold_percentile", type=float, default=98.0,
        help="Pick pixels above this lower percentile (e.g., 90 = top 10%)"
    )
    parser.add_argument(
        "--exclude_top_percentile", type=float, default=98.5,
        help="Exclude pixels above this upper percentile (e.g., 99 = exclude top 1%)"
    )
    parser.add_argument(
        "--ransac_iterations", type=int, default=2000,
        help="Max RANSAC iterations"
    )
    parser.add_argument(
        "--distance_threshold", type=float, default=50,
        help="Inlier distance threshold in pixels"
    )
    parser.add_argument(
        "--min_inliers", type=int, default=40,
        help="Minimum inliers for a valid circle"
    )
    parser.add_argument(
        "--min_radius", type=float, default=100.0,
        help="Minimum allowed circle radius"
    )
    parser.add_argument(
        "--max_radius", type=float, default=300.0,
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
    # 2) Bin the image and mask if requested
    # -------------------------------------------------------------
    if args.bin_factor > 1:
        print(f"Binning image and mask by factor {args.bin_factor}")
        # Bin the image
        img_binned = bin_image(img_full, args.bin_factor)
        # Bin the mask => use block mean, then threshold at 0.999...
        mask_binned_f = bin_image(raw_mask, args.bin_factor)
        mask_binned = (mask_binned_f > 0.999).astype(float)
    else:
        img_binned = img_full.copy()
        mask_binned = raw_mask.copy()

    # -------------------------------------------------------------
    # 3) Optional median filtering
    # -------------------------------------------------------------
    if args.median_kernel > 1:
        print(f"Applying median filter size {args.median_kernel}")
        img_binned = median_filter(img_binned, size=args.median_kernel)

    # -------------------------------------------------------------
    # 4) Apply the mask: invalid => 0 => NaN
    # -------------------------------------------------------------
    img_masked = img_binned * mask_binned
    img_masked[img_masked == 0] = np.nan

    # -------------------------------------------------------------
    # 5) Double-threshold ring pixels
    # -------------------------------------------------------------
    valid_vals = img_masked[~np.isnan(img_masked)]
    if valid_vals.size < 10:
        print("Not enough valid pixels after masking!")
        return

    # Lower threshold
    low_thresh_val = np.nanpercentile(valid_vals, args.threshold_percentile)
    # Upper threshold
    high_thresh_val = np.nanpercentile(valid_vals, args.exclude_top_percentile)

    if low_thresh_val >= high_thresh_val:
        print(f"Warning: lower threshold ({low_thresh_val:.2f}) >= upper threshold ({high_thresh_val:.2f})!")
        print("Adjust your percentiles. Exiting.")
        return

    # Keep intensities in [low_thresh_val, high_thresh_val]
    ring_pixels = (img_masked >= low_thresh_val) & (img_masked <= high_thresh_val)
    y_coords, x_coords = np.nonzero(ring_pixels)
    print(f"Found {len(x_coords)} ring pixels in [{low_thresh_val:.2f}, {high_thresh_val:.2f}] intensity.")

    if len(x_coords) < 3:
        print("Not enough ring pixels to fit a circle. Exiting.")
        return

    # -------------------------------------------------------------
    # 6) RANSAC Circle Fit
    # -------------------------------------------------------------
    fit_result = ransac_circle_fit(
        x_coords, y_coords,
        max_iterations=args.ransac_iterations,
        distance_threshold=args.distance_threshold,
        min_inliers=args.min_inliers,
        r_min=args.min_radius,
        r_max=args.max_radius
    )
    if fit_result is None:
        print("No valid circle found within the specified radius range.")
        return

    cx, cy, r, inlier_mask = fit_result
    num_inliers = np.count_nonzero(inlier_mask)
    print(f"Fitted circle => center=({cx:.2f}, {cy:.2f}), radius={r:.2f}, inliers={num_inliers}")

    # -------------------------------------------------------------
    # 7) Plot the result
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.title(f"Circle Fit with Intensity in [{low_thresh_val:.2f}, {high_thresh_val:.2f}]")

    plt.imshow(
        img_masked,
        origin='lower',
        cmap='gray',
        vmin=np.nanpercentile(img_masked, 2),
        vmax=np.nanpercentile(img_masked, 98)
    )
    plt.colorbar(label="Intensity")

    # Candidate ring pixels
    plt.scatter(x_coords, y_coords, s=5, c='r', alpha=0.3, label="Ring Candidates")

    # RANSAC inliers
    inlier_x = x_coords[inlier_mask]
    inlier_y = y_coords[inlier_mask]
    plt.scatter(inlier_x, inlier_y, s=5, c='lime', alpha=0.6, label="Inliers")

    # Circle
    theta = np.linspace(0, 2*np.pi, 360)
    circle_x = cx + r * np.cos(theta)
    circle_y = cy + r * np.sin(theta)
    plt.plot(circle_x, circle_y, 'y--', lw=2, label="Fitted Circle")

    # Center
    plt.scatter([cx], [cy], s=120, c='magenta', edgecolors='black', label="Beam Center")

    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()
