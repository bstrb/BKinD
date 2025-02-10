#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

###############################################################################
# Utility Functions
###############################################################################

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
    if len(x_data) < 3:
        return None
    result = least_squares(
        fun=circle_residuals,
        x0=initial_guess,
        args=(x_data, y_data),
        method='lm'
    )
    if not result.success:
        return None
    return result.x  # (cx, cy, r)

def taubin_circle_fit(x_data, y_data):
    """
    Taubin's algebraic circle fit.
    Returns (cx, cy, r) as an initial guess for the circle.
    Reference: G. Taubin, "Estimation Of Planar Curves, Surfaces
    And Nonplanar Space Curves Defined By Implicit Equations,
    with Applications to Edge and Range Image Segmentation,"
    IEEE Trans. PAMI, 13(11):1115-1138, 1991.

    If fit is degenerate, returns None.
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    if len(x) < 3:
        return None

    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Shift data to mean
    u = x - x_mean
    v = y - y_mean

    # Taubin's method matrices
    # Suu = sum(u^2)
    # Suv = sum(u*v)
    # ...
    Suu = np.sum(u*u)
    Suv = np.sum(u*v)
    Svv = np.sum(v*v)
    Suuu = np.sum(u*u*u)
    Svvv = np.sum(v*v*v)
    Suvv = np.sum(u*v*v)
    Svuu = np.sum(v*u*u)

    # Solve for center
    # Cf. for the exact formulas see e.g. references, but basically
    # we solve a linear system for alpha, beta, then compute radius.
    # Denominator
    den = 2.0 * (Suu * Svv - Suv * Suv)
    if abs(den) < 1e-14:
        # Degenerate (e.g. points are collinear or otherwise ill-conditioned)
        return None

    # The circle center in the 'u,v' coordinate system:
    alpha = (Svv*(Suuu + Suvv) - Suv*(Svvv + Svuu)) / den
    beta  = (Suu*(Svvv + Svuu) - Suv*(Suuu + Suvv)) / den

    # Center in original (x,y) coords
    cx = alpha/2.0 + x_mean
    cy = beta/2.0  + y_mean

    # Radius
    r = np.sqrt(alpha*alpha/4.0 + beta*beta/4.0 +
                (Suu + Svv)/len(x))

    return (cx, cy, r)

###############################################################################
# Main Script
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Find beam center using Taubin’s algebraic circle fit + least-squares refinement in successive intensity intervals."
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
        "--image_index", type=int, default=1100,
        help="Which image in the dataset"
    )
    parser.add_argument(
        "--threshold_percentile", type=float, default=99,
        help="Lower percentile for starting intensity threshold (e.g., 99%)"
    )
    parser.add_argument(
        "--exclude_top_percentile", type=float, default=99.5,
        help="Exclude pixels above this upper percentile (e.g., 99.5%)"
    )
    parser.add_argument(
        "--min_radius", type=float, default=50.0,
        help="Minimum allowed circle radius"
    )
    parser.add_argument(
        "--max_radius", type=float, default=500.0,
        help="Maximum allowed circle radius"
    )
    parser.add_argument(
        "--intensity_step", type=float, default=1,
        help="Step size used in intensity slicing (larger => fewer intervals, faster runtime)"
    )
    parser.add_argument(
        "--outlier_dist_threshold", type=float, default=5,
        help="Maximum allowed distance from the median center to keep a ring fit (in pixels)"
    )
    parser.add_argument(
        "--outlier_radius_threshold", type=float, default=100.0,
        help="Maximum allowed difference from the median radius to keep a ring fit"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------
    # 1) Load Image & Mask
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
    # 3) Determine global lower/upper intensity thresholds
    # -------------------------------------------------------------
    valid_vals = img_masked[~np.isnan(img_masked)]
    if valid_vals.size < 10:
        print("Not enough valid pixels after masking!")
        return

    global_low_val = np.nanpercentile(valid_vals, args.threshold_percentile)
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
    # -------------------------------------------------------------
    all_fitted_circles = []  # Will store dict with info for each ring

    current_lower = global_low_val
    epsilon = 1e-6
    step = args.intensity_step

    while current_lower < global_high_val:
        current_upper = current_lower + step
        if current_upper > global_high_val:
            current_upper = global_high_val

        # Identify pixels in [current_lower, current_upper]
        ring_mask = (img_masked >= current_lower) & (img_masked <= current_upper)
        y_coords, x_coords = np.nonzero(ring_mask)
        n_pixels = len(x_coords)

        if n_pixels >= 3:
            print(f"\nFitting ring in intensity range [{current_lower:.2f}, {current_upper:.2f}] with {n_pixels} pixels.")
            
            # 1) Get algebraic (Taubin) circle fit as an initial guess
            initial_guess = taubin_circle_fit(x_coords, y_coords)
            if initial_guess is not None:
                # 2) Refine with least-squares
                refined = refine_circle_leastsq(x_coords, y_coords, initial_guess)
                if refined is not None:
                    cx_ref, cy_ref, r_ref = refined
                    # Check radius constraints
                    if args.min_radius <= r_ref <= args.max_radius:
                        # Calculate sum-of-squared residuals & # points
                        resid = circle_residuals(refined, x_coords, y_coords)
                        resid_sum = np.sum(resid**2)
                        all_fitted_circles.append({
                            'cx': cx_ref,
                            'cy': cy_ref,
                            'r': r_ref,
                            'n_pts': n_pixels,
                            'resid_sum': resid_sum,
                            'lb': current_lower,
                            'ub': current_upper
                        })
                        print(f"  => Circle found: center=({cx_ref:.2f}, {cy_ref:.2f}), radius={r_ref:.2f}, sumSqResid={resid_sum:.2f}")
                    else:
                        print("  => Circle radius out of bounds, discarding.")
                else:
                    print("  => Least-squares refinement failed.")
            else:
                print("  => Taubin fit degenerate or failed.")
        else:
            print(f"Skipping interval [{current_lower:.2f}, {current_upper:.2f}] with only {n_pixels} pixels.")

        if current_upper >= global_high_val:
            break
        current_lower = current_upper + epsilon

    # -------------------------------------------------------------
    # 5) Filter outlier circles & compute Weighted Mean/Median
    # -------------------------------------------------------------
    if len(all_fitted_circles) == 0:
        print("No rings were successfully fitted.")
        return

    # Convert to arrays for easier handling
    cx_vals = np.array([c['cx'] for c in all_fitted_circles])
    cy_vals = np.array([c['cy'] for c in all_fitted_circles])
    r_vals  = np.array([c['r']  for c in all_fitted_circles])

    # Get median center + radius
    median_cx = np.median(cx_vals)
    median_cy = np.median(cy_vals)
    median_r  = np.median(r_vals)

    # Filter out circles whose center is > outlier_dist_threshold from median center
    # or whose radius is > outlier_radius_threshold away from the median radius
    dist_centers = np.sqrt((cx_vals - median_cx)**2 + (cy_vals - median_cy)**2)
    dist_r = np.abs(r_vals - median_r)

    inlier_mask = (dist_centers <= args.outlier_dist_threshold) & \
                  (dist_r <= args.outlier_radius_threshold)

    filtered_circles = np.array(all_fitted_circles)[inlier_mask]
    n_removed = len(all_fitted_circles) - len(filtered_circles)
    print(f"\nRemoved {n_removed} outlier circle fits based on thresholds.")

    if len(filtered_circles) == 0:
        print("All circle fits were outliers. No final center estimate.")
        return

    # Recompute arrays with filtered data
    cx_vals = np.array([c['cx'] for c in filtered_circles])
    cy_vals = np.array([c['cy'] for c in filtered_circles])
    r_vals  = np.array([c['r']  for c in filtered_circles])
    n_pts   = np.array([c['n_pts'] for c in filtered_circles])
    resid_sums = np.array([c['resid_sum'] for c in filtered_circles])

    # Weighted by (# points / resid_sum), as an example
    # (If resid_sum is zero for a trivial fit, guard against division by zero)
    weights = n_pts / np.maximum(1e-12, resid_sums)

    wsum = np.sum(weights)
    mean_cx_weighted = np.sum(weights * cx_vals) / wsum
    mean_cy_weighted = np.sum(weights * cy_vals) / wsum
    mean_r_weighted  = np.sum(weights * r_vals ) / wsum

    # Unweighted median as well
    median_cx = np.median(cx_vals)
    median_cy = np.median(cy_vals)
    median_r  = np.median(r_vals)

    print("\nSummary of final fitted rings (after outlier removal):")
    for idx, c in enumerate(filtered_circles):
        print(f"  Ring {idx+1}: range=[{c['lb']:.2f}, {c['ub']:.2f}], "
              f"center=({c['cx']:.2f},{c['cy']:.2f}), r={c['r']:.2f}, nPts={c['n_pts']}, sumSqResid={c['resid_sum']:.2f}")

    print(f"\nWeighted-mean circle => center=({mean_cx_weighted:.2f}, {mean_cy_weighted:.2f}), radius={mean_r_weighted:.2f}")
    print(f"Median circle       => center=({median_cx:.2f}, {median_cy:.2f}), radius={median_r:.2f}")

    # -------------------------------------------------------------
    # 6) Plot results (example: plot the last ring used + final circles)
    # -------------------------------------------------------------
    # Just pick the last circle in filtered_circles:
    last_circle = filtered_circles[-1]
    lb = last_circle['lb']
    ub = last_circle['ub']

    ring_mask = (img_masked >= lb) & (img_masked <= ub)
    y_coords, x_coords = np.nonzero(ring_mask)

    plt.figure(figsize=(8, 8))
    plt.title(f"Last Fitted Ring in [{lb:.2f}, {ub:.2f}] (magenta = median circle, yellow = weighted mean)")

    plt.imshow(
        img_masked,
        origin='lower',
        cmap='gray',
        vmin=np.nanpercentile(img_masked, 2),
        vmax=np.nanpercentile(img_masked, 98)
    )
    plt.colorbar(label="Intensity")

    # Draw points for the last ring
    plt.scatter(x_coords, y_coords, s=5, c='r', alpha=0.5, label="Ring Pixels")

    # Draw final circles
    theta = np.linspace(0, 2*np.pi, 360)

    # Weighted mean circle
    w_x = mean_cx_weighted + mean_r_weighted * np.cos(theta)
    w_y = mean_cy_weighted + mean_r_weighted * np.sin(theta)
    plt.plot(w_x, w_y, 'y-', lw=2, label="Weighted Mean Circle")
    plt.scatter([mean_cx_weighted], [mean_cy_weighted], s=120, c='y', edgecolors='k', label="Weighted Mean Ctr")

    # Median circle
    m_x = median_cx + median_r * np.cos(theta)
    m_y = median_cy + median_r * np.sin(theta)
    plt.plot(m_x, m_y, 'm--', lw=2, label="Median Circle")
    plt.scatter([median_cx], [median_cy], s=120, c='m', edgecolors='k', label="Median Ctr")

    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()
