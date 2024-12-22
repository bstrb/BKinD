# find_center_remove_background.py

#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

###############################################################################
# Utility Functions
###############################################################################

def radial_profile(image, center, nbins=200, rmax=None):
    """
    Compute median radial profile of `image` around `center = (cx, cy)`.
    Returns (bin_centers, median_vals).
    
    image : 2D numpy array (NaN for invalid pixels).
    center : tuple (cx, cy)
    nbins : int
    rmax : float or None
    """
    cy, cx = center  # note: typical image indexing is (y, x)
    
    height, width = image.shape
    Y, X = np.indices((height, width))
    
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    valid_mask = ~np.isnan(image)
    r_valid = r[valid_mask]
    vals_valid = image[valid_mask]
    
    if rmax is None:
        rmax = r_valid.max()
    
    r_edges = np.linspace(0, rmax, nbins+1)
    bin_indices = np.digitize(r_valid, r_edges) - 1
    
    median_vals = []
    bin_centers = []
    for i in range(nbins):
        in_bin = (bin_indices == i)
        if not np.any(in_bin):
            median_vals.append(np.nan)
        else:
            median_vals.append(np.nanmedian(vals_valid[in_bin]))
        bin_centers.append(0.5*(r_edges[i] + r_edges[i+1]))
    
    return np.array(bin_centers), np.array(median_vals)

def smooth_profile(profile, sigma=5):
    """
    Smooth the 1D profile array using a Gaussian filter.
    """
    return gaussian_filter1d(profile, sigma=sigma, mode="nearest")

def subtract_radial_background(image, center, r_centers, bg_profile):
    """
    Subtract `bg_profile` (defined at r_centers radii) from image.
    
    image : 2D array
    center : (cy, cx)
    r_centers : 1D array
    bg_profile : 1D array of background values at each radial bin
    """
    cy, cx = center
    height, width = image.shape
    Y, X = np.indices((height, width))
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    bg_values = np.interp(r, r_centers, bg_profile, left=np.nan, right=np.nan)
    result = image - bg_values
    # Keep invalid pixels as NaN
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

        # Refine with those n_initial_points
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

        # Refine with ALL points
        refined = refine_circle_leastsq(points[:,0], points[:,1],
                                        (cx_cand, cy_cand, r_cand))
        if refined is None:
            continue
        
        cx_ref, cy_ref, r_ref = refined
        resid = circle_residuals((cx_ref, cy_ref, r_ref),
                                 points[:,0], points[:,1])
        resid_sum = np.sum(resid**2)
        
        if resid_sum < best_resid_sum:
            best_resid_sum = resid_sum
            best_circle = (cx_ref, cy_ref, r_ref)

    return best_circle

###############################################################################
# Background-Removal Pipeline for a Single Image
###############################################################################
def process_image(img_full, raw_mask, args):
    """
    Given one raw image and its corresponding mask,
    find ring(s), fit circle(s), choose median center,
    then subtract radial background. Return the background-subtracted image.
    """

    # -------------------------------------------------------------------------
    # Apply mask (invalid => 0 => NaN)
    # -------------------------------------------------------------------------
    img_masked = img_full * raw_mask
    img_masked[img_masked == 0] = np.nan

    valid_vals = img_masked[~np.isnan(img_masked)]
    if valid_vals.size < 10:
        # Not enough valid pixels => just return the original
        return img_masked

    # -------------------------------------------------------------------------
    # Determine global lower and upper intensity thresholds
    # -------------------------------------------------------------------------
    global_low_val = np.nanpercentile(valid_vals, args.threshold_percentile)
    global_high_val = np.nanpercentile(valid_vals, args.exclude_top_percentile)
    if global_low_val >= global_high_val:
        # No meaningful range => return unmodified
        return img_masked

    # -------------------------------------------------------------------------
    # Successive ring fitting over expanded intensity intervals
    # -------------------------------------------------------------------------
    all_fitted_circles = []
    current_lower = global_low_val
    epsilon = 1e-6
    step = 1.0

    while current_lower < global_high_val:
        current_upper = current_lower + step

        # Expand until we get at least 3 pixels or reach top
        while True:
            if current_upper > global_high_val:
                current_upper = global_high_val
            ring_mask = (img_masked >= current_lower) & (img_masked <= current_upper)
            y_coords, x_coords = np.nonzero(ring_mask)
            if len(x_coords) >= 3 or current_upper >= global_high_val:
                break
            else:
                current_upper += step

        if len(x_coords) < 3:
            # Not enough to fit a circle
            break

        # RANSAC fit
        fit_result = ransac_circle_fit(
            x_coords, y_coords,
            max_iterations=args.ransac_iterations,
            r_min=args.min_radius,
            r_max=args.max_radius
        )
        if fit_result is not None:
            cx, cy, r = fit_result
            all_fitted_circles.append((cx, cy, r, current_lower, current_upper))

        if current_upper >= global_high_val:
            break
        current_lower = current_upper + epsilon

    if len(all_fitted_circles) == 0:
        # No valid circle => can't do radial subtraction
        return img_masked

    # -------------------------------------------------------------------------
    # Combine rings => use median circle (cx, cy, r)
    # -------------------------------------------------------------------------
    cx_vals = [c[0] for c in all_fitted_circles]
    cy_vals = [c[1] for c in all_fitted_circles]
    r_vals  = [c[2] for c in all_fitted_circles]

    median_cx = np.median(cx_vals)
    median_cy = np.median(cy_vals)
    median_r  = np.median(r_vals)

    # -------------------------------------------------------------------------
    # Radial background removal using the median center
    # -------------------------------------------------------------------------
    center_yx = (median_cy, median_cx)  # (y, x)

    # 1) Compute radial profile
    nbins = 300
    r_bins, rad_median = radial_profile(img_masked, center_yx, nbins=nbins)

    # 2) Smooth it if desired
    sigma = 1
    rad_median_smooth = gaussian_filter1d(rad_median, sigma=sigma, mode='nearest')

    # 3) Subtract
    img_subtracted = subtract_radial_background(img_masked, center_yx, r_bins, rad_median_smooth)

    return img_subtracted

def process_batch(batch_indices, dset_in, raw_mask, args):
    """
    Process a batch of images from the dataset.
    Returns: list of (index, processed_image) tuples
    """
    results = []
    for idx in batch_indices:
        img = dset_in[idx].astype(float)
        img_sub = process_image(img, raw_mask, args)
        results.append((idx, img_sub))
    return results

###############################################################################
# Main Script
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Find beam center + subtract radial background for ALL images in an HDF5 file (no plotting)."
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
        "--threshold_percentile", type=float, default=99.0,
        help="Lower percentile for starting intensity threshold (e.g., 99%)"
    )
    parser.add_argument(
        "--exclude_top_percentile", type=float, default=99.1,
        help="Exclude pixels above this upper percentile"
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
    parser.add_argument(
        "--n_processes", type=int, default=mp.cpu_count(),
        help="Number of processes to use for parallel processing"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Number of images to process in each batch"
    )
    args = parser.parse_args()

    # Open the original data file
    with h5py.File(args.datafile, "r") as hf_in:
        dset_in = hf_in["entry"]["data"]["images"]
        n_images, height, width = dset_in.shape
        print(f"Input file has {n_images} images, each {height}x{width}.")

        # Load mask
        with h5py.File(args.maskfile, "r") as hf_mask:
            raw_mask = hf_mask["/mask"][:].astype(float)
            if raw_mask.shape != (height, width):
                raise ValueError("Mask shape does not match image shape!")
            print(f"Mask loaded: shape={raw_mask.shape}")

        # Prepare output file
        out_filename = args.datafile.replace(".h5", "_background_removed.h5")
        if out_filename == args.datafile:
            out_filename = args.datafile + "_background_removed.h5"
        print(f"Writing output file => {out_filename}")

        with h5py.File(out_filename, "w") as hf_out:
            # Create the same group structure
            grp_entry = hf_out.create_group("entry")
            grp_data = grp_entry.create_group("data")
            dset_out = grp_data.create_dataset(
                "images",
                shape=(n_images, height, width),
                # dtype="float32",
                # compression="gzip"
            )

            # Prepare batches
            batch_size = args.batch_size
            n_batches = (n_images + batch_size - 1) // batch_size
            batches = [
                range(i * batch_size, min((i + 1) * batch_size, n_images))
                for i in range(n_batches)
            ]

            # Process batches in parallel
            process_func = partial(process_batch, dset_in=dset_in, raw_mask=raw_mask, args=args)
            
            print(f"Processing {n_images} images using {args.n_processes} processes...")
            with mp.Pool(processes=args.n_processes) as pool:
                # Process batches with progress bar
                results = []
                for batch_results in tqdm(
                    pool.imap(process_func, batches),
                    total=len(batches),
                    desc="Processing batches"
                ):
                    # Write results to output file
                    for idx, img_sub in batch_results:
                        dset_out[idx] = img_sub.astype("float32")

    print("Done. Background-subtracted dataset saved.")

if __name__ == "__main__":
    main()
