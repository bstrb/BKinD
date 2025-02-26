#!/usr/bin/env python3
"""
Refine Center by Angle-Binned Radial Profiles
=============================================

This script:
1. Loads a 3D stack of electron diffraction images from:
   /Users/xiaodong/Desktop/UOXs-3/UOXs.h5 at /entry/data/images
2. Loads a 2D mask from:
   /Users/xiaodong/mask/pxmask.h5 at /mask
3. For each image:
   - Computes an initial center guess via weighted centroid (ignoring masked pixels).
   - Refines that guess by binning the image by angle (N_ANGLE_BINS slices),
     computing radial mean profiles for each angular slice, and measuring how
     similar those profiles are to each other. Minimizes that difference.
4. Saves refined center coordinates to a CSV file.
5. Optionally plots intermediate steps for debugging and verification.

Dependencies
------------
- numpy, scipy, h5py, matplotlib

Usage
-----
python refine_center_by_angle.py

"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d

###############################
#       CONFIG SECTION
###############################
# HDF5 file with a stack of images (shape ~ (N, height, width))
H5_IMAGE_FILE = "/Users/xiaodong/Desktop/UOXs-3/UOXs.h5"
H5_IMAGE_PATH = "/entry/data/images"  # e.g. shape (100, 1024, 1024)

# Mask file + dataset
H5_MASK_FILE  = "/Users/xiaodong/mask/pxmask.h5"
H5_MASK_PATH  = "/mask"  # shape should match (1024, 1024) for each image

# Number of images expected to process. If None, will process all in the dataset.
NUM_IMAGES_TO_PROCESS = None  # or an integer

# Number of angle bins to slice the image. e.g. 36 => each bin spans 10 degrees
N_ANGLE_BINS = 3

# Number of radial bins for each angle slice
N_RADIAL_BINS = 200

# Optionally limit the maximum radius for the radial profile
# e.g. if you only trust the central ~300 px for center-finding
MAX_RADIUS = None  # or an integer (e.g. 300)

# If desired, apply 1D smoothing to each radial mean profile
RADIAL_SMOOTH_SIGMA = 2.0  # set to 0 or None to disable smoothing

# For large images, we might clamp the center to remain within [0..width, 0..height]
# or we can let it float. We'll do the bounding check in the objective function.
BIG_COST = 1e15

# Where to save the final center coordinates
OUTPUT_CSV = "refined_centers_anglebins.csv"

# If True, produce debug plots for each image
SAVE_DEBUG_PLOTS = True


def main():
    """
    Main routine that:
      1. Loads images + mask
      2. Loops over images
      3. Finds/prints initial centroid
      4. Refines using angle-binned radial approach
      5. Saves results
    """
    print(f"Loading images from {H5_IMAGE_FILE} at {H5_IMAGE_PATH}...")
    images = load_h5_dataset(H5_IMAGE_FILE, H5_IMAGE_PATH)
    n_images, height, width = images.shape
    print(f"  -> Loaded {n_images} images of shape ({height}, {width})")

    if NUM_IMAGES_TO_PROCESS is not None:
        n_images = min(n_images, NUM_IMAGES_TO_PROCESS)
        images = images[:n_images]

    print(f"Loading mask from {H5_MASK_FILE} at {H5_MASK_PATH}...")
    mask_2d = load_h5_dataset(H5_MASK_FILE, H5_MASK_PATH).astype(bool)
    if mask_2d.shape != (height, width):
        raise ValueError(f"Mask shape {mask_2d.shape} != image shape {(height, width)}")

    # Prepare array to store final refined centers
    refined_centers = np.zeros((n_images, 2), dtype=float)

    for i in range(n_images):
        print(f"\n=== Processing image {i+1}/{n_images} ===")
        img = images[i].astype(np.float64)

        # 1) Weighted centroid ignoring the mask
        cx0, cy0 = compute_weighted_centroid(img, mask_2d)
        print(f"  Initial center guess = ({cx0:.3f}, {cy0:.3f})")

        # 2) Refine
        initial_center = (cx0, cy0)
        refined = refine_center_angle(
            image=img,
            mask=mask_2d,
            initial_center=initial_center,
            n_angle_bins=N_ANGLE_BINS,
            n_r_bins=N_RADIAL_BINS,
            r_max=MAX_RADIUS,
            radial_smooth_sigma=RADIAL_SMOOTH_SIGMA
        )
        print(f"  Refined center = ({refined[0]:.3f}, {refined[1]:.3f})")

        refined_centers[i, :] = refined

        # 3) Debug plotting
        if SAVE_DEBUG_PLOTS:
            debug_plot_angle_slices(img, mask_2d,
                                    center_init=initial_center,
                                    center_refined=refined,
                                    angle_bins=N_ANGLE_BINS,
                                    r_bins=N_RADIAL_BINS,
                                    r_max=MAX_RADIUS,
                                    radial_smooth_sigma=RADIAL_SMOOTH_SIGMA,
                                    out_png=f"anglebin_debug_{i+1:03d}.png")

    # Save to CSV
    save_results_to_csv(refined_centers, OUTPUT_CSV)
    print(f"\nFinished! Refined centers saved to {OUTPUT_CSV}")
    print("Sample refined centers:", refined_centers[:5])


###############################
#    HELPER / UTILITY FNS
###############################

def load_h5_dataset(h5_path, internal_path):
    """Load a dataset from an HDF5 file."""
    import h5py
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Cannot find file: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        if internal_path not in f:
            raise KeyError(f"Path '{internal_path}' not found in HDF5 file.")
        data = f[internal_path][:]
    return data


def compute_weighted_centroid(image, mask):
    """
    Weighted centroid ignoring masked pixels. Returns (cx, cy) in (x, y) format.
    """
    h, w = image.shape
    valid = mask
    y_inds, x_inds = np.indices((h, w))
    total_intensity = np.sum(image[valid])
    if total_intensity < 1e-12:
        print("  [Warning] Weighted centroid can't be computed (mask covers everything?). Returning (0,0).")
        return (0.0, 0.0)
    cx = np.sum(x_inds[valid] * image[valid]) / total_intensity
    cy = np.sum(y_inds[valid] * image[valid]) / total_intensity
    return (cx, cy)


def refine_center_angle(image, mask, initial_center,
                        n_angle_bins=36, n_r_bins=200,
                        r_max=None, radial_smooth_sigma=2.0):
    """
    Refine center by angle-binned radial profiles. The objective:
      1. For each angle bin, compute radial MEAN (or median) profile wrt candidate center.
      2. Compare all angle-binned radial profiles. If the center is correct, they match well.
      3. Sum-of-squared-differences from the average profile is the cost.

    We do:
      cost = sum_over_angle [ sum_{r} (profile(angle) - profile_mean)^2 ]

    Minimizing cost => all angle slices share the same radial distribution => good center.

    Returns (cx, cy).
    """
    from scipy.optimize import minimize

    def objective_center(center):
        cx, cy = center
        # Bound check
        h, w = image.shape
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return BIG_COST

        angle_profiles = compute_angle_binned_radial_profiles(
            image, mask, center=(cx, cy),
            n_angle_bins=n_angle_bins,
            n_r_bins=n_r_bins,
            r_max=r_max
        )
        if angle_profiles is None:
            return BIG_COST

        # Optionally smooth each angle slice in radial dimension
        if radial_smooth_sigma and radial_smooth_sigma > 0:
            for a in range(n_angle_bins):
                angle_profiles[a,:] = gaussian_filter1d(angle_profiles[a,:], sigma=radial_smooth_sigma)

        # cost = sum of squared differences from the average
        cost_val = compare_angle_profiles(angle_profiles)
        return cost_val

    # Minimize using Nelder-Mead
    result = minimize(
        objective_center,
        x0=initial_center,
        method="Nelder-Mead",
        options={"maxiter": 300, "disp": True}
    )

    return result.x  # (cx, cy)


def compute_angle_binned_radial_profiles(image, mask, center,
                                         n_angle_bins=36, n_r_bins=200, r_max=None):
    """
    Slices the image into n_angle_bins by polar angle (0..2pi).
    For each angle slice, we compute a radial mean profile (n_r_bins).
    Returns an array shape (n_angle_bins, n_r_bins).

    If there's insufficient valid data, returns None.
    """
    h, w = image.shape
    cx, cy = center

    y_inds, x_inds = np.indices((h, w))
    rx = x_inds - cx
    ry = y_inds - cy

    # Radius and angle
    r_map = np.sqrt(rx*rx + ry*ry)
    theta_map = np.arctan2(ry, rx)  # range ~ (-pi, pi)

    if r_max is not None:
        valid = (mask) & (r_map <= r_max)
    else:
        valid = mask

    # Flatten valid region
    r_vals = r_map[valid].flatten()
    theta_vals = theta_map[valid].flatten()
    intens_vals = image[valid].flatten()

    if len(r_vals) < 10:
        return None

    # We want angles in [0..2*pi). Let's shift
    theta_vals[theta_vals < 0] += 2*np.pi

    # Bin angles
    angle_edges = np.linspace(0, 2*np.pi, n_angle_bins+1, endpoint=True)

    # We'll build up radial mean profiles for each angle bin
    angle_profiles = np.zeros((n_angle_bins, n_r_bins), dtype=float)

    # For each angle bin, let's isolate the intensities
    # then bin them radially
    for a in range(n_angle_bins):
        a_min = angle_edges[a]
        a_max = angle_edges[a+1]

        sel = (theta_vals >= a_min) & (theta_vals < a_max)
        r_slice = r_vals[sel]
        i_slice = intens_vals[sel]

        # If no data in this angle bin, skip
        if len(r_slice) < 10:
            # We'll fill this row with zeros => might cause cost to blow up
            angle_profiles[a,:] = 0.0
            continue

        # Bin radially
        r_max_eff = r_max if (r_max is not None) else r_slice.max()
        r_edges = np.linspace(0, r_max_eff, n_r_bins+1)
        radial_means, _, _ = binned_statistic(
            r_slice, i_slice, statistic="mean", bins=r_edges
        )
        angle_profiles[a,:] = radial_means

    return angle_profiles


def compare_angle_profiles(angle_profiles):
    """
    Let angle_profiles be shape (n_angle_bins, n_r_bins).
    We want them all to match. A straightforward cost:
      1) Compute the mean profile across angle_bins => shape (n_r_bins,)
      2) cost = sum_over_angle(sum_of_squares( angle_profiles[a] - mean_profile ))

    Lower => better match.

    Returns a scalar cost.
    """
    mean_profile = np.mean(angle_profiles, axis=0)  # shape (n_r_bins,)
    diffs = angle_profiles - mean_profile[np.newaxis, :]
    ssd = np.sum(diffs*diffs)  # sum of squared differences
    return ssd


def debug_plot_angle_slices(image, mask, center_init, center_refined,
                            angle_bins, r_bins, r_max,
                            radial_smooth_sigma, out_png):
    """
    Produce a debug figure:
    1) Show masked image with initial + refined center
    2) Plot average radial profile across angles for initial vs refined centers
       (just to get a sense of how they differ).
    3) Possibly also show the difference among angle slices.
    """

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    h, w = image.shape

    # (A) Image + centers
    axs[0].imshow(image*mask, cmap='gray', origin='lower', extent=(0,w,0,h))
    axs[0].plot(center_init[0], center_init[1], 'ro', label='Init Ctr')
    axs[0].plot(center_refined[0], center_refined[1], 'gx', label='Refined Ctr')
    axs[0].set_title("Masked Image + Centers")
    axs[0].legend()

    # (B) Compare the average radial profile from angle bin approach:
    #     We'll do an angle-binned radial profile for init and refined, then average across angles.
    # init
    angle_profiles_init = compute_angle_binned_radial_profiles(
        image, mask, center_init, angle_bins, r_bins, r_max
    )
    angle_profiles_ref = compute_angle_binned_radial_profiles(
        image, mask, center_refined, angle_bins, r_bins, r_max
    )
    if angle_profiles_init is not None:
        avg_init = np.mean(angle_profiles_init, axis=0)
        if radial_smooth_sigma and radial_smooth_sigma>0:
            avg_init = gaussian_filter1d(avg_init, sigma=radial_smooth_sigma)
        axs[1].plot(avg_init, 'r.-', label='Avg Profile Init')
    if angle_profiles_ref is not None:
        avg_ref = np.mean(angle_profiles_ref, axis=0)
        if radial_smooth_sigma and radial_smooth_sigma>0:
            avg_ref = gaussian_filter1d(avg_ref, sigma=radial_smooth_sigma)
        axs[1].plot(avg_ref, 'g.-', label='Avg Profile Refined')

    axs[1].set_xlabel("Radial bin index")
    axs[1].set_ylabel("Mean Intensity (across angle slices)")
    axs[1].set_title("Angle-Binned Radial Profiles")
    axs[1].legend()

    fig.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print(f"  [debug_plot] Saved angle-bin refinement figure => {out_png}")


def save_results_to_csv(refined_centers, csv_file):
    """
    Save refined centers to CSV with columns: image_index, center_x, center_y
    """
    n = len(refined_centers)
    idxs = np.arange(n)
    arr_out = np.column_stack([idxs, refined_centers[:,0], refined_centers[:,1]])
    header_str = "image_index,center_x,center_y"
    np.savetxt(csv_file, arr_out, delimiter=",", header=header_str, comments="", fmt="%.6f")


#####################################
# Main Script Entry Point
#####################################
if __name__ == "__main__":
    main()
