# centerrefinement_definitions.py

import os
import h5py
import fnmatch
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def find_friedel_pairs(peak_positions, current_center, tolerance):
    deviations = []
    for peak1 in peak_positions:
        for peak2 in peak_positions:
            deviation = peak1 + peak2 - 2 * np.array(current_center)
            if np.linalg.norm(deviation) < tolerance:
                deviations.append(deviation)
    return deviations

def refine_center_with_friedel(min_peaks, tolerance, resolution_limit, workingfile):
    all_deviations = []
    all_indices = []
    
    #with h5py.File(h5file_path, 'r') as workingfile:
    center_x = workingfile['entry/data/center_x'][:]
    center_y = workingfile['entry/data/center_y'][:]
        
    for i in range(len(workingfile['entry/data/index'])):
        num_peaks = workingfile['entry/data/nPeaks'][i]
            
        if num_peaks < min_peaks:
            continue
                
        peak_xpos_raw = workingfile['entry/data/peakXPosRaw'][i][:num_peaks]
        peak_ypos_raw = workingfile['entry/data/peakYPosRaw'][i][:num_peaks]
        peak_positions = np.column_stack((peak_xpos_raw, peak_ypos_raw))
            
        current_center = [center_x[i], center_y[i]]
            
       # Filter out peaks based on resolution_limit
        distances = np.linalg.norm(peak_positions - current_center, axis=1)
        peak_positions = peak_positions[distances < resolution_limit]
            
        index = workingfile['entry/data/index'][i]
            
        deviations = find_friedel_pairs(peak_positions, current_center, tolerance)
        if deviations:
            all_deviations.extend(deviations)
            all_indices.extend([index]*len(deviations))
            
    return np.array(all_indices), np.array(all_deviations)

def perform_lowess(indices, deviations, frac=0.1):
    if deviations.ndim != 2 or deviations.shape[1] != 2:
        raise ValueError
    
    lowess = sm.nonparametric.lowess
    z_x = lowess(deviations[:, 0], indices, frac=frac)
    z_y = lowess(deviations[:, 1], indices, frac=frac)
    return z_x, z_y

def plot_lowess_and_centers(lowess_fit_x, lowess_fit_y, original_center_x, original_center_y, existing_indices, itcount):
    print('plotting LOWESS fit')
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle("Center coordinates (it" + str(itcount) + ")")

    # Plot for X-coordinate
    ax1 = axes[0]
    ax1.plot(existing_indices, original_center_x, label="Previous Center X", marker='o')
    ax1.set_xlabel("Index")
    ax1.set_ylabel("X-coordinate")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.scatter(lowess_fit_x[:, 0], lowess_fit_x[:, 1], label="LOWESS fit for X-deviation", color='r', s=0.5)
    ax2.set_ylabel("LOWESS X-deviation")
    ax2.legend(loc='upper right')

    # Plot for Y-coordinate
    ax1 = axes[1]
    ax1.plot(existing_indices, original_center_y, label="Previous Center Y", marker='o')
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Y-coordinate")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.scatter(lowess_fit_y[:, 0], lowess_fit_y[:, 1], label="LOWESS fit for Y-deviation", color='g', s=0.5)
    ax2.set_ylabel("LOWESS Y-deviation")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_deviations(indices, deviations):
    # Perform LOWESS fit
    z_x, z_y = perform_lowess(indices, deviations)

    plt.figure(figsize=(16, 6))
    plt.scatter(indices, deviations[:, 0], label="X-deviation", marker='o', s=0.3)
    plt.scatter(indices, deviations[:, 1], label="Y-deviation", marker='x', s=0.3)
    
    # Plot LOWESS fits
    plt.plot(z_x[:, 0], z_x[:, 1], label="LOWESS fit for X-deviation", color='r')
    plt.plot(z_y[:, 0], z_y[:, 1], label="LOWESS fit for Y-deviation", color='g')

    plt.xlabel("Frame")
    plt.ylabel("Deviation")
    plt.legend()
    plt.show()
    plt.close()

def update_detector_shifts(h5file_path, updated_center_x, updated_center_y, framesize):
    pixels_per_meter = 17857.14285714286
    presumed_center = framesize / 2  # Half of the framesize
    det_shift_x_mm = -((updated_center_x - presumed_center) / pixels_per_meter) * 1000
    det_shift_y_mm = -((updated_center_y- presumed_center) / pixels_per_meter) * 1000
     
    # Write the updated centers to the HDF5 file
    with h5py.File(h5file_path, 'r+') as workingfile:
        for ds_name in ['entry/data/det_shift_x_mm', 'entry/data/det_shift_y_mm']:
            if ds_name in workingfile:
                del workingfile[ds_name]
        
        workingfile.create_dataset('entry/data/det_shift_x_mm', data=det_shift_x_mm, maxshape=(None,), dtype='float64')
        workingfile.create_dataset('entry/data/det_shift_y_mm', data=det_shift_y_mm, maxshape=(None,), dtype='float64')

    print('Updated detector shifts written to HDF5 file')

def refine(h5file_path, min_peaks, tolerance, resolution_limit, itcount, convergence_threshold, converged):
    # Open the HDF5 file to update the center_x and center_y datasets
    with h5py.File(h5file_path, 'r+') as workingfile:
        # Read existing center_x, center_y, and index
        existing_center_x = workingfile['entry/data/center_x'][:]
        existing_center_y = workingfile['entry/data/center_y'][:]
        existing_indices = workingfile['entry/data/index'][:]

        indices, deviations = refine_center_with_friedel(min_peaks, tolerance, resolution_limit, workingfile)
    
        # Perform LOWESS fit
        lowess_x, lowess_y = perform_lowess(indices, deviations)

        # Create interpolation functions for the LOWESS fit
        interp_lowess_x = interp1d(lowess_x[:, 0], lowess_x[:, 1], bounds_error=False, fill_value="extrapolate")
        interp_lowess_y = interp1d(lowess_y[:, 0], lowess_y[:, 1], bounds_error=False, fill_value="extrapolate")

        # Update centers based on LOWESS-smoothed deviations
        updated_center_x = existing_center_x + 0.5 * interp_lowess_x(existing_indices)
        updated_center_y = existing_center_y + 0.5 * interp_lowess_y(existing_indices)
        updated_center_x[0] = updated_center_x[1]
        updated_center_y[0] = updated_center_y[1]  

        plot_lowess_and_centers(lowess_x, lowess_y, existing_center_x, existing_center_y, existing_indices, itcount)

        # Update the HDF5 datasets
        workingfile['entry/data/center_x'][:] = updated_center_x
        workingfile['entry/data/center_y'][:] = updated_center_y

    # Check for convergence for each point
    convergence_x = all(np.abs(lowess_x[:, 1]) < convergence_threshold)
    convergence_y = all(np.abs(lowess_y[:, 1]) < convergence_threshold)
    
    plot_deviations(indices, deviations)

    if convergence_x and convergence_y:
        converged = True
        
    return indices, deviations, converged, itcount, updated_center_x, updated_center_y

def refine_center_and_update(h5file_path, tolerance, min_peaks, resolution_limit, max_iterations, convergence_threshold):
    framepath = 'entry/data/images'
        # Open HDF5 file to get total number of indices
    with h5py.File(h5file_path, 'r') as workingfile:
        _,framesize_x,framesize_y = workingfile[framepath].shape

    framesize = framesize_x

    print(f"Working with {os.path.basename(h5file_path)}")

    itcount = 0
    converged = False

    while itcount < max_iterations and not converged:
        itcount += 1
        indices, deviations, converged, itcount, updated_center_x, updated_center_y = refine(h5file_path, min_peaks, tolerance, resolution_limit, itcount, convergence_threshold, converged)
    
    if converged:
        print(f'Convergence criterion (deviation from LOWESS < {convergence_threshold}) met after {itcount} iteration(s)')
    else:
        print(f'Could not meet convergenc criterion ({convergence_threshold}) after {itcount} iterations, refinement terminated')
        
    update_detector_shifts(h5file_path, updated_center_x, updated_center_y, framesize)

def find_files_and_run_centerrefinement(folder_path, tolerance, min_peaks, resolution_limit, max_iterations, convergence_threshold):
    # Only list files in the given folder, no subdirectories
    for filename in fnmatch.filter(os.listdir(folder_path), '*.h5'):
        filepath = os.path.join(folder_path, filename)
        # Call your center refinement function
        refine_center_and_update(filepath, tolerance, min_peaks, resolution_limit, max_iterations, convergence_threshold)
