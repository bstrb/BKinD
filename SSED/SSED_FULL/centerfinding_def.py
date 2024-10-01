import os
import h5py
import fnmatch
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

def divide_into_quarters(total_indices):
    # Calculate the frame range for each quarter center
    quarter_size = len(total_indices) // 4
    remaining = len(total_indices) % 4
    
    quarters = []
    start = 0
    for i in range(4):
        end = start + quarter_size + (1 if i < remaining else 0)
        quarters.append(total_indices[start:end])
        start = end

    return quarters

def find_friedel_pairs(peak_positions, current_center, tolerance):
    deviations = []
    for peak1 in peak_positions:
        for peak2 in peak_positions:
            deviation = peak1 + peak2 - 2 * np.array(current_center)
            if np.linalg.norm(deviation) < tolerance:
                deviations.append(deviation)
    return deviations

def find_center_with_friedel(h5file_path, min_peaks, tolerance, resolution_limit, subset_indices, current_center):
    all_deviations = []
    all_indices = []

    with h5py.File(h5file_path, 'r') as workingfile:
        indices_to_process = subset_indices if subset_indices is not None else range(len(workingfile['entry/data/nPeaks']))
        for i in indices_to_process:
            num_peaks = int(workingfile['entry/data/nPeaks'][i])

            if num_peaks < min_peaks:
                continue
                
            peak_xpos_raw = workingfile['entry/data/peakXPosRaw'][i][:num_peaks]
            peak_ypos_raw = workingfile['entry/data/peakYPosRaw'][i][:num_peaks]
            peak_positions = np.column_stack((peak_xpos_raw, peak_ypos_raw))
            
            # Filter out peaks based on resolution_limit
            distances = np.linalg.norm(peak_positions - current_center, axis=1)
            peak_positions = peak_positions[distances < resolution_limit]
            
            index = workingfile['entry/data/index'][i]
            
            deviations = find_friedel_pairs(peak_positions, current_center, tolerance)
            if deviations:
                all_deviations.extend(deviations)
                all_indices.extend([index]*len(deviations))
            
    return np.array(all_indices), np.array(all_deviations)

def fit_gaussian_to_largest_cluster(deviations, min_samples_fraction):
    # Fit a Gaussian distribution to largest cluster of deviations

    # Use DBSCAN for clustering
    #clustering = DBSCAN(eps=0.1, min_samples=int(min_samples_fraction*len(deviations))).fit(deviations)
    clustering = DBSCAN(eps=0.1, min_samples=5).fit(deviations)
    cluster_labels = clustering.labels_

    # Find the cluster labels (ignoring -1, which are 'noise' points)
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # Find the largest cluster
    cluster_sizes = [np.sum(cluster_labels == i) for i in unique_labels]
    if len(cluster_sizes) == 0:
        return None, None  # Handle the case where no cluster is found
    
    largest_cluster_index = list(unique_labels)[np.argmax(cluster_sizes)]

    # Filter deviations belonging to the largest cluster
    largest_cluster_deviations = deviations[cluster_labels == largest_cluster_index]

    # Fit Gaussian to the largest cluster
    gmm = GaussianMixture(n_components=1).fit(largest_cluster_deviations)
    mean = gmm.means_[0]
    covariance = gmm.covariances_[0]

    return mean, covariance

def iterative_gaussian_fitting_on_subset(h5file_path, initial_center, subset_indices, tolerance, min_peaks, resolution_limit, quarter_index, min_samples_fraction):
    # Perform Gaussian fitting to each subset of data
    current_center = initial_center
    mean_x, mean_y = float('inf'), float('inf')
    itcount = 0
    breakcriterion = 20 # break the loop if a center wasn't found after 20 iterations

    while mean_x > 0.4 or mean_y > 0.4:
        indices, deviations = find_center_with_friedel(h5file_path, min_peaks, tolerance, resolution_limit, subset_indices, current_center)

        itcount = itcount + 1

        # Check if deviations are empty, and if so, break the loop
        if len(deviations) == 0:
            break
        
        if itcount == breakcriterion + 1:
            print(f'No valid center found in quarter {quarter_index} after {breakcriterion} iterations.')
            break

        plt.close()
    
        mean, _ = fit_gaussian_to_largest_cluster(deviations, min_samples_fraction)

        if mean is not None:
            # Update the current center
            current_center = [current_center[0] + mean[0] / 2, current_center[1] + mean[1] / 2]

            # Calculate new means
            mean_x, mean_y = abs(mean[0]), abs(mean[1])

            # Create a plot     
            plt.figure(figsize=(6, 6))
            plt.scatter(deviations[:, 0], deviations[:, 1], label="Deviation Cloud", marker='o', s=0.01)
            plt.scatter(mean_x, mean_y, color='red', label="Final Center (" + str(round(current_center[0],3)) + ", " + str(round(current_center[1],3)) + ")", marker='x')
            plt.xlabel("X-deviation")
            plt.ylabel("Y-deviation")
            plt.legend()
            plt.title("Deviation Cloud in X/Y Plane with Final Center (q" + str(quarter_index) + "/it" + str(itcount) + ")")
            plt.show()
            plt.close()
        else:
            print(f'No cluster in deviation cloud found ({quarter_index}/{itcount:0{3}d}).')

            # Create a plot     
            plt.figure(figsize=(6, 6))
            plt.scatter(deviations[:, 0], deviations[:, 1], label="Deviation Cloud", marker='o', s=0.01)
            plt.xlabel("X-deviation")
            plt.ylabel("Y-deviation")
            plt.legend()
            plt.title("Deviation Cloud in X/Y Plane with Final Center (q" + str(quarter_index) + "/it" + str(itcount) + "), no cluster found")
            plt.show()
            plt.close()
            break

    print(f'Quarter {quarter_index} processed in {itcount - 1} iterations, final center = [{round(current_center[0],3)}, {round(current_center[1],3)}], mean deviation = [{mean_x:.3e},{mean_y:.3e}]')    
      
    return current_center, mean_x, mean_y, itcount

def set_center_based_on_line_fit(h5file_path, quarter_centers, framesize, framepath, exclude=[]):
    print('Interpolating centers based on linear fit')

    with h5py.File(h5file_path, 'r') as workingfile:
        total_frames = workingfile[framepath].shape[0]
    
    # Calculate the frame range for each quarter center
    quarter_size = total_frames // 4
    quarter_positions = [(i + 0.5) * quarter_size for i in range(4)]
    
    # Exclude specified quarters
    quarter_centers = [c for i, c in enumerate(quarter_centers) if i not in exclude]
    quarter_positions = [p for i, p in enumerate(quarter_positions) if i not in exclude]
    
    # Fit a line through the remaining quarter centers
    x = np.array(quarter_positions)
    y_x = np.array([c[0] for c in quarter_centers])
    y_y = np.array([c[1] for c in quarter_centers])
    
    slope_x, intercept_x, rvalue_x, _, _ = linregress(x, y_x)
    slope_y, intercept_y, rvalue_y, _, _ = linregress(x, y_y)

    # Warn if R^2 is bad
    if rvalue_x ** 2 <= 0.25:
        print(f'Warning: poor quality of x-fit (R^2 = {rvalue_x ** 2}), review your dataset or consider excluding data')
        
    if rvalue_y ** 2 <= 0.25:
        print(f'Warning: poor quality of y-fit (R^2 = {rvalue_y ** 2}), review your dataset or consider excluding data')
        
    # Generate new centers for each frame based on the line fit
    frame_positions = np.linspace(0, total_frames, total_frames)
    updated_x = slope_x * frame_positions + intercept_x
    updated_y = slope_y * frame_positions + intercept_y

    print(f'Scope of drift: {round((max(updated_x)-min(updated_x)),3)} px in x, {round((max(updated_y)-min(updated_y)),3)} px in y')
    
    # Plot the fit
    plt.figure()
    plt.scatter(quarter_positions, [c[0] for c in quarter_centers], label='Quarter Centers X', color='red')
    plt.scatter(quarter_positions, [c[1] for c in quarter_centers], label='Quarter Centers Y', color='blue')
    plt.plot(frame_positions, updated_x, label='Fitted Line X', color='orange')
    plt.plot(frame_positions, updated_y, label='Fitted Line Y', color='green')
    plt.xlabel('Frame Position')
    plt.ylabel('Center Value')
    plt.legend()
    
    # Calculate detector shifts in mm
    pixels_per_meter = 17857.14285714286  # Given constant
    presumed_center = framesize / 2  # Given constant
    det_shift_x_mm = -((updated_x - presumed_center) / pixels_per_meter) * 1000
    det_shift_y_mm = -((updated_y - presumed_center) / pixels_per_meter) * 1000
     
    # Write the updated centers to the HDF5 file
    with h5py.File(h5file_path, 'a') as workingfile:
        for ds_name in ['entry/data/center_x', 'entry/data/center_y', 'entry/data/det_shift_x_mm', 'entry/data/det_shift_y_mm']:
            if ds_name in workingfile:
                del workingfile[ds_name]
        
        workingfile.create_dataset('entry/data/center_x', data=updated_x, maxshape=(None,), dtype='float64')
        workingfile.create_dataset('entry/data/center_y', data=updated_y, maxshape=(None,), dtype='float64')
        workingfile.create_dataset('entry/data/det_shift_x_mm', data=det_shift_x_mm, maxshape=(None,), dtype='float64')
        workingfile.create_dataset('entry/data/det_shift_y_mm', data=det_shift_y_mm, maxshape=(None,), dtype='float64')
    
    print('Interpolated detector shifts written to HDF5 file')

    plt.show()

def find_centers(h5file_path, x0, y0, tolerance, min_peaks, resolution_limit, min_samples_fraction):
    framepath = 'entry/data/images'
    # Open HDF5 file to get total number of indices
    with h5py.File(h5file_path, 'r') as workingfile:
        total_indices = np.arange(len(workingfile['entry/data/index']))
        _,framesize_x,framesize_y = workingfile[framepath].shape

    # Define parameters dependent on frame dimesions
    framesize = framesize_x
    initial_center = [x0, y0]

    # Divide total_indices into quarters
    quarters = divide_into_quarters(total_indices)

    # Process each quarter
    quarter_centers = []

    for i, subset_indices in enumerate(quarters):
        quarter_index = i+1
        final_center, mean_x, mean_y, itcount = iterative_gaussian_fitting_on_subset(h5file_path, initial_center, subset_indices, tolerance, min_peaks, resolution_limit, quarter_index, min_samples_fraction)
        quarter_centers.append(final_center)

    set_center_based_on_line_fit(h5file_path, quarter_centers, framesize, framepath, exclude=[])

def find_files_and_run_centerfinding(folder_path, x0, y0, tolerance, min_peaks, resolution_limit, min_samples_fraction):
    # Only list files in the given folder, no subdirectories
    for filename in fnmatch.filter(os.listdir(folder_path), '*.h5'):
        filepath = os.path.join(folder_path, filename)
        # Call your centerfinding function
        find_centers(filepath, x0, y0, tolerance, min_peaks, resolution_limit, min_samples_fraction)
