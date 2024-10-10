# import numpy as np

# # Function to calculate weighted RMSD between peaks and reflections
# def calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss):
#     total_rmsd = 0
#     total_weight = 0
#     for (fs, ss), intensity in zip(fs_ss, intensities):
#         min_distance = float('inf')
#         for ref_fs, ref_ss in ref_fs_ss:
#             distance = np.sqrt((fs - ref_fs) ** 2 + (ss - ref_ss) ** 2)
#             if distance < min_distance:
#                 min_distance = distance
#         total_rmsd += (min_distance ** 2) * intensity
#         total_weight += intensity

#     return np.sqrt(total_rmsd / total_weight) if total_weight > 0 else float('inf')

import numpy as np

# Function to calculate weighted RMSD between peaks and reflections, with outlier rejection
def calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss, tolerance_factor = 2.0):
    distances = []
    weights = []

    # Calculate the minimum distance for each peak to a reflection
    for (fs, ss), intensity in zip(fs_ss, intensities):
        min_distance = float('inf')
        for ref_fs, ref_ss in ref_fs_ss:
            distance = np.sqrt((fs - ref_fs) ** 2 + (ss - ref_ss) ** 2)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
        weights.append(intensity)

    # Convert to numpy arrays for easier manipulation
    distances = np.array(distances)
    weights = np.array(weights)

    # Calculate the mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Identify inliers based on the tolerance factor
    inliers = distances < (mean_distance + tolerance_factor * std_distance)

    # Calculate weighted RMSD using only inliers
    total_rmsd = np.sum((distances[inliers] ** 2) * weights[inliers])
    total_weight = np.sum(weights[inliers])

    return np.sqrt(total_rmsd / total_weight) if total_weight > 0 else float('inf')