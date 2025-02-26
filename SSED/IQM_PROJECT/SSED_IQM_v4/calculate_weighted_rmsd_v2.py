
import numpy as np

def calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss, tolerance_factor=2.0):
    # Ensure inputs are non-empty
    if len(fs_ss) == 0 or len(ref_fs_ss) == 0:
        print("Warning: Empty input for peaks or reflections.")
        return float('inf')  # Return infinity if no data is provided

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

    # Handle cases where distances or weights are empty or invalid
    if len(distances) == 0 or len(weights) == 0:
        # print("Warning: Distances or weights are empty.")
        return float('inf')

    # Filter out invalid weights (e.g., zero or negative values)
    valid_indices = weights > 0
    distances = distances[valid_indices]
    weights = weights[valid_indices]

    if len(distances) == 0:  # Check again after filtering
        print("Warning: No valid weights after filtering.")
        return float('inf')

    # Calculate the mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Handle cases where std_distance is zero (no variability)
    if std_distance == 0:
        print("Warning: Standard deviation of distances is zero.")
        return float('inf')

    # Identify inliers based on the tolerance factor
    inliers = distances < (mean_distance + tolerance_factor * std_distance)

    if not np.any(inliers):  # Check if there are any inliers
        print("Warning: No inliers found.")
        return float('inf')

    # Calculate weighted RMSD using only inliers
    total_rmsd = np.sum((distances[inliers] ** 2) * weights[inliers])
    total_weight = np.sum(weights[inliers])

    return np.sqrt(total_rmsd / total_weight) if total_weight > 0 else float('inf')
