import numpy as np

# Function to calculate weighted RMSD between peaks and reflections
def calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss):
    total_rmsd = 0
    total_weight = 0
    for (fs, ss), intensity in zip(fs_ss, intensities):
        min_distance = float('inf')
        for ref_fs, ref_ss in ref_fs_ss:
            distance = np.sqrt((fs - ref_fs) ** 2 + (ss - ref_ss) ** 2)
            if distance < min_distance:
                min_distance = distance
        total_rmsd += (min_distance ** 2) * intensity
        total_weight += intensity

    return np.sqrt(total_rmsd / total_weight) if total_weight > 0 else float('inf')
