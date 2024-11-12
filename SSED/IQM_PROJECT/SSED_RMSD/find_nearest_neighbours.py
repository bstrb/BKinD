# find_nearest_neighbours.py

import numpy as np
from scipy.spatial import KDTree

def find_nearest_neighbours(peaks, reflections, n=50):
    if not peaks or not reflections:
        # print("No peaks or reflections available for RMSD calculation")
        return None

    if len(peaks) > n:
        peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:n]
    peak_coords = np.array([(peak[0], peak[1]) for peak in peaks])
    reflection_coords = np.array([(reflection[0], reflection[1]) for reflection in reflections])

    if peak_coords.size == 0 or reflection_coords.size == 0:
        # print("Empty peak or reflection coordinates")
        return None

    # print(f"Number of peaks: {len(peaks)}, Number of reflections: {len(reflections)}")
    # print(f"Peak coordinates: {peak_coords[:5]}")  # Print first 5 peak coordinates
    # print(f"Reflection coordinates: {reflection_coords[:5]}")  # Print first 5 reflection coordinates

    tree = KDTree(reflection_coords)
    distances, _ = tree.query(peak_coords)
    rmsd = np.sqrt(np.mean(distances**2))
    # print(f"Calculated RMSD: {rmsd}")
    return rmsd