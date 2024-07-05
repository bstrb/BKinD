# %%
import numpy as np
from scipy.spatial.distance import cdist

def apply_pbc(coords):
    return coords - np.floor(coords)

def are_coordinates_equivalent(coord1, coord2, tolerance=0.1):
    for i in range(len(coord1)):
        if not (abs(coord1[i] - coord2[i]) < tolerance or abs(coord1[i] - (1 - coord2[i])) < tolerance):
            return False
    return True

def compare_molecules(coords1, coords2, tolerance=0.1):
    # Apply periodic boundary conditions
    coords1_pbc = apply_pbc(coords1)
    coords2_pbc = apply_pbc(coords2)
    
    # Check if each atom in set 1 has a corresponding close atom in set 2
    for i in range(len(coords1_pbc)):
        found_equivalent = False
        for j in range(len(coords2_pbc)):
            if are_coordinates_equivalent(coords1_pbc[i], coords2_pbc[j], tolerance):
                found_equivalent = True
                break
        if not found_equivalent:
            return False
    return True

# Coordinates with swapped x and y, rounded to 2 decimals
coords1_swapped = np.array([
    [0.50, 0.32, 0.87],
    [0.50, 0.28, 1.00],
    [0.50, 0.20, 0.80],
    [0.61, 0.39, 0.84]
])

coords2 = np.array([
    [0.60, 0.32, 0.03],
    [0.60, 0.28, 0.00],
    [0.60, 0.30, 0.20],
    [0.71, 0.29, 0.26]
])

are_same = compare_molecules(coords1_swapped, coords2)

print(f"Are the molecules the same? {are_same}")

# %%
