# %%

import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import permutations

def read_coordinates(file_name, start_keyword, end_keyword):
    coordinates = []
    reading = False

    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(end_keyword):
                break
            if reading and line and not line.startswith(start_keyword):
                entries = line.split()
                if len(entries) > 4:  # Ensure there are enough entries in the line
                    try:
                        coords = [float(entries[2]), float(entries[3]), float(entries[4])]
                        coordinates.append(coords)
                    except ValueError:
                        continue
            if line.startswith(start_keyword):
                reading = True

    return np.array(coordinates)

def compare_matrices(matrix1, matrix2, large_cost=1e6):
    n, m = matrix1.shape
    n2 = matrix2.shape[0]
    cost_matrix = np.zeros((n2, n2))

    for i in range(n2):
        for j in range(n2):
            if i < n and j < n:
                diff_matrix = np.minimum(np.abs(matrix1[i] - matrix2[j]), np.abs(1 - matrix1[i] - matrix2[j]))
                cost_matrix[i, j] = np.sum(diff_matrix)
            else:
                # Assign a large finite cost for dummy rows
                cost_matrix[i, j] = large_cost

    return cost_matrix

# Function to drop rows with NaN values
def drop_nan_rows(matrix):
    return matrix[~np.isnan(matrix).any(axis=1)]

def calculate_differences(matrix1, matrix2):
    diff = drop_nan_rows(matrix1-matrix2)
    summ = drop_nan_rows(matrix1+matrix2)
    differences = np.minimum(np.abs(diff), np.abs(1 - summ))
    return differences

if __name__ == "__main__":
    # File paths
    file_name_res_orig = input('Enter the file path to the original .res file: ')
    file_name_res = input('Enter the file path to the .res file with which you wish to compare: ')

    # Read coordinates and extract relevant columns
    coords_res_orig = read_coordinates(file_name_res_orig, 'PLAN', 'HKLF')
    coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF')

    min_cost = float('inf')
    best_permutation_orig = None
    best_permutation_filtered = None

    # Consider all column permutations
    for perm in permutations(range(3)):
        permuted_coords_res = coords_res[:, perm]

        # Compute cost matrix
        cost_matrix = compare_matrices(coords_res_orig, permuted_coords_res)

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Get the cost for this permutation
        cost = cost_matrix[row_ind, col_ind].sum()

        if cost < min_cost:
            min_cost = cost
            best_permutation_orig = np.pad(coords_res_orig[row_ind[row_ind < len(coords_res_orig)]], ((0, len(row_ind) - len(coords_res_orig)), (0, 0)), 'constant', constant_values=np.nan)
            best_permutation_filtered = permuted_coords_res[col_ind]

    # Calculate differences
    differences = calculate_differences(best_permutation_orig, best_permutation_filtered)

    # Display results
    print(f"Minimum cost (sum of minimized differences): {min_cost}")
    print("Original .res coords:")
    print(best_permutation_orig)
    print("Best fit after filtering .res coords:")
    print(best_permutation_filtered)
    print("Differences between original and best fit:")
    print(differences)
    print("Mean of difference:")
    print(np.mean(differences))
# %%
