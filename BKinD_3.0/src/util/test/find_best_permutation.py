# find_best_permutation.py

# %%
import numpy as np
from scipy.optimize import linear_sum_assignment

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

def compare_matrices(matrix1, matrix2):
    n, m = matrix1.shape
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff_matrix = np.minimum(np.abs(matrix1[i] - matrix2[j]), np.abs(1 - matrix1[i] - matrix2[j]))
            cost_matrix[i, j] = np.sum(diff_matrix)

    return cost_matrix

if __name__ == "__main__":
    # File paths
    file_name_res_orig = '/Users/xiaodong/Desktop/bkind_FEACAC13_to_80.0_completeness/bkind_a.res'
    file_name_res = '/Users/xiaodong/Desktop/bkind_FEACAC13_to_80.0_completeness/filtered_80.0/removed_data_80.0/removed_data_80.0_a.res'

    # Read coordinates and extract relevant columns
    coords_res_orig = read_coordinates(file_name_res_orig, 'PLAN', 'HKLF')
    coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF')

    # Compute cost matrix
    cost_matrix = compare_matrices(coords_res_orig, coords_res)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get the minimum cost
    min_cost = cost_matrix[row_ind, col_ind].sum()

    # Get the best permutations
    best_permutation_orig = coords_res_orig[row_ind]
    best_permutation_filtered = coords_res[col_ind]

    # Display results
    print(f"Minimum cost: {min_cost}")
    print("Best permutation (Original):")
    print(best_permutation_orig)
    print("Best permutation (Filtered):")
    print(best_permutation_filtered)
    
# %%
