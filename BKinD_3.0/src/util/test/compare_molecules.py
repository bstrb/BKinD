# %%

from read_coordinates import read_coordinates

import numpy as np
from itertools import permutations

def periodic_difference(col1, col2):
    diff = np.abs(np.round(col1 - col2, 4))
    p_diff = np.abs(np.round(1 - col1 - col2, 4))
    return np.minimum(diff, p_diff)

def sum_of_differences(matrix1, matrix2):
    total_diff = 0
    for i in range(matrix1.shape[1]):
        diff_matrix = periodic_difference(matrix1[:, i], matrix2[:, i])
        col_sum_diff = np.sum(diff_matrix)
        total_diff += col_sum_diff
        # print(f"Column {i}: sum_diff = {col_sum_diff:.4f}")
    return total_diff

def find_best_match_columnwise(matrix1, matrix2):
    n_cols = matrix1.shape[1]
    min_total_diff = np.inf
    best_permutation = None
    
    for perm in permutations(range(n_cols)):
        permuted_matrix2 = matrix2[:, perm]
        total_diff = sum_of_differences(matrix1, permuted_matrix2)
        # print(f"Permutation {perm}: total_diff = {total_diff:.4f}")
        
        if total_diff < min_total_diff:
            min_total_diff = total_diff
            best_permutation = perm
    
    return best_permutation, min_total_diff

def compare_matrices(matrix1, matrix2, tolerance=0.1):
    best_permutation, total_sum_diff = find_best_match_columnwise(matrix1, matrix2)
    permuted_matrix2 = matrix2[:, best_permutation]
    
    print(f"Total sum of differences: {total_sum_diff:.4f}")
    print(f"Best permutation: {best_permutation}")
    print("Best matches (column-wise):")
    for i, j in enumerate(best_permutation):
        col_from_A = matrix1[:, i]
        col_from_B = permuted_matrix2[:, i]
        diff_matrix = periodic_difference(col_from_A, col_from_B)
        print(f"Column from A: {col_from_A}, Column from B: {col_from_B}, Difference: {diff_matrix}, Sum of Differences: {np.sum(diff_matrix):.4f}")
    
    return total_sum_diff < tolerance

# # Example usage
# coords1 = np.array([
#     [0.5000, 0.3200, 0.8700],
#     [0.5000, 0.2800, 1.0000],
#     [0.5000, 0.2000, 0.8000],
#     [0.6100, 0.3900, 0.8400]
# ])

# coords2 = np.array([
#     [0.3200, 0.1200, 0.5000],
#     [0.2800, 0.0000, 0.5800],
#     [0.2000, 0.2000, 0.5000],
#     [0.3900, 0.1600, 0.6100]
# ])

# # Check if the coordinates are equivalent under periodic boundary conditions
# are_same = compare_matrices(coords1, coords2, tolerance=0.1)

# print(f"Are the molecules the same? {are_same}")

# Example usage for solve_filtered.ins
file_name_ins = '/mnt/c/Users/bubl3932/Desktop/bkind_LTA_to_96.0_completeness/filtered_96.0/solve_filtered/solve_filtered.ins'
coords_ins = read_coordinates(file_name_ins, 'FVAR', 'HKLF 4')
print("Coordinates from solve_filtered.ins:")
print(coords_ins)

# Example usage for solve_filtered_a.res
file_name_res = '/mnt/c/Users/bubl3932/Desktop/bkind_LTA_to_96.0_completeness/filtered_96.0/solve_filtered/solve_filtered_a.res'
coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF 4')
print("Coordinates from solve_filtered_a.res:")
print(coords_res)

compare_ins_res = compare_matrices(coords_ins, coords_res)


print(f"Are the molecules the same? {compare_ins_res}")
# %%
