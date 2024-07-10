# %%

import os
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
    
    # print(f"Total sum of differences: {total_sum_diff:.4f}")
    # print(f"Best permutation: {best_permutation}")
    # print("Best matches (column-wise):")
    for i, j in enumerate(best_permutation):
        col_from_A = matrix1[:, i]
        col_from_B = permuted_matrix2[:, i]
        diff_matrix = periodic_difference(col_from_A, col_from_B)
        print(f"Column from A: {col_from_A},")
        print(f"Column from B: {col_from_B},")
        print(f"Difference: {diff_matrix},")
        print(f"Sum of Differences: {np.sum(diff_matrix):.4f}")
    
    return total_sum_diff < tolerance

file_name_res_orig = '/Users/xiaodong/Desktop/bkind_XRAY_feacacmerge_to_89.0_completeness/filtered_91.87/filtered_91.87_a.res'
file_name_res = '/Users/xiaodong/Desktop/bkind_XRAY_feacacmerge_to_89.0_completeness/filtered_89.0/solve_filtered/solve_filtered_a.res'


file_name_res_orig = '/Users/xiaodong/Desktop/bkind_XRAY_feacacmerge_to_89.0_completeness/filtered_91.87/filtered_91.87_a.res'
file_name_res = '/Users/xiaodong/Desktop/bkind_XRAY_feacacmerge_to_89.0_completeness/filtered_89.0/solve_filtered/solve_filtered_a.res'


coords_res_orig = read_coordinates(file_name_res_orig, 'PLAN', 'HKLF 4')
# print("Coordinates from filtered_orig_a.res:")
# print(coords_res_orig)
coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF 4')
# print("Coordinates from solve_filtered_a.res:")
# print(coords_res)

# print(len(coords_res_orig))
# print(len(coords_res))
compare_ins_res = compare_matrices(coords_res_orig, coords_res)

print(f"Are the molecules the same? {compare_ins_res}")
# %%
