# %%

import os
import numpy as np
from itertools import permutations, combinations, product
from read_coordinates import read_coordinates

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
    return total_diff

def find_best_match(matrix1, matrix2):
    n_rows_A = matrix1.shape[0]
    n_cols_A = matrix1.shape[1]
    n_rows_B = matrix2.shape[0]
    n_cols_B = matrix2.shape[1]
    
    min_total_diff = np.inf
    best_combination = None
    best_permutation = None
    best_row_permutation = None
    
    for row_perm in permutations(range(n_rows_B), n_rows_A):
        submatrix_B_rows = matrix2[row_perm, :]
        
        for combination in combinations(range(n_cols_B), n_cols_A):
            submatrix_B = submatrix_B_rows[:, combination]
            
            for perm in permutations(range(n_cols_A)):
                permuted_matrix2 = submatrix_B[:, perm]
                total_diff = sum_of_differences(matrix1, permuted_matrix2)
                
                if total_diff < min_total_diff:
                    min_total_diff = total_diff
                    best_combination = combination
                    best_permutation = perm
                    best_row_permutation = row_perm
    
    return best_combination, best_permutation, best_row_permutation, min_total_diff

def compare_matrices(matrix1, matrix2, tolerance=0.1):
    best_combination, best_permutation, best_row_permutation, total_sum_diff = find_best_match(matrix1, matrix2)
    submatrix_B_rows = matrix2[best_row_permutation, :]
    submatrix_B = submatrix_B_rows[:, best_combination]
    permuted_matrix2 = submatrix_B[:, best_permutation]
    
    print(f"Total sum of differences: {total_sum_diff:.4f}")
    print(f"Best combination of columns from B: {best_combination}")
    print(f"Best permutation of columns: {best_permutation}")
    print(f"Best permutation of rows: {best_row_permutation}")
    print("Best matches (column-wise):")
    for i, j in enumerate(best_permutation):
        col_from_A = matrix1[:, i]
        col_from_B = permuted_matrix2[:, i]
        diff_matrix = periodic_difference(col_from_A, col_from_B)
        print(f"Column from A: {col_from_A}, Column from B: {col_from_B}, Difference: {diff_matrix}, Sum of Differences: {np.sum(diff_matrix):.4f}")
    
    return total_sum_diff < tolerance

# # Paths to the files
# path_res_orig = '/Users/xiaodong/Desktop/bkind_LTA_to_90.0_completeness/filtered_99.43'
# file_name_res_orig = os.path.join(path_res_orig, 'filtered_99.43_a.res')
# path_res = '/Users/xiaodong/Desktop/bkind_LTA_to_97.5_completeness/filtered_97.5/solve_filtered'
# file_name_res = os.path.join(path_res, 'solve_filtered_a.res')

# # Read coordinates
# coords_res_orig = read_coordinates(file_name_res_orig, 'PLAN', 'HKLF 4')
# coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF 4')

coords_res_orig = read_coordinates('/Users/xiaodong/Desktop/bkind_XRAY_feacacmerge_to_89.0_completeness/filtered_91.87/filtered_91.87_a.res', 'PLAN', 'HKLF 4')
coords_res = read_coordinates('/Users/xiaodong/Desktop/bkind_XRAY_feacacmerge_to_89.0_completeness/filtered_89.0/solve_filtered/solve_filtered_a.res', 'PLAN', 'HKLF 4')


# Print coordinates
print("Coordinates from filtered_orig_a.res:")
print(coords_res_orig)
print("Coordinates from solve_filtered_a.res:")
print(coords_res)

# Compare matrices
compare_ins_res = compare_matrices(coords_res_orig, coords_res)
print(f"Are the molecules the same? {compare_ins_res}")

# %%
