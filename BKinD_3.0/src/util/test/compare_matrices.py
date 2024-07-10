# compare_matrices.py

# %%

import numpy as np

from generate_permutations import generate_permutations
from generate_complementary_matrices import generate_complementary_matrices
from subtract_and_sum_matrix_groups import subtract_and_sum_matrix_groups
from read_coordinates import read_coordinates


file_name_res_orig = '/mnt/c/Users/bubl3932/Desktop/bkind_XRAY_FEACAC_to_89.0_completeness/filtered_91.87/filtered_91.87_a.res'
file_name_res = '/mnt/c/Users/bubl3932/Desktop/bkind_XRAY_FEACAC_to_89.0_completeness/filtered_89.0/solve_filtered/solve_filtered_a.res'

file_name_res_orig = '/mnt/c/Users/bubl3932/Desktop/bkind_LTA_to_96.0_completeness/bkind_a.res'
file_name_res = '/mnt/c/Users/bubl3932/Desktop/bkind_LTA_to_96.0_completeness/filtered_96.0/solve_filtered_96.0/solve_filtered_96.5_a.res'

coords_res_orig = read_coordinates(file_name_res_orig, 'PLAN', 'HKLF 4')
coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF 4')

matrix1 = coords_res_orig
matrix2 = coords_res

# Generate all permutations of matrix1 swapping rows and columns
all_permuted_matrices = generate_permutations(matrix1)

# Generate all complementary matrices to matrix 2
complementary_matrices = generate_complementary_matrices(matrix2)

# Perform subtraction and summation of absolute differences
results, combinations = subtract_and_sum_matrix_groups(all_permuted_matrices, complementary_matrices)

# Find the minimum sum and corresponding matrix combination
min_sum = min(results)
min_index = results.index(min_sum)
min_combination = combinations[min_index]

# Display the minimum sum and the matrices that gave the minimum sum
print(f"Minimum sum of all resulting matrices: {min_sum}")
print(f"Matrix A index: {min_combination[0]}")
print(f"Matrix B index: {min_combination[1]}")
print(f"Resulting matrix:\n{min_combination[2]}")
print(f"Matrix A:\n{all_permuted_matrices[min_combination[0]]}")
print(f"Matrix B:\n{complementary_matrices[min_combination[1]]}")

# %%
