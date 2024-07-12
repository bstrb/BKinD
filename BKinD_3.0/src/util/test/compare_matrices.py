# %%

import numpy as np
import concurrent.futures
from generate_column_permutations import generate_column_permutations
from generate_row_permutations import generate_row_permutations
from generate_complementary_matrices import generate_complementary_matrices
from read_coordinates import read_coordinates

def compare_matrices(permuted_matrix, complementary_matrix):
    return np.sum(np.abs(permuted_matrix - complementary_matrix))

if __name__ == "__main__":
    # File paths
    file_name_res_orig = '/Users/xiaodong/Desktop/bkind_FEACAC13_to_79.0_completeness/bkind_a.res'
    file_name_res = '/Users/xiaodong/Desktop/bkind_FEACAC13_to_79.0_completeness/filtered_79.0/removed_data_79.0/removed_data_79.0_a.res'

    coords_res_orig = read_coordinates(file_name_res_orig, 'PLAN', 'HKLF 4')
    coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF 4')

    matrix1 = coords_res_orig
    matrix2 = coords_res

    # Generate column permutations
    column_permutations = generate_column_permutations(matrix1)

    # Generate complementary matrices
    complementary_matrices = generate_complementary_matrices(matrix2)

    # Parallel processing of column comparisons
    column_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compare_matrices, perm, comp) for perm in column_permutations for comp in complementary_matrices]
        for future in concurrent.futures.as_completed(futures):
            column_results.append(future.result())

    # Find the best column permutation and complementary matrix combination
    min_column_sum = min(column_results)
    min_column_index = column_results.index(min_column_sum)
    best_col_perm_index = min_column_index // len(complementary_matrices)
    best_comp_matrix_index = min_column_index % len(complementary_matrices)

    best_col_permuted_matrix = column_permutations[best_col_perm_index]
    best_complementary_matrix = complementary_matrices[best_comp_matrix_index]

    # Generate row permutations for the best column-matched matrices
    # row_permutations = generate_row_permutations(best_col_permuted_matrix)

    # Compare row permutations with the best complementary matrix
    # row_results = []
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(compare_matrices, perm, best_complementary_matrix) for perm in row_permutations]
    #     for future in concurrent.futures.as_completed(futures):
    #         row_results.append(future.result())

    # Find the best row permutation
    # min_row_sum = min(row_results)
    # min_row_index = row_results.index(min_row_sum)
    # best_row_perm_index = min_row_index

    # Display the minimum sum and the matrices that gave the minimum sum
    print(f"Minimum sum of column permutations: {min_column_sum}")
    # print(f"Minimum sum of row permutations: {min_row_sum}")
    print(f"Best column-permuted matrix index: {best_col_perm_index}")
    print(f"Best complementary matrix index: {best_comp_matrix_index}")
    # print(f"Best row-permuted matrix index: {best_row_perm_index}")

    print(f"Best column-permuted matrix:\n{best_col_permuted_matrix}")
    print(f"Best complementary matrix:\n{best_complementary_matrix}")
    # print(f"Best row-permuted matrix:\n{row_permutations[best_row_perm_index]}")

# %%
