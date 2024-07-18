# subtract_and_sum_matrix_groups.py

import numpy as np

def subtract_and_sum_matrix_groups(group_A, group_B):
    results = []
    combinations = []
    for i, matrix_A in enumerate(group_A):
        for j, matrix_B in enumerate(group_B):
            result_matrix = np.abs(matrix_A - matrix_B)
            matrix_sum = np.sum(result_matrix)
            results.append(matrix_sum)
            combinations.append((i, j, result_matrix))
    return results, combinations