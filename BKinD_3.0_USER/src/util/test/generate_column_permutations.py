# generate_column_permutations.py

import itertools

def generate_column_permutations(matrix):
    cols = matrix.shape[1]
    col_permutations = itertools.permutations(range(cols))
    permuted_matrices = [matrix[:, perm] for perm in col_permutations]
    return permuted_matrices
