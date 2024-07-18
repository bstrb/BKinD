# generate_row_permutations.py

import itertools

# Function to generate row permutations
def generate_row_permutations(matrix):
    rows = matrix.shape[0]
    row_permutations = itertools.permutations(range(rows))
    permuted_matrices = [matrix[perm, :] for perm in row_permutations]
    return permuted_matrices