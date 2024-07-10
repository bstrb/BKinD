# generate_complementary_matrices.py

import numpy as np
from itertools import product

def generate_complementary_matrices(matrix):
    rows, cols = matrix.shape
    matrices = []
    
    # Generate all combinations of original and complemented columns
    for comb in product([0, 1], repeat=cols):
        new_matrix = np.copy(matrix)
        for col_index in range(cols):
            if comb[col_index] == 1:
                new_matrix[:, col_index] = np.abs(1 - new_matrix[:, col_index])
        matrices.append(new_matrix)
    
    return matrices
