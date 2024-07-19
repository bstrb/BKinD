# generate_permutations.py

import itertools
import numpy as np

def generate_permutations(matrix):
    rows, cols = matrix.shape
    
    # Generate all row permutations
    row_permutations = list(itertools.permutations(matrix, rows))
    
    all_permuted_matrices = set()
    
    for row_perm in row_permutations:
        # Convert to numpy array to easily permute columns
        row_perm_matrix = np.array(row_perm)
        
        # Generate all column permutations for the current row permutation
        col_permutations = itertools.permutations(range(cols), cols)
        
        for col_perm in col_permutations:
            # Permute columns according to the current column permutation
            permuted_matrix = row_perm_matrix[:, col_perm]
            
            # Convert to tuple of tuples for immutability and to use in a set
            permuted_matrix_tuple = tuple(map(tuple, permuted_matrix))
            
            # Add the permuted matrix to the set of all permuted matrices
            all_permuted_matrices.add(permuted_matrix_tuple)
    
    # Convert set of tuples back to list of numpy arrays
    all_permuted_matrices = [np.array(matrix) for matrix in all_permuted_matrices]
    
    return all_permuted_matrices
