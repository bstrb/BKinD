import numpy as np

# Function to calculate deviation from target cell parameters
def calculate_cell_deviation(cell_params, target_params):
    a, b, c, al, be, ga = cell_params
    target_a, target_b, target_c, target_al, target_be, target_ga = target_params
    return np.sqrt((a - target_a) ** 2 + (b - target_b) ** 2 + (c - target_c) ** 2 +
                   (al - target_al) ** 2 + (be - target_be) ** 2 + (ga - target_ga) ** 2)
