import numpy as np

# Function to calculate normalized deviation from target cell parameters
def calculate_cell_deviation(cell_params, target_params):
    # Extract cell parameters
    a, b, c, al, be, ga = cell_params
    target_a, target_b, target_c, target_al, target_be, target_ga = target_params

    # Normalize deviations by dividing by target values to make them dimensionless
    length_deviation = np.sqrt(((a - target_a) / target_a) ** 2 +
                               ((b - target_b) / target_b) ** 2 +
                               ((c - target_c) / target_c) ** 2)

    angle_deviation = np.sqrt(((al - target_al) / target_al) ** 2 +
                              ((be - target_be) / target_be) ** 2 +
                              ((ga - target_ga) / target_ga) ** 2)
    
    # Combine deviations considering different scales (lengths and angles are now dimensionless)
    total_deviation = length_deviation + angle_deviation
    return total_deviation

# # Example usage
# cell_params = [7.89618, 9.52121, 10.61437, 91.07093, 89.84015, 89.81503]
# target_params = [8.0, 9.5, 10.6, 90.0, 90.0, 90.0]
# deviation = calculate_cell_deviation(cell_params, target_params)
# print(f"Deviation from target cell parameters: {deviation:.5f}")
