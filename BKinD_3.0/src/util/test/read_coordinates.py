# read_coordinates.py

# %%

import numpy as np

def read_coordinates(file_name, start_keyword, end_keyword):
    coordinates = []
    reading = False

    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(end_keyword):
                break
            if reading and line and not line.startswith(start_keyword):
                entries = line.split()
                if len(entries) > 4:  # Ensure there are enough entries in the line
                    # Convert the second, third, and fourth numerical entries to float and store them as coordinates
                    try:
                        coords = [float(entries[2]), float(entries[3]), float(entries[4])]
                        coordinates.append(coords)
                    except ValueError:
                        continue
            if line.startswith(start_keyword):
                reading = True

    return np.array(coordinates)

# Example usage for solve_filtered.ins
file_name_ins = '/mnt/c/Users/bubl3932/Desktop/bkind_LTA_to_96.0_completeness/filtered_96.0/solve_filtered/solve_filtered.ins'
coords_ins = read_coordinates(file_name_ins, 'FVAR', 'HKLF 4')
print("Coordinates from solve_filtered.ins:")
print(coords_ins)

# Example usage for solve_filtered_a.res
file_name_res = '/mnt/c/Users/bubl3932/Desktop/bkind_LTA_to_96.0_completeness/filtered_96.0/solve_filtered/solve_filtered_a.res'
coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF 4')
print("Coordinates from solve_filtered_a.res:")
print(coords_res)


# %%
