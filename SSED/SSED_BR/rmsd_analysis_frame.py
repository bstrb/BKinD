# rmsd_analysis_frame.py

import numpy as np
import matplotlib.pyplot as plt

from parse_stream_file import parse_stream_file
from find_nearest_neighbours import find_nearest_neighbours

def rmsd_analysis_frame(file_paths, n, target_serial_number):

    rmsd_values = {}
    x_coords = set()
    y_coords = set()

    # Loop through each stream file
    for file_path in file_paths:
        filename = file_path.split('/')[-1]  # Extract filename
        
        # Check if the filename matches the expected pattern
        if filename.count('_') < 2 or not filename.endswith('.stream'):
            print(f"Skipping file {filename} as it doesn't match the expected naming pattern.")
            continue
        
        coords = filename.split('_')[-2:]  # Extract the last two parts as coordinates
        coords[1] = coords[1].replace('.stream', '')  # Remove the ".stream" extension
        x, y = float(coords[0]), float(coords[1])
        
        chunks = parse_stream_file(file_path)

        for chunk in chunks:
            if chunk['serial'] == target_serial_number:  # Check if the serial number matches the target
                rmsd = find_nearest_neighbours(chunk['peaks'], chunk['reflections'], n)
                if rmsd is not None:
                    rmsd_values[(x, y)] = rmsd
                    x_coords.add(x)
                    y_coords.add(y)
                break  # Exit the loop once the target serial number is found in this stream

    # Check if any data was collected
    if not rmsd_values:
        print(f"No RMSD data found for image serial number {target_serial_number}.")
        return rmsd_values

    # Sort and convert coordinates to lists for consistent ordering
    x_coords = sorted(list(x_coords))
    y_coords = sorted(list(y_coords))

    # Create a 2D array for RMSD values
    heatmap_rmsd = np.full((len(y_coords), len(x_coords)), np.nan)  # Initialize with NaNs

    # Fill the heatmap array with RMSD values
    for (x, y), rmsd in rmsd_values.items():
        x_index = x_coords.index(x)
        y_index = y_coords.index(y)
        heatmap_rmsd[y_index, x_index] = rmsd  # Note: y_index first due to row-major order

    # Plot the RMSD heatmap for the specific frame
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_rmsd, cmap='viridis', origin='lower', extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)])
    plt.colorbar(label='RMSD')
    plt.title(f'RMSD Heatmap for Image Serial Number {target_serial_number}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    plt.show()

    return rmsd_values
