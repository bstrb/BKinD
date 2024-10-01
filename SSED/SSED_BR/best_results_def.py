# best_results_definitions.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from find_nearest_neighbours import find_nearest_neighbours
from parse_stream_file import parse_stream_file
from find_stream_files import find_stream_files
from post_process_best_results import write_best_results_to_stream, print_output_file_statistics, find_and_copy_header

def rmsd_analysis(file_paths, n):

    best_results = {}
    stream_stats = {}

    for file_path in file_paths:
        chunks = parse_stream_file(file_path)
        sum_rmsd = 0
        rmsd_count = 0
        indexed_patterns_count = 0  # Initialize counter for indexed patterns

        for chunk in chunks:
            rmsd = find_nearest_neighbours(chunk['peaks'], chunk['reflections'], n)
            if rmsd:  # Ensure there's a valid rmsd value
                sum_rmsd += rmsd
                rmsd_count += 1
                if chunk['serial'] not in best_results or best_results[chunk['serial']]['rmsd'] > rmsd:
                    best_results[chunk['serial']] = {'file_path': file_path, 'chunk': chunk, 'rmsd': rmsd}
            if chunk['reflections']:  # Check if the chunk has reflections, indicating indexing
                indexed_patterns_count += 1

        # Calculate and store statistics for this stream file
        if rmsd_count > 0:
            avg_rmsd = sum_rmsd / rmsd_count
        else:
            avg_rmsd = None

        stream_stats[file_path] = {
        'avg_rmsd': avg_rmsd,
        'chunk_count': len(chunks),
        'indexed_patterns_count': indexed_patterns_count  # Store the count of indexed patterns
        }

    # Initialize dictionaries to hold RMSD values and coordinates
    rmsd_values = {}
    indexed_patterns_values = {}
    x_coords = set()
    y_coords = set()

    # Extract coordinates and RMSD values from file paths
    for file_path, stats in stream_stats.items():
        filename = file_path.split('/')[-1]  # Extract filename
        # coords = filename.split('_')[1:3]  # Extract coordinates, e.g., ['-512.0', '-512.02.stream']
        # coords[1] = coords[1].replace('.stream', '')  # Remove the ".stream" extension
        # x, y = float(coords[0]), float(coords[1])
    
        if filename.count('_') < 2 or not filename.endswith('.stream'):
            print(f"Skipping file {filename} as it doesn't match the expected naming pattern.")
            continue

        try:
            coords = filename.split('_')[1:3]  # Extract coordinates, e.g., ['-512.0', '-512.02.stream']
            coords[1] = coords[1].replace('.stream', '')  # Remove the ".stream" extension
            x, y = float(coords[0]), float(coords[1])
        except (IndexError, ValueError) as e:
            print(f"Skipping file {filename} due to error in extracting coordinates: {e}")
            continue


        # Add coordinates and RMSD/indexed patterns values
        rmsd_values[(x, y)] = stats['avg_rmsd']
        indexed_patterns_values[(x, y)] = stats['indexed_patterns_count']
        x_coords.add(x)
        y_coords.add(y)

    # Sort and convert coordinates to lists for consistent ordering
    x_coords = sorted(list(x_coords))
    y_coords = sorted(list(y_coords))

    # Create 2D arrays for RMSD and indexed patterns values
    heatmap_rmsd = np.full((len(y_coords), len(x_coords)), np.nan)  # Initialize with NaNs
    heatmap_indexed_patterns = np.full((len(y_coords), len(x_coords)), np.nan)  # Initialize with NaNs

    # Fill the heatmap arrays with RMSD and indexed patterns values
    for (x, y), rmsd in rmsd_values.items():
        x_index = x_coords.index(x)
        y_index = y_coords.index(y)
        heatmap_rmsd[y_index, x_index] = rmsd  # Note: y_index first due to row-major order
        heatmap_indexed_patterns[y_index, x_index] = indexed_patterns_values[(x, y)]

    # Plot the RMSD heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_rmsd, cmap='viridis', origin='lower', extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)])
    plt.colorbar(label='Average RMSD')
    plt.title('Average RMSD Heatmap')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    plt.show()

    # Plot the indexed patterns count heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_indexed_patterns, cmap='plasma', origin='lower', extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)])
    plt.colorbar(label='Indexed Patterns Count')
    plt.title('Indexed Patterns Count Heatmap')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    plt.show()

    # Print statistics for each stream file, including the number of indexed patterns
    for file_path, stats in stream_stats.items():
        print(f"File: {file_path}, Average RMSD: {stats['avg_rmsd']}, Chunk Count: {stats['chunk_count']}, Indexed Patterns: {stats['indexed_patterns_count']}")

    return best_results

def find_best_results(folder_path, output_path):

    file_paths = find_stream_files(folder_path)

    print(file_paths)

    n = 10
    
    best_results = rmsd_analysis(file_paths, n)

    write_best_results_to_stream(best_results, output_path)

    print_output_file_statistics(output_path, best_results)

    find_and_copy_header(folder_path, output_path)