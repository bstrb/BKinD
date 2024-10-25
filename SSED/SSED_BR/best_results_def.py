# best_results_definitions.py

import numpy as np
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
