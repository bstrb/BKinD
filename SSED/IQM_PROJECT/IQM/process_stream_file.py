import os
import re
from tqdm import tqdm

from extract_chunk_data import extract_chunk_data
from calculate_combined_metric import calculate_combined_metric

def process_stream_file(stream_file_path):

    metric_weights = {
        'weighted_rmsd': 12,
        'length_deviation': 12,
        'angle_deviation': 10,
        'num_peaks': -12,
        'num_reflections': 12,
        'peak_resolution': -15,
        'diffraction_resolution': 10,
        'profile_radius': 13,
        'percentage_indexed': -13  # Negative weight to favor higher percentages
    }

    results = []
    none_results = []
    all_metrics = {
        'weighted_rmsd': [],
        'length_deviation': [],
        'angle_deviation': [],
        'num_peaks': [],
        'num_reflections': [],
        'peak_resolution': [],
        'diffraction_resolution': [],
        'profile_radius': [],
        'percentage_indexed': []  # New metric added
    }

    with open(stream_file_path, 'r') as file:
        content = file.read()
        header, *chunks = re.split(r'----- Begin chunk -----', content)  # Split by chunk delimiter

        # Extract original cell parameters from the header
        cell_params_match = re.search(r'a = ([\d.]+) A\nb = ([\d.]+) A\nc = ([\d.]+) A\nal = ([\d.]+) deg\nbe = ([\d.]+) deg\nga = ([\d.]+) deg', header)
        if cell_params_match:
            original_cell_params = tuple(map(float, cell_params_match.groups()))
        else:
            original_cell_params = None
            print("No original cell parameters found in header.")

        # Iterate over each chunk
        for chunk in tqdm(chunks, desc=f"Processing chunks in {os.path.basename(stream_file_path)}", unit="chunk"): #tqdm progress bar for all chunks simultatenously
        # for chunk in chunks:
            if "indexed_by = none" in chunk.lower():
                continue  # Skip unindexed chunks

            (event_number, weighted_rmsd, length_deviation, angle_deviation,
             num_peaks, num_reflections, peak_resolution, diffraction_resolution,
             profile_radius, percentage_indexed, chunk_content) = extract_chunk_data(chunk, original_cell_params)

            if event_number is not None:
                if None not in (weighted_rmsd, length_deviation, angle_deviation, peak_resolution, diffraction_resolution, percentage_indexed):
                    results.append((
                        os.path.basename(stream_file_path), event_number, weighted_rmsd, length_deviation, angle_deviation,
                        num_peaks, num_reflections, peak_resolution, diffraction_resolution, profile_radius, percentage_indexed, chunk_content
                    ))
                    all_metrics['weighted_rmsd'].append(weighted_rmsd)
                    all_metrics['length_deviation'].append(length_deviation)
                    all_metrics['angle_deviation'].append(angle_deviation)
                    all_metrics['num_peaks'].append(num_peaks)
                    all_metrics['num_reflections'].append(num_reflections)
                    all_metrics['peak_resolution'].append(peak_resolution)
                    all_metrics['diffraction_resolution'].append(diffraction_resolution)
                    all_metrics['profile_radius'].append(profile_radius)
                    all_metrics['percentage_indexed'].append(percentage_indexed)  # Collect the new metric
                else:
                    none_results.append((os.path.basename(stream_file_path), event_number, "None"))

    # Normalize each metric to a value between 0 and 1
    for key in all_metrics:
        if all_metrics[key]:
            min_value = min(all_metrics[key])
            max_value = max(all_metrics[key])
            if max_value != min_value:
                all_metrics[key] = [(value - min_value) / (max_value - min_value) for value in all_metrics[key]]
            else:
                all_metrics[key] = [1 for _ in all_metrics[key]]  # If all values are the same, assign 0.5

    # Update results with normalized metrics and compute combined metric
    for i, result in enumerate(results):
        filename, event_number, weighted_rmsd, length_deviation, angle_deviation, num_peaks, num_reflections, peak_resolution, diffraction_resolution, profile_radius, percentage_indexed, chunk_content = result

        # Compute combined metric using the product approach
        combined_metric = calculate_combined_metric(i, all_metrics, metric_weights)

        # Update result with computed metric
        results[i] = (os.path.basename(stream_file_path), event_number, combined_metric, chunk_content)

    # Sort results by combined metric value in ascending order
    results.sort(key=lambda x: x[2])

    return results, none_results, header
