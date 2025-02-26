import os
import re
import csv
from tqdm import tqdm
from extract_chunk_data import extract_chunk_data
from calculate_combined_metric import calculate_combined_metric
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock

def process_stream_file(stream_file_path, metric_weights=None):
    metric_names = [
        'weighted_rmsd', 'length_deviation', 'angle_deviation', 'num_peaks',
        'num_reflections', 'peak_resolution', 'diffraction_resolution', 'profile_radius',
        'percentage_indexed'  # New metric added
    ]

    # If metric_weights is an array, convert it to a dictionary
    if isinstance(metric_weights, (list, tuple)) and len(metric_weights) == len(metric_names):
        metric_weights = dict(zip(metric_names, metric_weights))
    elif metric_weights is None:
        metric_weights = {
            'weighted_rmsd': 1,
            'length_deviation': 2,
            'angle_deviation': 3,
            'num_peaks': -1,
            'num_reflections': 1,
            'peak_resolution': -1,
            'diffraction_resolution': 1,
            'profile_radius': 1,
            'percentage_indexed': -2  # Negative weight to favor higher percentages
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


# Helper function to process a file and store results
def process_and_store(stream_file_path, metric_weights, all_results, best_results, header, lock):
    results, none_results, file_header = process_stream_file(stream_file_path, metric_weights)

    if file_header and not header:
        with lock:
            if not header:
                header.append(file_header)

    with lock:
        all_results.extend(results)
        all_results.extend(none_results)

        # Update best_results to keep only the lowest combined metric for each event number
        best_results_dict = {result[1]: result for result in best_results}
        for result in results:
            event_number = result[1]
            if event_number not in best_results_dict or result[2] < best_results_dict[event_number][2]:
                best_results_dict[event_number] = result

        best_results[:] = list(best_results_dict.values())

# Function to process all stream files in a folder using multiprocessing
def process_all_stream_files(folder_path, metric_weights=None):
    manager = Manager()
    all_results = manager.list()
    best_results = manager.list()
    header = manager.list()
    lock = manager.Lock()

    # Remove existing best_results stream files in the folder
    for f in os.listdir(folder_path):
        if f.startswith('best_results') and f.endswith('.stream'):
            os.remove(os.path.join(folder_path, f))

    # Iterate over all stream files in the folder
    stream_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.stream')]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_and_store, stream_file, metric_weights, all_results, best_results, header, lock): stream_file for stream_file in stream_files}
        for future in as_completed(futures):
            futures.pop(future)

    # Sort best_results by combined metric value in ascending order
    best_results = list(best_results)
    best_results.sort(key=lambda x: x[2])

    weight_combination_str = '_'.join([f'{value}' for value in (metric_weights or [])])
    output_dir = os.path.join(folder_path, f'IQM_{weight_combination_str}')
    os.makedirs(output_dir, exist_ok=True)

    output_csv_path = os.path.join(output_dir, "IQM_values.csv")
    output_stream_path = os.path.join(output_dir, "best_IQM.stream")



    # Write all results to CSV
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['stream_file', 'event_number', 'combined_metric'])
        for result in all_results:
            csv_writer.writerow(result[:3])

    # Write best results to a stream file (keeping all unique event numbers with the lowest combined metric)
    if best_results and header:
        with open(output_stream_path, 'w') as stream_file:
            stream_file.write(header[0])  # Write the header to the output stream file
            for result in best_results:
                stream_file.write("----- Begin chunk -----\n")
                stream_file.write(result[3])
                stream_file.write("----- End chunk -----\n")

        print(f'Combined metrics CSV written to {output_csv_path}')
        print(f'Best results stream file written to {output_stream_path}')
    else:
        print("No valid chunks found in any stream file.")

# Example usage
if __name__ == "__main__":
    folder_path = "/home/buster/UOX1/different_index_params/3x3"
    metric_weights = (1, 2, 3, -1, 1, -1, 1, 1, -1)
    process_all_stream_files(folder_path, metric_weights)
