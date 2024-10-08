import os
from tqdm import tqdm
from find_stream_files import find_stream_files
from parse_stream_file import parse_stream_file
from extract_target_cell_params import extract_target_cell_params
from extract_chunk_data import extract_chunk_data
from calculate_cell_deviation import calculate_cell_deviation
from calculate_weighted_rmsd import calculate_weighted_rmsd

# Function to parse multiple stream files, evaluate indexing, and create a combined output file
def evaluate_multiple_streams(stream_file_folder):
    stream_file_paths = find_stream_files(stream_file_folder)
    all_metrics = []
    header = None
    output_file_path = os.path.join(stream_file_folder, 'best_results.stream')

    # Remove existing best_results.stream if it exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    for stream_file_path in tqdm(stream_file_paths, desc="Processing stream files"):
        current_header, chunks = parse_stream_file(stream_file_path)
        if header is None:
            header = current_header  # Use the header from the first stream file
        target_params = extract_target_cell_params(current_header)

        # Loop through each chunk (ignoring the first, which is the header)
        for chunk in chunks[1:]:
            event_number, num_reflections, cell_params, fs_ss, intensities, ref_fs_ss = extract_chunk_data(chunk)
            
            if event_number is None or cell_params is None or not fs_ss or not ref_fs_ss:
                continue

            cell_deviation = calculate_cell_deviation(cell_params, target_params)
            weighted_rmsd = calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss)

            # Combine metrics (weights can be adjusted as needed)
            combined_metric = (0.5 * weighted_rmsd) + (0.3 * cell_deviation) - (0.2 * num_reflections)
            all_metrics.append((event_number, combined_metric, chunk))

    # Sort all metrics by combined metric value in ascending order
    all_metrics.sort(key=lambda x: x[1])

    # Write the combined metrics and corresponding chunks with the lowest scores to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.write(header + '\n')  # Write the header from any of the stream files
        written_events = set()

        for event_number, _, chunk in tqdm(all_metrics, desc="Writing best results"):
            if event_number not in written_events:
                output_file.write("----- Begin chunk -----\n")
                output_file.write(chunk)
                written_events.add(event_number)

    print(f'Combined metrics and selected chunks written to {output_file_path}')

# Example usage
# evaluate_multiple_streams('stream_files_folder')