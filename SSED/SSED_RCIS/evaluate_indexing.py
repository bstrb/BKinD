from parse_stream_file import parse_stream_file
from extract_target_cell_params import extract_target_cell_params
from extract_chunk_data import extract_chunk_data
from calculate_cell_deviation import calculate_cell_deviation
from calculate_weighted_rmsd import calculate_weighted_rmsd
from write_combined_metrics import write_combined_metrics

# Function to parse the stream file and evaluate each indexed frame based on combined metrics
def evaluate_indexing(stream_file_path):
    header, chunks = parse_stream_file(stream_file_path)
    target_params = extract_target_cell_params(header)
    metrics = []

    # Loop through each chunk (ignoring the first, which is the header)
    for chunk in chunks[1:]:
        event_number, num_reflections, cell_params, fs_ss, intensities, ref_fs_ss = extract_chunk_data(chunk)
        
        if event_number is None or cell_params is None or not fs_ss or not ref_fs_ss:
            continue

        cell_deviation = calculate_cell_deviation(cell_params, target_params)
        weighted_rmsd = calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss)

        # Combine metrics (weights can be adjusted as needed)
        combined_metric = (0.5 * weighted_rmsd) + (0.3 * cell_deviation) - (0.2 * num_reflections)
        metrics.append((event_number, combined_metric))

    write_combined_metrics(metrics, stream_file_path)

# Example usage
# evaluate_indexing('example.stream')