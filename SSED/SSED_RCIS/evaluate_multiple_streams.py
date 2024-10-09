# -------------------------
# Part 1: Evaluate Multiple Streams
# -------------------------
import os
from tqdm import tqdm
from find_stream_files import find_stream_files
from parse_stream_file import parse_stream_file
from extract_target_cell_params import extract_target_cell_params
from extract_chunk_data import extract_chunk_data
from calculate_cell_deviation import calculate_cell_deviation
from calculate_weighted_rmsd import calculate_weighted_rmsd

# Function to parse multiple stream files, evaluate indexing, and create a combined output file
def evaluate_multiple_streams(stream_file_folder, wrmsd_weight, cd_weight, rpr_weight):
    try:
        stream_file_paths = [path for path in find_stream_files(stream_file_folder) if not os.path.basename(path).startswith("best_results")]
        all_metrics = []
        header = None
        output_file_path = os.path.join(stream_file_folder, f'best_results_RCIS_{wrmsd_weight}_{cd_weight}_{rpr_weight}.stream')

        # Remove existing best_results.stream if it exists
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        with tqdm(total=len(stream_file_paths), desc="Processing stream files", unit="file", leave=True) as stream_pbar:
            for stream_file_path in stream_file_paths:
                try:
                    current_header, chunks = parse_stream_file(stream_file_path)
                    if header is None:
                        header = current_header  # Use the header from the first stream file
                    target_params = extract_target_cell_params(current_header)

                    # Loop through each chunk (ignoring the first, which is the header)
                    for i, chunk in enumerate(chunks[1:], start=1):
                            event_number, num_reflections, num_peaks, cell_params, fs_ss, intensities, ref_fs_ss = extract_chunk_data(chunk)

                            if event_number is None or cell_params is None or not fs_ss or not ref_fs_ss or num_peaks == 0:
                                continue

                            cell_deviation = calculate_cell_deviation(cell_params, target_params)
                            weighted_rmsd = calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss)

                            # Use num_reflections/num_peaks instead of just num_reflections
                            reflection_to_peak_ratio = num_reflections / num_peaks if num_peaks > 0 else 0

                            # Combine metrics (weights can be adjusted as needed)
                            combined_metric = (wrmsd_weight * weighted_rmsd) + (cd_weight * cell_deviation) - (rpr_weight * reflection_to_peak_ratio)
                            all_metrics.append((event_number, combined_metric, chunk))
                except Exception as e:
                    tqdm.write(f"Error processing stream file {stream_file_path}: {e}")
                finally:
                    stream_pbar.update(1)

        # Sort all metrics by combined metric value in ascending order
        all_metrics.sort(key=lambda x: x[1])

        # Write the combined metrics and corresponding chunks with the lowest scores to the output file
        with open(output_file_path, 'w') as output_file:
            output_file.write(header + '\n')  # Write the header from any of the stream files
            written_events = set()

            for event_number, _, chunk in all_metrics:
                if event_number not in written_events:
                    output_file.write("----- Begin chunk -----\n")
                    output_file.write(chunk)
                    written_events.add(event_number)

        print(f'Combined metrics and selected chunks written to {output_file_path}')
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
