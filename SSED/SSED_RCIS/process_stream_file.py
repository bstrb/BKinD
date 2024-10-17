from parse_stream_file import parse_stream_file
from extract_target_cell_params import extract_target_cell_params
from extract_chunk_data import extract_chunk_data
from calculate_cell_deviation import calculate_cell_deviation
from calculate_weighted_rmsd import calculate_weighted_rmsd

# Function to parse a single stream file and evaluate its indexing
def process_stream_file(stream_file_path, wrmsd_exp, cd_exp, rpr_exp, progress_queue):
    try:
        current_header, chunks = parse_stream_file(stream_file_path)
        target_params = extract_target_cell_params(current_header)

        metrics = []
        # Loop through each chunk (ignoring the first, which is the header)
        for i, chunk in enumerate(chunks[1:], start=1):
            event_number, num_reflections, num_peaks, cell_params, fs_ss, intensities, ref_fs_ss = extract_chunk_data(chunk)

            if event_number is None or cell_params is None or not fs_ss or not ref_fs_ss or num_peaks == 0:
                progress_queue.put(1)  # Update progress
                continue

            cell_deviation = calculate_cell_deviation(cell_params, target_params)
            weighted_rmsd = calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss)

            # Use num_reflections/num_peaks instead of just num_reflections
            reflection_to_peak_ratio = num_reflections / num_peaks if num_peaks > 0 else 0

            # Combine metrics (weights can be adjusted as needed)
            combined_metric = (weighted_rmsd ** wrmsd_exp) * (cell_deviation ** cd_exp) * (reflection_to_peak_ratio ** rpr_exp)
            metrics.append((event_number, combined_metric, chunk))

            # Update progress for each processed chunk
            progress_queue.put(1)

        return current_header, metrics
    except Exception as e:
        progress_queue.put(1)  # Update progress in case of error
        return None, []
