from parse_stream_file import parse_stream_file
from extract_target_cell_params import extract_target_cell_params
from extract_chunk_data import extract_chunk_data
from calculate_cell_deviation import calculate_cell_deviation
from calculate_weighted_rmsd import calculate_weighted_rmsd
import numpy as np

# Function to parse a single stream file and evaluate its indexing
def process_stream_file(stream_file_path, wrmsd_exp, cd_exp, np_exp, nr_exp, pr_exp, nit_exp, progress_queue):
    try:
        current_header, chunks = parse_stream_file(stream_file_path)
        target_params = extract_target_cell_params(current_header)

        metrics = []
        # Loop through each chunk (ignoring the first, which is the header)
        for i, chunk in enumerate(chunks[1:], start=1):
            event_number, num_reflections, num_peaks, cell_params, fs_ss, intensities, ref_fs_ss, profile_radius, n_indexing_tries = extract_chunk_data(chunk)

            if event_number is None or cell_params is None or not fs_ss or not ref_fs_ss or num_peaks == 0:
                if i % 100 == 0:
                    progress_queue.put(100)  # Update progress every 100 chunks
                continue

            cell_deviation = calculate_cell_deviation(cell_params, target_params)
            weighted_rmsd = calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss)

            # Append raw metrics for normalization
            metrics.append((event_number, weighted_rmsd, cell_deviation, num_reflections, num_peaks, profile_radius, n_indexing_tries, chunk))

            # Update progress every 100 chunks
            if i % 100 == 0:
                progress_queue.put(100)

        # Normalize metrics
        if metrics:
            weighted_rmsds = [metric[1] for metric in metrics]
            cell_deviations = [metric[2] for metric in metrics]
            num_peaks_list = [metric[4] for metric in metrics]
            num_reflections_list = [metric[3] for metric in metrics]
            profile_radii = [metric[5] for metric in metrics]
            n_indexing_tries_list = [metric[6] for metric in metrics]

            min_rmsd, max_rmsd = min(weighted_rmsds), max(weighted_rmsds)
            min_cd, max_cd = min(cell_deviations), max(cell_deviations)
            min_np, max_np = min(num_peaks_list), max(num_peaks_list)
            min_nr, max_nr = min(num_reflections_list), max(num_reflections_list)
            min_pr, max_pr = min(profile_radii), max(profile_radii)
            min_nit, max_nit = min(n_indexing_tries_list), max(n_indexing_tries_list)

            normalized_metrics = []
            for event_number, weighted_rmsd, cell_deviation, num_reflections, num_peaks, profile_radius, n_indexing_tries, chunk in metrics:
                normalized_rmsd = (weighted_rmsd - min_rmsd) / (max_rmsd - min_rmsd) if max_rmsd > min_rmsd else 0
                normalized_cd = (cell_deviation - min_cd) / (max_cd - min_cd) if max_cd > min_cd else 0
                normalized_np = (num_peaks - min_np) / (max_np - min_np) if max_np > min_np else 0
                normalized_nr = (num_reflections - min_nr) / (max_nr - min_nr) if max_nr > min_nr else 0
                normalized_pr = (profile_radius - min_pr) / (max_pr - min_pr) if max_pr > min_pr else 0
                normalized_nit = (n_indexing_tries - min_nit) / (max_nit - min_nit) if max_nit > min_nit else 0

                # Combine normalized metrics (weights can be adjusted as needed)
                combined_metric = (normalized_rmsd ** wrmsd_exp) * (normalized_cd ** cd_exp) * (normalized_np ** np_exp) * (normalized_nr ** nr_exp) * (normalized_pr ** pr_exp) * (normalized_nit ** nit_exp)
                normalized_metrics.append((event_number, combined_metric, chunk))

            return current_header, normalized_metrics
        else:
            return current_header, []
    except Exception as e:
        progress_queue.put(1)  # Update progress in case of error
        return None, []
