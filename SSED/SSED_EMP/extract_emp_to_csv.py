import os
import csv
import re

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from find_stream_files import find_stream_files
from parse_stream_file import parse_stream_file
from extract_target_cell_params import extract_target_cell_params
from calculate_cell_deviation import calculate_cell_deviation
from calculate_weighted_rmsd import calculate_weighted_rmsd

# Function to extract relevant data from each chunk
def extract_chunk_data(chunk):
    # Check if the chunk is indexed
    if "indexed_by = none" in chunk.lower():
        return None, None, None, None, None, None, None, None

    else:
        # Extract event number
        event_match = re.search(r'Event: //(\d+)', chunk)
        event_number = int(event_match.group(1)) if event_match else None

        # Extract number of reflections and peaks
        num_reflections_match = re.search(r'num_reflections = (\d+)', chunk)
        num_reflections = int(num_reflections_match.group(1)) if num_reflections_match else 0

        num_peaks_match = re.search(r'num_peaks = (\d+)', chunk)
        num_peaks = int(num_peaks_match.group(1)) if num_peaks_match else 0

        # Extract cell parameters
        cell_params_match = re.search(r'Cell parameters ([\d.]+) ([\d.]+) ([\d.]+) nm, ([\d.]+) ([\d.]+) ([\d.]+) deg', chunk)
        if cell_params_match:
            a, b, c = map(lambda x: float(x) * 10, cell_params_match.groups()[:3])  # Convert from nm to A
            al, be, ga = map(float, cell_params_match.groups()[3:])
            cell_params = (a, b, c, al, be, ga)
        else:
            cell_params = None

        # Extract peak list
        peak_list_match = re.search(r'Peaks from peak search\n(.*?)End of peak list', chunk, re.S)
        if peak_list_match:
            peaks = re.findall(r'\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0', peak_list_match.group(1))
            fs_ss, intensities = [], []
            for peak in peaks:
                fs, ss, _, intensity = map(float, peak)
                fs_ss.append((fs, ss))
                intensities.append(intensity)
        else:
            fs_ss, intensities = [], []

        # Extract reflections
        reflections_match = re.search(r'Reflections measured after indexing\n(.*?)End of reflections', chunk, re.S)
        if reflections_match:
            reflections = re.findall(r'\s+-?\d+\s+-?\d+\s+-?\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0', reflections_match.group(1))
            ref_fs_ss = [(float(fs), float(ss)) for fs, ss in reflections]
        else:
            ref_fs_ss = []

        # Extract profile_radius
        profile_radius_match = re.search(r'profile_radius = ([\d.]+) nm\^-1', chunk)
        profile_radius = float(profile_radius_match.group(1)) if profile_radius_match else None

        return event_number, num_reflections, num_peaks, cell_params, fs_ss, intensities, ref_fs_ss, profile_radius
    
# Function to parse a single stream file and evaluate its indexing
def process_stream_file(stream_file_path, wrmsd_exp, cld_exp, cad_exp, np_exp, nr_exp, pr_exp, progress_queue):
    try:
        current_header, chunks = parse_stream_file(stream_file_path)
        target_params = extract_target_cell_params(current_header)

        metrics = []
        # Loop through each chunk (ignoring the first, which is the header)
        for i, chunk in enumerate(chunks[1:], start=1):
            event_number, num_reflections, num_peaks, cell_params, fs_ss, intensities, ref_fs_ss, profile_radius = extract_chunk_data(chunk)

            if event_number is None or cell_params is None or not fs_ss or not ref_fs_ss or num_peaks == 0:
                progress_queue.put(1)
                continue

            cell_length_deviation, cell_angle_deviation = calculate_cell_deviation(cell_params, target_params)
            weighted_rmsd = calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss)

            # Append raw metrics for normalization
            metrics.append((event_number, weighted_rmsd, cell_length_deviation, cell_angle_deviation, num_reflections, num_peaks, profile_radius, chunk))

            # Update progress every chunk
            progress_queue.put(1)
            
        # Normalize metrics
        if metrics:
            weighted_rmsds = [metric[1] for metric in metrics]
            cell_length_deviations = [metric[2] for metric in metrics]
            cell_angle_deviations = [metric[3] for metric in metrics]
            num_peaks_list = [metric[5] for metric in metrics]
            num_reflections_list = [metric[4] for metric in metrics]
            profile_radii = [metric[6] for metric in metrics]

            min_rmsd, max_rmsd = min(weighted_rmsds), max(weighted_rmsds)
            min_cld, max_cld = min(cell_length_deviations), max(cell_length_deviations)
            min_cad, max_cad = min(cell_angle_deviations), max(cell_angle_deviations)
            min_np, max_np = min(num_peaks_list), max(num_peaks_list)
            min_nr, max_nr = min(num_reflections_list), max(num_reflections_list)
            min_pr, max_pr = min(profile_radii), max(profile_radii)

            normalized_metrics = []
            for event_number, weighted_rmsd, cell_length_deviation, cell_angle_deviation, num_reflections, num_peaks, profile_radius, chunk in metrics:
                normalized_rmsd = (weighted_rmsd - min_rmsd) / (max_rmsd - min_rmsd) if max_rmsd > min_rmsd else 0
                normalized_cld = (cell_length_deviation - min_cld) / (max_cld - min_cld) if max_cld > min_cld else 0
                normalized_cad = (cell_angle_deviation - min_cad) / (max_cad - min_cad) if max_cad > min_cad else 0
                normalized_np = (num_peaks - min_np) / (max_np - min_np) if max_np > min_np else 0
                normalized_nr = (num_reflections - min_nr) / (max_nr - min_nr) if max_nr > min_nr else 0
                normalized_pr = (profile_radius - min_pr) / (max_pr - min_pr) if max_pr > min_pr else 0

                # Combine normalized metrics (weights can be adjusted as needed)
                combined_metric = ((1 + normalized_rmsd) ** wrmsd_exp) * ((1 + normalized_cld) ** cld_exp) * ((1 + normalized_cad) ** cad_exp) * ((1 + normalized_np) ** np_exp) * ((1 + normalized_nr) ** nr_exp) * ((1 + normalized_pr) ** pr_exp)
                normalized_metrics.append((event_number, combined_metric, chunk))

            return current_header, normalized_metrics
        else:
            return current_header, []
    except Exception as e:
        progress_queue.put(1)  # Update progress in case of error
        return None, []


# Function to parse multiple stream files, evaluate indexing, and create a combined output file and CSV
def evaluate_multiple_streams(stream_file_folder, exp, EMP):
    try:
        wrmsd_exp, cld_exp, cad_exp, np_exp, nr_exp, pr_exp = exp
        stream_file_paths = [path for path in find_stream_files(stream_file_folder) if not os.path.basename(path).startswith("best_results")]
        all_metrics = []
        header = None
        output_file_path = os.path.join(stream_file_folder, f'best_results_{EMP}.stream')
        output_csv_path = os.path.join(stream_file_folder, f'combined_metrics_{EMP}.csv')

        # Remove existing best_results.stream and combined_metrics.csv if they exist
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)

        # Initialize progress tracking
        manager = Manager()
        progress_queue = manager.Queue()
        total_chunks = sum(len(parse_stream_file(path)[1]) - 1 for path in stream_file_paths)

        # Launch progress bar updater
        with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as progress_bar:
            def progress_updater():
                while True:
                    item = progress_queue.get()
                    if item == "STOP":
                        break
                    progress_bar.update(item)

            from threading import Thread
            progress_thread = Thread(target=progress_updater, daemon=True)
            progress_thread.start()

            # Process stream files in parallel
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(process_stream_file, path, wrmsd_exp, cld_exp, cad_exp, np_exp, nr_exp, pr_exp, progress_queue): path for path in stream_file_paths}
                for future in futures:
                    try:
                        current_header, metrics = future.result()
                        if header is None and current_header is not None:
                            header = current_header
                        all_metrics.extend(metrics)
                    except Exception as e:
                        tqdm.write(f"Error processing stream file {futures[future]}: {e}")

            # Stop the progress updater thread
            progress_queue.put("STOP")
            progress_thread.join()

        # Sort all metrics by combined metric value in ascending order
        all_metrics.sort(key=lambda x: x[1])

        # Write the combined metrics and corresponding chunks with the lowest scores to the output file
        with open(output_file_path, 'w') as output_file:
            if header is not None:
                output_file.write(header + '\n')  # Write the header from any of the stream files
            written_events = set()

            for event_number, _, chunk in all_metrics:
                if event_number not in written_events:
                    output_file.write("----- Begin chunk -----\n")
                    output_file.write(chunk)
                    written_events.add(event_number)

        # Write the combined metrics to a CSV file
        with open(output_csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row
            csv_writer.writerow(['stream_file', 'event_number', 'combined_metric'])

            for stream_file_path in stream_file_paths:
                filename = os.path.basename(stream_file_path)
                for event_number, combined_metric, _ in all_metrics:
                    csv_writer.writerow([filename, event_number, combined_metric])

        print(f'Combined metrics and selected chunks written to {output_file_path}')
        print(f'Combined metrics CSV written to {output_csv_path}')
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

        # Example Usage
if __name__ == "__main__":
    # Folder containing the stream files
    stream_file_folder = "/home/buster/UOX1/3x3"

    # Exponential weights for the metrics (e.g., 0.5, 1, -1, 1, 0, 1)
    # These values correspond to wrmsd_exp, cld_exp, cad_exp, np_exp, nr_exp, pr_exp
    exp_weights = [1, 2, 3, -1, 1, 1]

    # A unique identifier for the evaluation run, such as "EMP_001"
    EMP_identifier = "EMP_test"

    # Evaluate multiple stream files in the specified folder
    evaluate_multiple_streams(stream_file_folder, exp_weights, EMP_identifier)
