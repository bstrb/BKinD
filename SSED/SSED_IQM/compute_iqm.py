import os
import re
import csv
from tqdm import tqdm

# Function to extract relevant data from each chunk and compute the combined metric
def extract_chunk_data(chunk):
    # Extract event number
    event_match = re.search(r'Event: //(\d+)', chunk)
    event_number = int(event_match.group(1)) if event_match else None
    if event_number is None:
        print("No event number found in chunk.")

    # Extract peak list
    peak_list_match = re.search(r'Peaks from peak search(.*?)End of peak list', chunk, re.S)
    if peak_list_match:
        peaks = re.findall(r'\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0', peak_list_match.group(1))
        fs_ss, intensities = [], []
        for peak in peaks:
            fs, ss, _, intensity = map(float, peak)
            fs_ss.append((fs, ss))
            intensities.append(intensity)
        if not peaks:
            print("No peaks found in chunk.")
    else:
        fs_ss, intensities = [], []
        print("No peak list found in chunk.")

    # Extract reflections
    reflections_match = re.search(r'Reflections measured after indexing(.*?)End of reflections', chunk, re.S)
    if reflections_match:
        reflections = re.findall(r'\s+-?\d+\s+-?\d+\s+-?\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0', reflections_match.group(1))
        ref_fs_ss = [(float(fs), float(ss)) for fs, ss in reflections]
        if not reflections:
            print("No reflections found in chunk.")
    else:
        ref_fs_ss = []
        print("No reflections section found in chunk.")

    # Calculate weighted RMSD if possible
    if fs_ss and ref_fs_ss:
        weighted_rmsd = calculate_weighted_rmsd(fs_ss, intensities, ref_fs_ss)
    else:
        weighted_rmsd = None
        print("Unable to calculate weighted RMSD for chunk.")

    # Extract cell parameters (lengths and angles)
    cell_params_match = re.search(r'Cell parameters ([\d.]+) ([\d.]+) ([\d.]+) nm, ([\d.]+) ([\d.]+) ([\d.]+) deg', chunk)
    if cell_params_match:
        a, b, c = map(lambda x: float(x) * 10, cell_params_match.groups()[:3])  # Convert from nm to A
        al, be, ga = map(float, cell_params_match.groups()[3:])
        cell_params = (a, b, c, al, be, ga)
    else:
        cell_params = None
        print("No cell parameters found in chunk.")

    # Extract number of peaks and reflections
    num_peaks_match = re.search(r'num_peaks = (\d+)', chunk)
    num_peaks = int(num_peaks_match.group(1)) if num_peaks_match else 0
    if num_peaks == 0:
        print("No peaks count found in chunk.")

    num_reflections_match = re.search(r'num_reflections = (\d+)', chunk)
    num_reflections = int(num_reflections_match.group(1)) if num_reflections_match else 0
    if num_reflections == 0:
        print("No reflections count found in chunk.")

    # Extract peak resolution limit and diffraction resolution limit
    peak_resolution_match = re.search(r'peak_resolution = [\d.]+ nm\^-1 or ([\d.]+) A', chunk)
    peak_resolution = float(peak_resolution_match.group(1)) if peak_resolution_match else None
    if peak_resolution is None:
        print("No peak resolution found in chunk.")

    diffraction_resolution_match = re.search(r'diffraction_resolution_limit = [\d.]+ nm\^-1 or ([\d.]+) A', chunk)
    diffraction_resolution = float(diffraction_resolution_match.group(1)) if diffraction_resolution_match else None
    if diffraction_resolution is None:
        print("No diffraction resolution found in chunk.")

    # Extract profile radius
    profile_radius_match = re.search(r'profile_radius = ([\d.]+) nm\^-1', chunk)
    profile_radius = float(profile_radius_match.group(1)) if profile_radius_match else None
    if profile_radius is None:
        print("No profile radius found in chunk.")

    # Compute the combined metric (example formula, weights can be adjusted as needed)
    if weighted_rmsd is not None and cell_params is not None and peak_resolution is not None and diffraction_resolution is not None:
        a, b, c, al, be, ga = cell_params
        combined_metric = (weighted_rmsd * 0.2) + ((a + b + c) / 3 * 0.1) + ((al + be + ga) / 3 * 0.1) + (num_peaks * 0.1) + (num_reflections * 0.1) + (peak_resolution * 0.2) + (diffraction_resolution * 0.2) + (profile_radius * 0.1 if profile_radius is not None else 0)
    else:
        combined_metric = None
        print("Unable to compute combined metric for chunk.")

    return event_number, combined_metric

# Function to process a single stream file
def process_stream_file(stream_file_path):
    results = []

    with open(stream_file_path, 'r') as file:
        content = file.read()
        chunks = re.split(r'----- Begin chunk -----', content)[1:]  # Split by chunk delimiter and ignore the header

        # Iterate over each chunk with progress tracking
        for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
            if "indexed_by = none" in chunk.lower():
                continue  # Skip unindexed chunks

            event_number, combined_metric = extract_chunk_data(chunk)
            if event_number is not None and combined_metric is not None:
                results.append((os.path.basename(stream_file_path), event_number, combined_metric, chunk))

    # Sort results by combined metric value in ascending order
    results.sort(key=lambda x: x[2])

    # Write results to CSV and stream file
    output_csv_path = os.path.join(os.path.dirname(stream_file_path), 'combined_metrics.csv')
    output_stream_path = os.path.join(os.path.dirname(stream_file_path), 'best_results.stream')

    if results:
        with open(output_csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['stream_file', 'event_number', 'combined_metric'])
            for result in results:
                csv_writer.writerow(result[:3])

        with open(output_stream_path, 'w') as stream_file:
            for result in results:
                stream_file.write("----- Begin chunk -----\n")
                stream_file.write(result[3])
                stream_file.write("----- End chunk -----\n")

        print(f'Combined metrics CSV written to {output_csv_path}')
        print(f'Best results stream file written to {output_stream_path}')
    else:
        print("No valid chunks found in the stream file.")

# Example usage
if __name__ == "__main__":
    stream_file_path = "/home/buster/UOX1/3x3/UOX1_-511.99_-511.99.stream"
    process_stream_file(stream_file_path)
