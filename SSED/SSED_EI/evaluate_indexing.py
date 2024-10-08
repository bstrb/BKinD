import re
import numpy as np
import os

# Function to parse the stream file and evaluate each indexed frame based on combined metrics
def evaluate_indexing(stream_file_path):
    with open(stream_file_path, 'r') as f:
        content = f.read()

    # Split content into individual chunks
    chunks = content.split("----- Begin chunk -----")
    metrics = []

    # Extract target unit cell parameters from the header
    header = chunks[0]
    target_cell_params_match = re.search(r'a = ([\d.]+) A\nb = ([\d.]+) A\nc = ([\d.]+) A\nal = ([\d.]+) deg\nbe = ([\d.]+) deg\nga = ([\d.]+) deg', header)
    if target_cell_params_match:
        target_a, target_b, target_c = map(float, target_cell_params_match.groups()[:3])
        target_al, target_be, target_ga = map(float, target_cell_params_match.groups()[3:])
    else:
        raise ValueError("Target unit cell parameters not found in the header.")

    # Loop through each chunk (ignoring the first, which is the header)
    for chunk in chunks[1:]:
        # Extract the event number
        event_match = re.search(r'Event: //(\d+)', chunk)
        event_number = int(event_match.group(1)) if event_match else None

        # Extract the number of indexed reflections
        num_reflections_match = re.search(r'num_reflections = (\d+)', chunk)
        num_reflections = int(num_reflections_match.group(1)) if num_reflections_match else 0

        # Extract cell parameters from the indexed chunk
        cell_params_match = re.search(r'Cell parameters ([\d.]+) ([\d.]+) ([\d.]+) nm, ([\d.]+) ([\d.]+) ([\d.]+) deg', chunk)
        if cell_params_match:
            a, b, c = map(lambda x: float(x) * 10, cell_params_match.groups()[:3])  # Convert from nm to A
            al, be, ga = map(float, cell_params_match.groups()[3:])
        else:
            continue

        # Calculate deviation from target cell parameters
        cell_deviation = np.sqrt((a - target_a) ** 2 + (b - target_b) ** 2 + (c - target_c) ** 2 +
                                 (al - target_al) ** 2 + (be - target_be) ** 2 + (ga - target_ga) ** 2)

        # Extract peaks from the peak list
        peak_list_match = re.search(r'Peaks from peak search\n(.*?)End of peak list', chunk, re.S)
        if peak_list_match:
            peaks = re.findall(r'\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0', peak_list_match.group(1))
            fs_ss, intensities = [], []
            for peak in peaks:
                fs, ss, _, intensity = map(float, peak)
                fs_ss.append((fs, ss))
                intensities.append(intensity)
        else:
            continue

        # Extract reflections from the indexing list
        reflections_match = re.search(r'Reflections measured after indexing\n(.*?)End of reflections', chunk, re.S)
        if reflections_match:
            reflections = re.findall(r'\s+-?\d+\s+-?\d+\s+-?\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0', reflections_match.group(1))
            ref_fs_ss = [(float(fs), float(ss)) for fs, ss in reflections]
        else:
            continue

        # Calculate weighted RMSD between peaks and reflections
        total_rmsd = 0
        total_weight = 0
        for (fs, ss), intensity in zip(fs_ss, intensities):
            # Find the closest reflection
            min_distance = float('inf')
            for ref_fs, ref_ss in ref_fs_ss:
                distance = np.sqrt((fs - ref_fs) ** 2 + (ss - ref_ss) ** 2)
                if distance < min_distance:
                    min_distance = distance
            # Calculate RMSD contribution weighted by intensity
            total_rmsd += (min_distance ** 2) * intensity
            total_weight += intensity

        if total_weight > 0:
            weighted_rmsd = np.sqrt(total_rmsd / total_weight)
        else:
            weighted_rmsd = float('inf')

        # Combine metrics (weights can be adjusted as needed)
        combined_metric = (0.5 * weighted_rmsd) + (0.3 * cell_deviation) - (0.2 * num_reflections)
        metrics.append((event_number, combined_metric))

    # Write the combined metrics to an output file in the same folder as the input stream file
    output_file_path = os.path.join(os.path.dirname(stream_file_path), 'combined_metrics.txt')
    with open(output_file_path, 'w') as output_file:
        for event_number, combined_metric in metrics:
            output_file.write(f'Event #{event_number}: Combined metric value = {combined_metric}\n')

    print(f'Combined metrics written to {output_file_path}')

# Example usage
# evaluate_indexing('example.stream')