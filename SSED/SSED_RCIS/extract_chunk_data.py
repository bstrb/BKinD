import re

# Function to extract relevant data from each chunk
def extract_chunk_data(chunk):
    event_match = re.search(r'Event: //(\d+)', chunk)
    event_number = int(event_match.group(1)) if event_match else None

    num_reflections_match = re.search(r'num_reflections = (\d+)', chunk)
    num_reflections = int(num_reflections_match.group(1)) if num_reflections_match else 0

    cell_params_match = re.search(r'Cell parameters ([\d.]+) ([\d.]+) ([\d.]+) nm, ([\d.]+) ([\d.]+) ([\d.]+) deg', chunk)
    if cell_params_match:
        a, b, c = map(lambda x: float(x) * 10, cell_params_match.groups()[:3])  # Convert from nm to A
        al, be, ga = map(float, cell_params_match.groups()[3:])
        cell_params = (a, b, c, al, be, ga)
    else:
        cell_params = None

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

    reflections_match = re.search(r'Reflections measured after indexing\n(.*?)End of reflections', chunk, re.S)
    if reflections_match:
        reflections = re.findall(r'\s+-?\d+\s+-?\d+\s+-?\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0', reflections_match.group(1))
        ref_fs_ss = [(float(fs), float(ss)) for fs, ss in reflections]
    else:
        ref_fs_ss = []

    return event_number, num_reflections, cell_params, fs_ss, intensities, ref_fs_ss
