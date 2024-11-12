import re
from calculate_weighted_rmsd import calculate_weighted_rmsd
from calculate_cell_deviation import calculate_cell_deviation
from match_peaks_to_reflections import match_peaks_to_reflections
def extract_chunk_data(chunk, original_cell_params, tolerance=1.0):
    # Extract event number
    event_match = re.search(r'Event: //(\d+)', chunk)
    event_number = int(event_match.group(1)) if event_match else None
    if event_number is None:
        print("No event number found in chunk.")

    # Extract peak list
    peak_list_match = re.search(r'Peaks from peak search(.*?)End of peak list', chunk, re.S)
    if peak_list_match:
        peaks = re.findall(r'\s+(\d+\.\d+)\s+(\d+\.\d+)\s+[\d.]+\s+[\d.]+\s+p0', peak_list_match.group(1))
        fs_ss = [(float(fs), float(ss)) for fs, ss in peaks]
        intensities = []  # Intensities are not used in matching, so we can omit them here
        if not peaks:
            print("No peaks found in chunk.")
    else:
        fs_ss = []
        print("No peak list found in chunk.")

    # Extract reflections
    reflections_match = re.search(r'Reflections measured after indexing(.*?)End of reflections', chunk, re.S)
    if reflections_match:
        reflections = re.findall(
            r'\s+-?\d+\s+-?\d+\s+-?\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+p0',
            reflections_match.group(1)
        )
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
        a, b, c = map(lambda x: float(x) * 10, cell_params_match.groups()[:3])  # Convert from nm to Å
        al, be, ga = map(float, cell_params_match.groups()[3:])
        cell_params = (a, b, c, al, be, ga)
    else:
        cell_params = None
        print("No cell parameters found in chunk.")

    # Calculate cell deviation if possible
    if cell_params is not None and original_cell_params is not None:
        length_deviation, angle_deviation = calculate_cell_deviation(cell_params, original_cell_params)
    else:
        length_deviation, angle_deviation = None, None
        print("Unable to calculate cell deviation for chunk.")

    # Extract number of peaks and reflections
    num_peaks_match = re.search(r'num_peaks = (\d+)', chunk)
    num_peaks = int(num_peaks_match.group(1)) if num_peaks_match else len(fs_ss)
    if num_peaks == 0:
        print("No peaks count found in chunk.")

    num_reflections_match = re.search(r'num_reflections = (\d+)', chunk)
    num_reflections = int(num_reflections_match.group(1)) if num_reflections_match else len(ref_fs_ss)
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

    # Calculate percentage of peaks indexed
    if fs_ss and ref_fs_ss:
        number_matched_peaks = match_peaks_to_reflections(fs_ss, ref_fs_ss, tolerance=tolerance)
        percentage_indexed = (number_matched_peaks / len(fs_ss)) * 100
        # print(percentage_indexed)
    else:
        number_matched_peaks = 0
        percentage_indexed = 0.0
        print("Unable to calculate percentage of peaks indexed for chunk.")

    return (
        event_number,
        weighted_rmsd,
        length_deviation,
        angle_deviation,
        num_peaks,
        num_reflections,
        peak_resolution,
        diffraction_resolution,
        profile_radius,
        percentage_indexed,
        chunk
    )
