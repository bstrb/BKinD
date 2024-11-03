import os
import re

def extract_min_resolutions(input_stream_file):
    min_peak_resolution = float('inf')
    min_diffraction_resolution = float('inf')

    try:
        with open(input_stream_file, 'r') as stream_file:
            for line in stream_file:
                if 'peak_resolution' in line:
                    match = re.search(r'or ([\d\.]+) A', line)
                    if match:
                        try:
                            value = float(match.group(1))
                            min_peak_resolution = min(min_peak_resolution, value)
                        except ValueError:
                            pass
                elif 'diffraction_resolution_limit' in line:
                    match = re.search(r'or ([\d\.]+) A', line)
                    if match:
                        try:
                            value = float(match.group(1))
                            min_diffraction_resolution = min(min_diffraction_resolution, value)
                        except ValueError:
                            pass

    except Exception as e:
        print(f"An exception occurred while extracting resolutions from {input_stream_file}: {e}")

    return min_peak_resolution, min_diffraction_resolution

# Function to extract resolutions from all stream files in a folder
def extract_resolutions_from_folder(input_folder):
    overall_min_peak_resolution = float('inf')
    overall_min_diffraction_resolution = float('inf')

    try:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.stream'):
                file_path = os.path.join(input_folder, file_name)
                min_peak_res, min_diff_res = extract_min_resolutions(file_path)
                overall_min_peak_resolution = min(overall_min_peak_resolution, min_peak_res)
                overall_min_diffraction_resolution = min(overall_min_diffraction_resolution, min_diff_res)

        print(f"Overall minimum peak resolution: {overall_min_peak_resolution}")
        print(f"Overall minimum diffraction resolution limit: {overall_min_diffraction_resolution}")
    except Exception as e:
        print(f"An exception occurred while processing the folder {input_folder}: {e}")

    return overall_min_peak_resolution, overall_min_diffraction_resolution

# Example usage
if __name__ == "__main__":
    input_folder = '/home/buster/R2aOx'   # Replace with your actual folder path
    overall_min_peak_res, overall_min_diff_res = extract_resolutions_from_folder(input_folder)
