import os
import subprocess
from tqdm import tqdm

from list_h5_files import list_h5_files
from modify_geometry_file import modify_geometry_file
from run_indexamajig_sim import run_indexamajig
from generate_xy_pairs import generate_xy_pairs
 
# def gandalf_iterator(geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance, step=0.01, layers=1):
def gandalf_iterator(geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_iterations, tolerance, step=0.01, layers=1):
    
    list_h5_files(input_path)

    xdefault = -512
    ydefault = -512

    # Generate xy pairs including the default coordinates
    xy_pairs = [(xdefault, ydefault)] + list(generate_xy_pairs(xdefault, ydefault, step, layers))


    # Iterate over all xy pairs
    for x, y in tqdm(xy_pairs, desc="Processing XY pairs"):
        print(f"Running for x={x}, y={y}")
        try:
            temp_geomfile_path = modify_geometry_file(geomfile_path, x, y)  # Create temporary geom file for each iteration
            # run_indexamajig(x, y, temp_geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance)
            run_indexamajig(x, y, temp_geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_iterations, tolerance)

        except KeyboardInterrupt:
            print("Process interrupted by user.")
            break
        except subprocess.CalledProcessError as e:
            print(f"Error during indexamajig execution: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            if os.path.exists(temp_geomfile_path):
                os.remove(temp_geomfile_path)  # Ensure the file is deleted if not already done