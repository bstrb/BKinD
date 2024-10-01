import os
import subprocess

from list_h5_files import list_h5_files
from modify_geometry_file import modify_geometry_file
from run_indexamajig import run_indexamajig
from generate_xy_pairs import generate_xy_pairs
 
def gandalf_iterator(geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance, step=0.01, layers=1):
    
    list_h5_files(input_path)

    xdefault = -512
    ydefault = -512

    print(f"Running for initial x={str(xdefault)}, y={str(ydefault)}")
    
    # Start with the specified coordinates (-512, -512)
    try:
        temp_geomfile_path = modify_geometry_file(geomfile_path, xdefault, ydefault)  # Create temporary geom file
        run_indexamajig(xdefault, ydefault, temp_geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance)
        os.remove(temp_geomfile_path)  # Remove temporary file after use

    except KeyboardInterrupt:
        print("Process interrupted by user.")
        exit()
    except subprocess.CalledProcessError as e:
        print(f"Error during initial indexamajig execution: {e}")
        exit()
    except Exception as e:
        print(f"Unexpected error during initial execution: {e}")
        exit()

    # Continue with the rest of the xy pairs
    xy_pairs = generate_xy_pairs(xdefault, ydefault, step, layers)  # Adjust the number of layers as needed

    for x, y in xy_pairs:
        print(f"Running for x={x}, y={y}")
        try:
            temp_geomfile_path = modify_geometry_file(geomfile_path, x, y)  # Create temporary geom file for each iteration
            run_indexamajig(x, y, temp_geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance)
            os.remove(temp_geomfile_path)  # Remove the temporary file after use

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
