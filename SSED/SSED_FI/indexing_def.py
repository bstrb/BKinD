import os
import itertools
import subprocess

def generate_xy_pairs(x, y, step=0.01, layers=1): #1 layer means 2 layers i.e 5x5 pixels
    """Generate (x, y) pairs in a square pattern around a center."""
    center = (x,y)
    for layer in range(1, layers + 1):
        for dx, dy in itertools.product(range(-layer, layer + 1), repeat=2):
            if abs(dx) == layer or abs(dy) == layer:
                yield (center[0] + dx * step, center[1] + dy * step)

def run_indexamajig(x, y, geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance):

    output_file = f"{output_file_base}_{x}_{y}.stream"
    # output_path = os.path.join(input_path, output_file)
    output_path = os.path.join(output_dir, output_file)
    listfile_path = os.path.join(input_path, 'list.lst')

    base_command = (
        f"indexamajig -g {geomfile_path} -i {listfile_path} -o {output_path} "
        f"-j {num_threads} -p {cellfile_path} --indexing={indexing_method} "
        f"--push-res={resolution_push} --no-retry --no-revalidate --multi "
        f"--integration={integration_method} --no-half-pixel-shift "
        f"--int-radius={int_radius} --no-refine --peaks=cxi "
        f"--max-indexer-threads=1 --min-peaks={min_peaks} "
        f"--xgandalf-tolerance={xgandalf_tolerance} "
        f"--xgandalf-sampling-pitch={xgandalf_sampling_pitch} "
        f"--xgandalf-min-lattice-vector-length={xgandalf_min_vector_length} "
        f"--xgandalf-max-lattice-vector-length={xgandalf_max_vector_length} "
        f"--xgandalf-grad-desc-iterations={xgandalf_iterations} "
        f"--tolerance={tolerance} --no-non-hits-in-stream --no-image-data"
    )

    subprocess.run(base_command, shell=True, check=True)

def modify_geometry_file(template_file_path, modified_file_path, x, y):
    """Modify the geometry file with new x, y values."""
    with open(template_file_path, 'r') as file:
        lines = file.readlines()
    
    with open(modified_file_path, 'w') as file:
        for line in lines:
            if line.startswith("p0/corner_x"):
                file.write(f"p0/corner_x = {x}\n")
            elif line.startswith("p0/corner_y"):
                file.write(f"p0/corner_y = {y}\n")
            else:
                file.write(line)

def list_h5_files(input_path):
    # Function to prepare list file
    listfile_path = os.path.join(input_path, 'list.lst')
    
    with open(listfile_path, 'w') as list_file:
        for file in os.listdir(input_path):
            if file.endswith('.h5'):
                list_file.write(os.path.join(input_path, file) + '\n')

def gandalf_iterator(geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance):
    list_h5_files(input_path)

    xdefault = -512
    ydefault = -512
    print(f"Running for initial x={str(xdefault)}, y={str(ydefault)}")
    
    # Start with the specified coordinates (-512, -512)
    try:
        
        
        #modify_geometry_file(-512, -512)
        modify_geometry_file(geomfile_path, geomfile_path, xdefault, ydefault)
        run_indexamajig(xdefault, ydefault, geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance)
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
    xy_pairs = generate_xy_pairs(xdefault, ydefault, step=0.01, layers=2)  # Adjust the number of layers as needed

    for x, y in xy_pairs:
        print(f"Running for x={x}, y={y}")
        try:
            modify_geometry_file(geomfile_path, geomfile_path, x, y)
            run_indexamajig(x, y, geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance)

        except KeyboardInterrupt:
            print("Process interrupted by user.")
            break
        except subprocess.CalledProcessError as e:
            print(f"Error during indexamajig execution: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
