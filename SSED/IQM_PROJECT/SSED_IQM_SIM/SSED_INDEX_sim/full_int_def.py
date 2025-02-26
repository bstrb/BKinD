# full_int_def.py

import os
import subprocess

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

def run_indexamajig_with_frames(x, y, geomfile_path, cellfile_path, input_path, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_iterations, tolerance):
    modify_geometry_file(geomfile_path, geomfile_path, x, y)
    
    try:
        listfile_path = next((os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.lst')), None)
        solfile_path = next((os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.sol')), None)
        if not listfile_path or not solfile_path:
            raise FileNotFoundError("Required .list or .sol file not found.")
        
        _, output_file_base = os.path.split(input_path)
        output_file = f"{output_file_base}_{x}_{y}_pushres{resolution_push}_rings_{int_radius.replace(',', '-')}.stream"
        output_path = os.path.join(input_path, output_file)

        base_command = (
            f"indexamajig -g {geomfile_path} -i {listfile_path} -o {output_path} -j {num_threads} -p {cellfile_path} --indexing={indexing_method} --no-revalidate --no-check-peaks --no-retry --push-res={resolution_push} --integration={integration_method} --overpredict --no-refine --fromfile-input-file={solfile_path} --no-half-pixel-shift --int-radius={int_radius} --no-check-cell --peaks=cxi --max-indexer-threads=4 --min-peaks={min_peaks} --xgandalf-tolerance={xgandalf_tolerance} --xgandalf-sampling-pitch={xgandalf_sampling_pitch} --xgandalf-grad-desc-iterations={xgandalf_iterations} --tolerance={tolerance} --no-non-hits-in-stream"
        )

        print(base_command)

        subprocess.run(base_command, shell=True, check=True)
    except Exception as e:
        print(f"An error occurred: {e}")