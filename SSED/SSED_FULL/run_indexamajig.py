# run_indexamajig.py

import os
import subprocess

# def run_indexamajig(x, y, geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch, xgandalf_min_vector_length, xgandalf_max_vector_length, xgandalf_iterations, tolerance):

#     output_file = f"{output_file_base}_{x}_{y}.stream"
#     output_path = os.path.join(output_dir, output_file)
#     listfile_path = os.path.join(input_path, 'list.lst')

#     base_command = (
#         f"indexamajig -g {geomfile_path} -i {listfile_path} -o {output_path} "
#         f"-j {num_threads} -p {cellfile_path} --indexing={indexing_method} "
#         f"--push-res={resolution_push} --no-retry --no-revalidate "
#         f"--integration={integration_method} --no-half-pixel-shift "
#         f"--int-radius={int_radius} --no-refine --peaks=cxi "
#         f"--max-indexer-threads=1 --min-peaks={min_peaks} "
#         f"--xgandalf-tolerance={xgandalf_tolerance} "
#         f"--xgandalf-sampling-pitch={xgandalf_sampling_pitch} "
#         f"--xgandalf-min-lattice-vector-length={xgandalf_min_vector_length} "
#         f"--xgandalf-max-lattice-vector-length={xgandalf_max_vector_length} "
#         f"--xgandalf-grad-desc-iterations={xgandalf_iterations} "
#         f"--tolerance={tolerance} --no-non-hits-in-stream --no-image-data"
#     )

#     subprocess.run(base_command, shell=True, check=True)

def run_indexamajig(x, y, geomfile_path, cellfile_path, input_path, output_file_base, output_dir, num_threads, indexing_method, resolution_push, integration_method, int_radius, min_peaks, xgt, xgsp, xgi, tolerance):

    output_file = f"{output_file_base}_{x}_{y}.stream"
    output_path = os.path.join(output_dir, output_file)
    listfile_path = os.path.join(input_path, 'list.lst')

    base_command = (
        f"indexamajig -g {geomfile_path} -i {listfile_path} -o {output_path} -p {cellfile_path} "
        f"-j {num_threads} --indexing={indexing_method} "#--integration={integration_method} --int-radius={int_radius} "
        f"--tolerance={tolerance} --min-peaks={min_peaks} "#--push-res={resolution_push} "
        f"--xgandalf-tolerance={xgt} --xgandalf-sampling-pitch={xgsp} --xgandalf-grad-desc-iterations={xgi} "
        f"--no-half-pixel-shift --peaks=cxi --max-indexer-threads=1 --no-non-hits-in-stream --no-image-data "
        f"--no-refine --no-revalidate --no-check-peaks " # --no-check-cell  --no-retry 
    )

    subprocess.run(base_command, shell=True, check=True)
