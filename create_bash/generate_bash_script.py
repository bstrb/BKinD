import os

def find_first_file(directory, extension):
    # Find the first file in the directory with the specified extension
    for file_name in os.listdir(directory):
        if file_name.endswith(extension):
            return os.path.join(directory, file_name)
    return None

def generate_bash_script(bash_file_name, bash_file_directory,
                        process_output_name, process_file_directory, 
                        num_threads, geom_file=None, lst_file=None, 
                        cell_file=None, input_sol_file='best_results.sol'
                        ):
    # Generate the full path for the bash file
    bash_file_path = f"{bash_file_directory}/{bash_file_name}.sh"

    # If no geom_file, lst_file, or cell_file is provided, find the first one in the directory
    geom_file_path = geom_file or find_first_file(bash_file_directory, ".geom")
    lst_file_path = lst_file or find_first_file(bash_file_directory, ".lst")
    cell_file_path = cell_file or find_first_file(bash_file_directory, ".cell")

    # Error if any of the required files are not found
    if not geom_file_path:
        raise FileNotFoundError("No .geom file found in the directory.")
    if not lst_file_path:
        raise FileNotFoundError("No .lst file found in the directory.")
    if not cell_file_path:
        raise FileNotFoundError("No .cell file found in the directory.")

    # Create the content of the bash file
    bash_script_content = f"""#!/bin/bash
export HDF5_PLUGIN_PATH=$HOME/anaconda3/envs/diffractem2/lib/python3.10/site-packages/hdf5plugin/plugins

indexamajig -g {geom_file_path} -i {lst_file_path} -o {process_file_directory}/{process_output_name}.stream -j {num_threads} -p {cell_file_path} \\
--indexing=file --no-revalidate --no-retry --integration=rings --no-refine \\
--fromfile-input-file={input_sol_file} --no-half-pixel-shift --no-check-cell \\
--pinkIndexer-max-refinement-disbalance=1 --peaks=cxi --max-indexer-threads=1 \\
--min-peaks=25 --camera-length-estimate=1.17 --xgandalf-tolerance=0.02 --xgandalf-sampling-pitch=5 \\
--xgandalf-min-lattice-vector-length=2 --xgandalf-max-lattice-vector-length=40 --xgandalf-grad-desc-iterations=2 \\
--tolerance=2.5,2.5,2.5,4 --no-non-hits-in-stream --pinkIndexer-considered-peaks-count=3 \\
--pinkIndexer-angle-resolution=2 --pinkIndexer-refinement-type=5 --pinkIndexer-tolerance=0.02 \\
--pinkIndexer-reflection-radius=0.003 --pinkIndexer-max-resolution-for-indexing=2
"""

    # Write the content to the bash file
    with open(bash_file_path, 'w') as bash_file:
        bash_file.write(bash_script_content)

    # Make the bash file executable
    os.chmod(bash_file_path, 0o755)

    # Output the full path of the generated bash file
    print(f"Bash script generated: {bash_file_path}")

# Example usage:
bash_file_name = "run_indexing"
bash_file_directory = "/path/to/bash_file"
process_output_name = "process_output"
process_file_directory = "/path/to/process_output"
num_threads = 12

generate_bash_script(bash_file_name, bash_file_directory, process_output_name, process_file_directory, num_threads)
