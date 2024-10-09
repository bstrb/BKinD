# -------------------------
# Part 2: Automate Evaluation and Integration
# -------------------------
import os
import traceback
from tqdm import tqdm
from evaluate_multiple_streams import evaluate_multiple_streams
from read_stream_write_sol import read_stream_write_sol
from rcis_int_def import fast_integration
from generate_bash_script import generate_bash_script
from rcis_merge_def import merge_and_write_mtz
from rcis_ref_def import process_run_folders

def automate_evaluation_and_integration(stream_file_folder, weights_list, lattice, ring_size, cellfile_path, pointgroup, num_threads, bins, pdb_file):
    for weights in weights_list:
        try:
            wrmsd_weight, cd_weight, rpr_weight = weights

            # Evaluate multiple stream files
            tqdm.write(f"Evaluating multiple stream files with weights: {weights}")
            evaluate_multiple_streams(stream_file_folder, wrmsd_weight, cd_weight, rpr_weight)

            RCIS = f"RCIS_{wrmsd_weight}_{cd_weight}_{rpr_weight}"

            # Write stream file to .sol file
            input_stream_folder = f"{stream_file_folder}/best_results_{RCIS}.stream"
            tqdm.write(f"Writing .sol file from stream: {input_stream_folder}")
            read_stream_write_sol(input_stream_folder, lattice)

            # Generate bash script for fast integration
            bash_file_name = f"fast_int_{RCIS}"
            bash_file_path = os.path.join(stream_file_folder, bash_file_name) + ".sh"
            output_stream_format = f"{stream_file_folder}/fast_int_{RCIS}/fast_int.stream"
            
            # tqdm.write(f"Generating bash script: {bash_file_name}")
            generate_bash_script(bash_file_name, stream_file_folder, num_threads)

            # Run fast integration
            tqdm.write("Running fast integration...")
            fast_integration(bash_file_path, output_stream_format, integration="rings", ring_sizes=ring_size)

            # Merge Fast Integration Results
            input_folder_path = f"{stream_file_folder}/fast_int_{RCIS}"
            tqdm.write("Merging fast integration results...")
            merge_and_write_mtz(input_folder_path, cellfile_path, pointgroup, num_threads)

            # Refine Fast Integration Results
            tqdm.write("Refining fast integration results...")
            process_run_folders(input_folder_path, pdb_file, bins)

            #  Completion message
            print(f"Automation process completed successfully for weights: {weights}")
        except Exception as e:
            tqdm.write(f"An error occurred during processing of weights {weights}: {e}")
            traceback.print_exc()
