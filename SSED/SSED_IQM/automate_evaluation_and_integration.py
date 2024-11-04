# -------------------------
# Part 2: Automate Evaluation and Integration
# -------------------------
import os
import traceback
from tqdm import tqdm
from iqm_process_all_stream_files_v2 import process_all_stream_files
from read_stream_write_sol import read_stream_write_sol
from iqm_int_def import fast_integration
from generate_bash_script import generate_bash_script 
from iqm_merge_def import merge_and_write_mtz
from iqm_ref_def import process_run_folders

def automate_evaluation_and_integration(stream_file_folder, exp_list, lattice, ring_size, cellfile_path, pointgroup, num_threads, bins, pdb_file, min_res = 2, iterations = 3):
    for exp in exp_list:
        try:
            wrmsd_exp, cld_exp, cad_exp, np_exp, nr_exp, pres_exp, dres_exp, pr_exp = exp

            IQM = f"IQM_{wrmsd_exp}_{cld_exp}_{cad_exp}_{np_exp}_{nr_exp}_{pres_exp}_{dres_exp}_{pr_exp}"

            # Evaluate multiple stream files
            tqdm.write(f"Evaluating multiple stream files with exponents: {exp}")
            process_all_stream_files(stream_file_folder, exp)
            # evaluate_multiple_streams(stream_file_folder, wrmsd_exp, cld_exp, cad_exp, np_exp, nr_exp, pr_exp)


            # Write stream file to .sol file
            best_results_stream = f"{stream_file_folder}/best_results_{IQM}.stream"
            best_results_sol = f"{stream_file_folder}/best_results_{IQM}.sol"
            tqdm.write(f"Writing .sol file from stream: {best_results_stream}")
            num_indexed_frames = read_stream_write_sol(best_results_stream, lattice)

            # Generate bash script for fast integration
            bash_file_name = f"{IQM}"
            bash_file_path = os.path.join(stream_file_folder, bash_file_name) + ".sh"
            output_stream_format = f"{stream_file_folder}/{IQM}/int.stream"
            
            # tqdm.write(f"Generating bash script: {bash_file_name}")
            generate_bash_script(bash_file_name, stream_file_folder, num_threads, best_results_sol)

            # Run fast integration
            tqdm.write("Running fast integration...")
            fast_integration(bash_file_path, output_stream_format, num_indexed_frames, integration="rings", ring_sizes=ring_size)

            # Merge Fast Integration Results
            integration_output_folder = f"{stream_file_folder}/{IQM}"
            tqdm.write("Merging fast integration results...")
            merge_and_write_mtz(integration_output_folder, cellfile_path, pointgroup, num_threads, iterations)

            # Refine Fast Integration Results
            tqdm.write("Refining fast integration results...")
            process_run_folders(integration_output_folder, pdb_file, bins, min_res)

            #  Completion message
            print(f"Automation process completed successfully for exponents: {exp}")
        except Exception as e:
            tqdm.write(f"An error occurred during processing of exponents {exp}: {e}")
            traceback.print_exc()
