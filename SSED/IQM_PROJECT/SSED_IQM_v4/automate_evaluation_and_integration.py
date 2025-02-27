# -------------------------
# Automate Evaluation and Integration
# -------------------------
import os
import traceback
from tqdm import tqdm
from iqm_process_all_stream_files import process_all_stream_files
from read_stream_write_sol import read_stream_write_sol
from iqm_int_def import fast_integration
from generate_bash_script import generate_bash_script 
from iqm_merge_def import merge_and_write_mtz
from iqm_ref_def import process_run_folders
from extract_final_rfactor import extract_final_rfactor

def automate_evaluation_and_integration(stream_file_folder, weight_list, lattice, ring_size, cellfile_path, pointgroup, num_threads, bins, pdb_file, min_res = 2, iterations = 3, integration = True, merging = True, refining = True):
    for weight in weight_list:
        try:
            wrmsd_weight, cld_weight, cad_weight, np_weight, nr_weight, pres_weight, dres_weight, pr_weight, ipp_weight = weight

            IQM = f"IQM_SUM_{wrmsd_weight}_{cld_weight}_{cad_weight}_{np_weight}_{nr_weight}_{pres_weight}_{dres_weight}_{pr_weight}_{ipp_weight}"

            # Evaluate multiple stream files
            print(f"Evaluating multiple stream files with weights: {weight}")
            process_all_stream_files(stream_file_folder, weight)

            # # Write stream file to .sol file
            best_results_stream = f"{stream_file_folder}/best_results_{IQM}.stream"
            best_results_sol = f"{stream_file_folder}/best_results_{IQM}.sol"
            tqdm.write(f"Writing .sol file from stream: {best_results_stream}")
            num_indexed_frames = read_stream_write_sol(best_results_stream, lattice)

            if integration:
                # # Generate bash script for fast integration
                bash_file_name = f"{IQM}"
                bash_file_path = os.path.join(stream_file_folder, bash_file_name) + ".sh"
                output_stream_format = f"{stream_file_folder}/{IQM}/int.stream"
                
                # # tqdm.write(f"Generating bash script: {bash_file_name}")
                generate_bash_script(bash_file_name, stream_file_folder, num_threads, best_results_sol)

                # Run fast integration
                tqdm.write("Running fast integration...")
                fast_integration(bash_file_path, output_stream_format, num_indexed_frames, integration="rings", ring_sizes=ring_size)

            if merging:
                # Merge Fast Integration Results
                integration_output_folder = f"{stream_file_folder}/{IQM}"
                tqdm.write("Merging fast integration results...")
                merge_and_write_mtz(integration_output_folder, cellfile_path, pointgroup, num_threads, iterations)

            if refining:
                # Refine Fast Integration Results
                tqdm.write("Refining fast integration results...")
                process_run_folders(integration_output_folder, pdb_file, bins, min_res)

                rfactor = extract_final_rfactor(integration_output_folder)

                #  Completion message
                print(f"Automation process completed successfully for weights: {weight} with a final R factor of {rfactor}")

            # return rfactor
        
        except Exception as e:
            tqdm.write(f"An error occurred during processing of weights {weight}: {e}")
            traceback.print_exc()
