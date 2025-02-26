# -------------------------
# Automate Evaluation and Integration
# -------------------------

import os
import traceback
from tqdm import tqdm
from iqm_process_all_stream_files import process_all_stream_files
from iqm_merge_def import merge_and_write_mtz
from iqm_ref_def import process_run_folders
from extract_final_rfactor import extract_final_rfactor

def automate_evaluation_and_integration(stream_file_folder, weight_list, cellfile_path, pointgroup, num_threads, bins, pdb_file, min_res = 2, iterations = 3, merging = True, refining = True):
    for weight in weight_list:
        try:
            weight_combination_str = '_'.join([f'{value}' for value in (weight or [])])
            output_stream_path = os.path.join(stream_file_folder, f'IQM_{weight_combination_str}')

            # Evaluate multiple stream files
            print(f"Evaluating multiple stream files with weights: {weight}")
            process_all_stream_files(stream_file_folder, weight)

            if merging:
                # Merge Fast Integration Results
                integration_output_folder = f"{output_stream_path}"
                tqdm.write("Merging fast integration results...")
                merge_and_write_mtz(integration_output_folder, cellfile_path, pointgroup, num_threads, iterations)

            if refining:
                # Refine Fast Integration Results
                tqdm.write("Refining fast integration results...")
                process_run_folders(output_stream_path, pdb_file, bins, min_res)

                rfactor = extract_final_rfactor(output_stream_path)

                #  Completion message
                print(f"Automation process completed successfully for weights: {weight} with a final R factor of {rfactor}")

            # return rfactor
        
        except Exception as e:
            tqdm.write(f"An error occurred during processing of weights {weight}: {e}")
            traceback.print_exc()
