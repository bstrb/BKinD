# -------------------------
# Automate Evaluation and Integration
# -------------------------
import traceback
from tqdm import tqdm
from iqm_process_all_stream_files import process_all_stream_files
from read_stream_write_sol import read_stream_write_sol

def automate_evaluation_and_integration(stream_file_folder, weight_list, lattice):
    for weight in weight_list:
        try:
            wrmsd_weight, cld_weight, cad_weight, np_weight, nr_weight, pres_weight, dres_weight, pr_weight, ipp_weight = weight

            IQM = f"IQM_SUM_{wrmsd_weight}_{cld_weight}_{cad_weight}_{np_weight}_{nr_weight}_{pres_weight}_{dres_weight}_{pr_weight}_{ipp_weight}"

            # Evaluate multiple stream files
            print(f"Evaluating multiple stream files with weights: {weight}")
            process_all_stream_files(stream_file_folder, weight)

            # # Write stream file to .sol file
            best_results_stream = f"{stream_file_folder}/best_results_{IQM}.stream"
            tqdm.write(f"Writing .sol file from stream: {best_results_stream}")
            num_indexed_frames = read_stream_write_sol(best_results_stream, lattice)
        
        except Exception as e:
            tqdm.write(f"An error occurred during processing of weights {weight}: {e}")
            traceback.print_exc()
