# solve_filtered.py

# Standard Library Imports
import os

# Third-Party Imports
from tqdm import tqdm

# Utility File Imports
from util.file.manage_files import manage_files
from util.file.rem_merg_zero import rem_merg_zero

# Utility Process Imports
from util.process.run_process import run_process

# ED Imports
from ed.modify_xds_inp import modify_xds_inp
from ed.create_xdsconv import create_xdsconv
from ed.copy_and_reduce_hkl import copy_and_reduce_hkl

# X-ray Imports
from xray.convert_csv_to_hkl import convert_csv_to_hkl

def solve_filtered(output_folder, target_percentages, xds_directory, xray, update_progress=None):
    def process_target(target_directory):
        if xray:
            convert_csv_to_hkl(target_directory)
        else:
            copy_and_reduce_hkl(output_folder, target_directory)
            manage_files('copy', xds_directory, target_directory, filename='xds.inp')
            modify_xds_inp(target_directory)
            create_xdsconv(target_directory)
            run_process(["xds"], target_directory, suppress_output=True)
            run_process(["xdsconv"], target_directory, suppress_output=True)

        manage_files('copy', output_folder, target_directory, new_filename='solve_filtered' + '.ins', extension='.ins')
        rem_merg_zero(target_directory)
        run_process(["shelxt"], target_directory, input_file='.ins', suppress_output=True)

    for i, target in enumerate(tqdm(target_percentages, desc="Solving structure for filtered data")):
        target_directory = os.path.join(output_folder, f'filtered_{target}/solve_filtered')
        process_target(target_directory)

        # Update progress bar if callback is provided
        if update_progress:
            update_progress('Solving structure for filtered data', i + 1)