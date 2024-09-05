# process_filtering_results.py

# Standard Library Imports
import os

# Third-Party Imports
from tqdm import tqdm

# Utility File Imports
from util.file.manage_files import manage_files
from util.file.rem_merg_zero import rem_merg_zero
from util.file.rem_anis import rem_anis
from util.file.lock_atom_positions import lock_atom_positions
from util.file.isotropic_refinement import isotropic_refinement

# Utility Process Imports
from util.process.run_process import run_process

# ED Imports
from ed.modify_xds_inp import modify_xds_inp
from ed.create_xdsconv import create_xdsconv
from ed.copy_and_reduce_hkl import copy_and_reduce_hkl

# X-ray Imports
from xray.convert_csv_to_hkl import convert_csv_to_hkl

def process_filtering_results(output_folder, target_percentages, xds_directory, xray, update_progress=None):
    def process_target(target, target_directory):

        # Define directories for remaining and removed data
        removed_target_dir = os.path.join(target_directory, f'removed_data_{target}')
        remaining_target_dir = os.path.join(target_directory, f'remaining_data_{target}')

        # Move the CSV files to their respective subfolders
        manage_files('move', target_directory, removed_target_dir, 'removed_data.csv')
        manage_files('move', target_directory, remaining_target_dir, 'remaining_data.csv')

        # Process both remaining and removed data directories
        for data_dir in [removed_target_dir, remaining_target_dir]:
            if xray:
                convert_csv_to_hkl(data_dir)
            else:
                copy_and_reduce_hkl(output_folder, data_dir)
                manage_files('copy', xds_directory, data_dir, filename='xds.inp')
                modify_xds_inp(data_dir)
                create_xdsconv(data_dir)
                run_process(["xds"], data_dir, suppress_output=True)
                run_process(["xdsconv"], data_dir, suppress_output=True)

            manage_files('copy', output_folder, data_dir, new_filename=os.path.basename(data_dir) + '.ins', extension='.ins')
            rem_merg_zero(data_dir)

            if data_dir == removed_target_dir:
                rem_anis(data_dir)
                lock_atom_positions(data_dir)
                isotropic_refinement(data_dir)  

            run_process(["shelxl"], data_dir, input_file='.ins', suppress_output=True)

    for i, target in enumerate(tqdm(target_percentages, desc="Processing Filtering Results")):
        target_directory = os.path.join(output_folder, f'filtered_{target}')
        process_target(target, target_directory)

        # Update progress bar if callback is provided
        if update_progress:
            update_progress('Processing Filtering Results', i + 1)
