# solve_structure.py
 
# Standard Library Imports
import os

# Third-Party Imports
from tqdm import tqdm

# Utility Process Imports
from util.process.run_process import run_process

# Utility Stats Imports
from util.read.extract_space_group_symbol_from_cif import extract_space_group_symbol_from_cif

def solve_structure(output_folder, target_percentages, solve_remaining=True, solve_removed=True, update_progress=None):
    """
    Solves structures for the remaining or removed data subfolders based on the input parameters.

    Parameters:
    - output_folder (str): The root folder containing the target subfolders.
    - target_percentages (list): A list of target percentages to process.
    - solve_remaining (bool): If True, process the remaining_data subfolders.
    - solve_removed (bool): If True, process the removed_data subfolders.
    - update_progress (callable, optional): A function to update the progress.
    """
    for i, target in enumerate(tqdm(target_percentages, desc="Solving Structure")):
        # Define the directories for remaining and removed data
        remaining_target_dir = os.path.join(output_folder, f'filtered_{target}', f'remaining_data_{target}')
        removed_target_dir = os.path.join(output_folder, f'filtered_{target}', f'removed_data_{target}')

        # Process remaining_data subfolder if solve_remaining is True
        if solve_remaining and os.path.exists(remaining_target_dir):
            # run_process(["shelxl"], remaining_target_dir, input_file='.ins', suppress_output=True)
            sgs = extract_space_group_symbol_from_cif(remaining_target_dir)
            run_process(["shelxt"], remaining_target_dir, input_file='.ins', suppress_output=True, additional_command=f'-s{sgs}')

            if update_progress:
                update_progress('Solving Structure for Remaining Data', i + 1)

        # Process removed_data subfolder if solve_removed is True and .hkl file exists
        hkl_file = os.path.join(removed_target_dir, f'removed_data_{target}.hkl')
        if solve_removed and os.path.exists(removed_target_dir) and os.path.exists(hkl_file):
            # run_process(["shelxl"], removed_target_dir, input_file='.ins', suppress_output=True)
            sgs = extract_space_group_symbol_from_cif(removed_target_dir)
            run_process(["shelxt"], removed_target_dir, input_file='.ins', suppress_output=True, additional_command=f'-s{sgs}')

            if update_progress:
                update_progress('Solving Structure for Removed Data', i + 1)
