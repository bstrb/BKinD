# solve_remaining.py

# Standard Library Imports
import os

# Third-Party Imports
from tqdm import tqdm

# Utility Process Imports
from util.process.run_process import run_process

# Utility Stats Imports
from util.read.extract_space_group_symbol_from_cif import extract_space_group_symbol_from_cif

def solve_remaining(output_folder, target_percentages, update_progress=None):
    for i, target in enumerate(tqdm(target_percentages, desc="Solving Structure for Remaining Data")):
        target_directory = os.path.join(output_folder, f'filtered_{target}')
        run_process(["shelxl"], target_directory, input_file='.ins', suppress_output=True)
        sgs = extract_space_group_symbol_from_cif(target_directory)
        run_process(["shelxt"], target_directory, input_file='.ins', suppress_output=True, additional_command=f'-s{sgs}')

        # Update progress bar if callback is provided
        if update_progress:
            update_progress('Solving Structure for Remaining Data', i + 1)