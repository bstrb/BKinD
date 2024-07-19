# refine_wght_progress.py

# Standard Library Imports
import os

# Third-Party Imports
from tqdm import tqdm

# Utility Process Imports
from util.process.run_process import run_process
from util.process.refine_wght import refine_wght


def refine_wght_progress(output_folder, target_percentages, update_progress=None):
    for i, target in enumerate(tqdm(target_percentages, desc="Refining WGHT")):
        target_directory = os.path.join(output_folder, f'filtered_{target}')
        run_process(["shelxl"], target_directory, input_file='.ins', suppress_output=True)
        refine_wght(target_directory)

        # Update progress bar if callback is provided
        if update_progress:
            update_progress('Refining WGHT', i + 1)
