# # extract.py

# Standard Library Imports
import os

# Third-Party Imports
from tqdm import tqdm

# Utility Process Imports
# from util.process.run_process import run_process

# Utility Stats Imports
from util.stats.extract_stats import extract_stats
from util.stats.merge_stats_sections import merge_stats_sections
from util.stats.append_refinement_stats import append_refinement_stats

def extract_stats_from_filtering(output_folder, target_percentages, update_progress=None):
    for i, target in enumerate(tqdm(target_percentages, desc="Extracting Stats From Filtering")):
        target_directory = os.path.join(output_folder, f'filtered_{target}')
        remaining_target_dir = os.path.join(target_directory, f'remaining_data_{target}')
        extract_stats(remaining_target_dir)

        # Update progress bar if callback is provided
        if update_progress:
            update_progress('Extracting Stats From Filtering', i + 1)

    append_refinement_stats(output_folder, target_percentages)
    merge_stats_sections(output_folder)
