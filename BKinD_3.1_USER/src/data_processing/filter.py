# filter.py

# Third-Party Imports
from tqdm import tqdm

# Utility File Imports
from util.file.aggregate import aggregate_filtered
from util.file.clean_aggregate import clean_aggregate

# Utility Process Imports
from util.process.filter_function import filter_extreme_data


def filter(output_folder, target_percentages, extreme_percent_step, update_progress=None):
    for i, target in enumerate(tqdm(target_percentages, desc="Filtering Away Extreme Data")):
        filter_extreme_data(output_folder, target, extreme_percent_step)

        # Update progress bar if callback is provided
        if update_progress:
            update_progress('Filtering Away Extreme Data', i + 1)

    aggregate_folder = aggregate_filtered(output_folder, target_percentages)
    clean_aggregate(aggregate_folder)
