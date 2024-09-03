# filter_reflections_randomly.py

import os
import random
from save_filtered_data import save_filtered_data

def filter_reflections_randomly(header_lines, df, num_to_remove, random_folder):
    # Randomly select `num_to_remove` indices
    random_indices = random.sample(df.index.tolist(), num_to_remove)
    
    # Filter out the selected reflections
    filtered_df = df.drop(index=random_indices)

    # Save the filtered data to a new file
    output_file_path = os.path.join(random_folder, 'XDS_ASCII_filtered.HKL')
    save_filtered_data(header_lines, filtered_df, output_file_path)
    # print(f"Randomly filtered data saved to {output_file_path}")
