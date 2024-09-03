# filter_reflections.py

import os
from save_filtered_data import save_filtered_data

def filter_reflections(header_lines, df, model, predicted_folder):
    # Predict dynamical effects
    df['predicted'] = model.predict(df[['cbi', 'snr']])
    
    # Identify reflections predicted to be affected by dynamical effects
    removed_df = df[df['predicted'] == 1]
    
    # Print removed reflections
    print("Removed reflections:")
    print(removed_df)

    # Filter out reflections predicted to be affected by dynamical effects
    filtered_df = df[df['predicted'] == 0].drop(columns=['predicted'])
    num_removed = len(df) - len(filtered_df)

    # Save the filtered data to a new file
    output_file_path = os.path.join(predicted_folder, 'XDS_ASCII_filtered.HKL')
    save_filtered_data(header_lines, filtered_df, output_file_path)
    # print(f"Filtered data saved to {output_file_path}")

    return num_removed  # Return the number of removed reflections
