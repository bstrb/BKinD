# prepare_data.py

import pandas as pd

def prepare_data(file_path):
    # Initialize variables to collect header lines and data rows
    header_lines = []
    data_rows = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        header_passed = False

        for line in file:
            # Collect header lines until we reach the end of the header
            if not header_passed:
                header_lines.append(line)
                if line.startswith('!END_OF_HEADER'):
                    header_passed = True
                continue

            # Break if we reach the end of data marker
            if line.startswith('!END_OF_DATA'):
                break

            # Split the line into fields and append to data_rows
            fields = line.split()
            data_rows.append(fields)

    # Convert the data to a DataFrame
    df = pd.DataFrame(data_rows, columns=[
        'h', 'k', 'l', 'iobs', 'sigma_iobs', 'xd', 'yd', 'zd', 
        'rlp', 'peak', 'corr', 'psi', 'cbi'
    ])
    
    # Convert columns to appropriate data types
    df = df.astype({
        'h': int, 'k': int, 'l': int,
        'iobs': float, 'sigma_iobs': float,
        'xd': float, 'yd': float, 'zd': float,
        'rlp': float, 'peak': float, 'corr': float,
        'psi': float, 'cbi': float
    })

    # Calculate Signal-to-Noise Ratio (SNR)
    df['snr'] = df['iobs'] / df['sigma_iobs']

    return df, header_lines
