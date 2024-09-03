# parse_xds_ascii.py

import pandas as pd

def prepare_dataframe(features):
    df = pd.DataFrame(features, columns=[
        'hkl', 'iobs', 'sigma_iobs', 'xd', 'yd', 'zd', 
        'rlp', 'peak', 'corr', 'psi', 'cbi', 'asu', 'resolution'
    ])
    return df

def parse_xds_ascii(file_path):
    data = []
    with open(file_path, 'r') as f:
        # Skip lines until the end of the header
        for line in f:
            if line.startswith('!END_OF_HEADER'):
                break
        
        # Now read the data section
        for line in f:
            if line.startswith('!END_OF_DATA'):
                break

            fields = line.split()
            
            try:
                # Parsing each field according to expected data types
                hkl = tuple(map(int, fields[:3]))  # Miller indices (h, k, l)
                iobs = float(fields[3])  # IOBS
                sigma_iobs = float(fields[4])  # SIGMA(IOBS)
                xd, yd, zd = map(float, fields[5:8])  # Detector coordinates (XD, YD, ZD)
                rlp = float(fields[8])  # Reciprocal lattice position (RLP)
                peak = float(fields[9])  # Peak value
                corr = float(fields[10])  # Correlation coefficient (CORR)
                psi = float(fields[11])  # Azimuthal angle (PSI)
                cbi = float(fields[12])  # Center Beam Intensity (CBI)
                asu = tuple(map(int, fields[13:16]))  # ASU indices (ASU_H, ASU_K, ASU_L)
                resolution = float(fields[16])  # Resolution (RES)
                
                # Append the parsed data as a list to the main data list
                data.append([hkl, iobs, sigma_iobs, xd, yd, zd, rlp, peak, corr, psi, cbi, asu, resolution])
            
            except ValueError as e:
                print(f"Error parsing line: {line.strip()}")
                print(f"Exception: {e}")
                continue  # Skip this line and move to the next

    # Convert to a pandas DataFrame for further processing
    df = prepare_dataframe(data)
    
    return df
