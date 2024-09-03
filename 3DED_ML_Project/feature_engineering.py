# feature_engineering.py


import numpy as np
import pandas as pd

def feature_engineering(df):

    features = {}

    try:
        # Extracting individual columns from the DataFrame
        hkl = np.array(df['hkl'].tolist())  # Miller indices (h, k, l)
        iobs = df['iobs'].values.astype(np.float64)  # Observed intensity
        sigma_iobs = df['sigma_iobs'].values.astype(np.float64)  # Standard deviation of IOBS
        xd = df['xd'].values.astype(np.float64)  # Detector coordinate XD
        yd = df['yd'].values.astype(np.float64)  # Detector coordinate YD
        zd = df['zd'].values.astype(np.float64)  # Detector coordinate ZD
        rlp = df['rlp'].values.astype(np.float64)  # Reciprocal lattice position
        peak = df['peak'].values.astype(np.int32)  # Peak value
        corr = df['corr'].values.astype(np.int32)  # Correlation coefficient
        psi = df['psi'].values.astype(np.float64)  # Azimuthal angle
        cbi = df['cbi'].values.astype(np.float64)  # Center Beam Intensity (CBI)
        asu = np.array(df['asu'].tolist())  # ASU as tuples
        resolution = df['resolution'].values.astype(np.float64)  # Resolution

        # Unpacking hkl and asu into separate columns
        features['h'] = hkl[:, 0]
        features['k'] = hkl[:, 1]
        features['l'] = hkl[:, 2]

        features['asu_h'] = asu[:, 0]
        features['asu_k'] = asu[:, 1]
        features['asu_l'] = asu[:, 2]

    except ValueError as e:
        print(f"Error converting data to numpy arrays: {e}")
        raise

    # Normalize intensities
    mean_iobs = np.mean(iobs)
    features['normalized_iobs'] = iobs / mean_iobs

    # Calculate SNR
    features['snr'] = iobs / (sigma_iobs + 1e-10)  # Adding small value to avoid division by zero

    # Handle cases where SNR might be negative or zero (if such situations are possible)
    snr = features['snr']
    snr = np.where(snr <= 0, 1e-10, snr)  # Replace zero or negative values with a small positive constant
    features['log_snr'] = np.log(snr + 1)

    # Logarithmic transformations
    iobs = np.where(iobs <= 0, 1e-10, iobs)  # Replace zero or negative values with a small positive constant
    features['log_iobs'] = np.log(iobs)
    
    # Add original features to the dictionary
    features['iobs'] = iobs
    features['sigma_iobs'] = sigma_iobs
    features['xd'] = xd
    features['yd'] = yd
    features['zd'] = zd
    features['rlp'] = rlp
    features['peak'] = peak
    features['corr'] = corr
    features['psi'] = psi
    features['cbi'] = cbi
    features['resolution'] = resolution

    # Convert the dictionary to a pandas DataFrame for easier manipulation later
    return pd.DataFrame(features)

def calculate_structure_factors(df):
    """
    Simplified estimation of structure factors based on Iobs.
    In practice, this may involve more complex calculations.
    """
    df['Fhkl'] = np.sqrt(df['iobs'])  # Simplified: Fhkl ~ sqrt(Iobs)
    return df

def apply_sayre_equation(df):
    """
    Apply the Sayre equation to estimate Fhkl based on other reflections.
    """
    df = calculate_structure_factors(df)
    sayre_fhkl = []

    # Iterate over each reflection in df
    for idx, row in df.iterrows():
        h, k, l = row['h'], row['k'], row['l']
        fhkl_sum = 0

        # Iterate over all possible h', k', l' combinations
        for _, row_prime in df.iterrows():
            h_prime, k_prime, l_prime = row_prime['h'], row_prime['k'], row_prime['l']
            
            # Find the reflection h-h', k-k', l-l'
            match = df[
                (df['h'] == h - h_prime) &
                (df['k'] == k - k_prime) &
                (df['l'] == l - l_prime)
            ]
            if not match.empty:
                fhkl_sum += row_prime['Fhkl'] * match.iloc[0]['Fhkl']

        sayre_fhkl.append(fhkl_sum)
    
    df['Sayre_Fhkl'] = sayre_fhkl
    return df

def feature_engineering_with_sayre(data):
    """
    Enhanced feature engineering including the Sayre equation.
    """
    df = feature_engineering(data)
    df = apply_sayre_equation(df)
    
    return df
