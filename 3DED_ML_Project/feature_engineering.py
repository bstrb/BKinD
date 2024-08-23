# feature_engineering.py

import numpy as np
import pandas as pd

def feature_engineering(data):
    features = {}

    # Extracting individual columns for easier access
    hkl = np.array([x[0] for x in data])  # Miller indices
    iobs = np.array([x[1] for x in data], dtype=np.float64)  # Observed intensity
    sigma_iobs = np.array([x[2] for x in data], dtype=np.float64)  # Standard deviation of IOBS
    xd, yd, zd = np.array([x[3:6] for x in data]).T  # Detector coordinates
    rlp = np.array([x[6] for x in data], dtype=np.float64)  # Reciprocal lattice position
    peak = np.array([x[7] for x in data], dtype=np.int32)  # Peak value
    corr = np.array([x[8] for x in data], dtype=np.int32)  # Correlation coefficient
    psi = np.array([x[9] for x in data], dtype=np.float64)  # Azimuthal angle

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

    # Convert the dictionary to a pandas DataFrame for easier manipulation later
    return pd.DataFrame(features)
