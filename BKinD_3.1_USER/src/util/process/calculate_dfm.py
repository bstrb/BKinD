# calculate_dfm.py

# Third-Party Imports
import numpy as np

def calculate_dfm(u, sample_df):
    """
    Calculates the DFM values based on the given instability factor 'u'.
    """
    return (sample_df['Fo^2'] - sample_df['Fc^2']) / np.sqrt(sample_df['Fo^2_sigma']**2 + (2*u * sample_df['Fc^2'])**2)
    
    # print('Using DFM equation with F_o instead of F_c in denominator')
    # return (sample_df['Fo^2'] - sample_df['Fc^2']) / np.sqrt(sample_df['Fo^2_sigma']**2 + (2*u * sample_df['Fo^2'])**2) # sqrt(P) from Jana