# objective_function.py

# Third-Party Imports
import numpy as np

# Utility Process Imports
from util.process.calculate_dfm import calculate_dfm

def objective_function(u, sample_df):
    """
    Objective function to minimize the absolute difference between mean and median of DFM.
    """
    dfm_values = calculate_dfm(u, sample_df)
    return abs(np.mean(dfm_values) - np.median(dfm_values))