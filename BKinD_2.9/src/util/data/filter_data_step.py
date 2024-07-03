# filter_data_step.py

def filter_data_step(df, current_mean, cutoff):
    """
    Filter data points in a DataFrame based on deviation from the mean DFM value.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'DFM' and 'asu' columns.
    - current_mean (float): Current mean of the 'DFM' values.
    - cutoff (float): Cutoff value for filtering.

    Returns:
    - remaining_df (pd.DataFrame): DataFrame with data points within the cutoff.
    - current_filtered (pd.DataFrame): DataFrame with filtered data points.
    """
    deviation = abs(df['DFM'] - current_mean)
    current_filtered = df[deviation > cutoff]
    remaining_df = df[deviation <= cutoff]
    return remaining_df, current_filtered