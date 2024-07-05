# # filter_data_step.py

# def filter_data_step(df, current_mean, cutoff):
#     """
#     Filter data points in a DataFrame based on deviation from the mean DFM value.

#     Parameters:
#     - df (pd.DataFrame): DataFrame containing 'DFM' and 'asu' columns.
#     - current_mean (float): Current mean of the 'DFM' values.
#     - cutoff (float): Cutoff value for filtering.

#     Returns:
#     - remaining_df (pd.DataFrame): DataFrame with data points within the cutoff.
#     - current_filtered (pd.DataFrame): DataFrame with filtered data points.
#     """
#     deviation = abs(df['DFM'] - current_mean)
#     current_filtered = df[deviation > cutoff]
#     remaining_df = df[deviation <= cutoff]
#     return remaining_df, current_filtered

def filter_data_step(df, current_mean, cutoff, min_removals=1):
    """
    Filter data points in a DataFrame based on deviation from the mean DFM value.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'DFM' and 'asu' columns.
    - current_mean (float): Current mean of the 'DFM' values.
    - cutoff (float): Cutoff value for filtering.
    - min_removals (int): Minimum number of data points to remove if filtering is ineffective.

    Returns:
    - remaining_df (pd.DataFrame): DataFrame with data points within the cutoff.
    - current_filtered (pd.DataFrame): DataFrame with filtered data points.
    """
    deviation = abs(df['DFM'] - current_mean)
    current_filtered = df[deviation > cutoff]
    remaining_df = df[deviation <= cutoff]
    
    # Check if fewer than 5 data points are removed
    if len(current_filtered) < 2 and len(df) > min_removals:
        # Adjust the cutoff to ensure at least min_removals data points are removed
        sorted_deviation = deviation.sort_values(ascending=False)
        cutoff = sorted_deviation.iloc[min_removals - 1]  # The deviation value at the min_removals-th largest position
        current_filtered = df[deviation > cutoff]
        remaining_df = df[deviation <= cutoff]
    
    return remaining_df, current_filtered
