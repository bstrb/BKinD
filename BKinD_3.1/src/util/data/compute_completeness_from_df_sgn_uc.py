# compute_completeness_from_df_sgn_uc.py

# Standard Library Imports
import pandas as pd

# CCTBX Imports
from cctbx import crystal, miller
from cctbx.array_family import flex
from cctbx.sgtbx import space_group_info
from cctbx.uctbx import unit_cell


def extract_unique_asu_indices(df):
    """
    Extracts the ASU Miller indices from the 'asu' column in the DataFrame and returns
    a DataFrame with columns 'h', 'k', and 'l'.

    Parameters:
    - df: DataFrame, a DataFrame containing the 'asu' column.

    Returns:
    - DataFrame: A DataFrame containing the Miller indices as separate columns 'h', 'k', and 'l'.
    """
    if 'asu' not in df.columns:
        raise ValueError("The DataFrame must contain the column: 'asu'.")

    if df.empty:
        # print("Warning: The DataFrame is empty. Returning an empty DataFrame for Miller indices.")
        return pd.DataFrame(columns=['h', 'k', 'l'])

    # Remove parentheses and split the 'asu' column into separate 'h', 'k', 'l' columns
    miller_indices = df['asu'].str.strip('()').str.split(',', expand=True).astype(int)
    miller_indices.columns = ['h', 'k', 'l']

    return miller_indices

def compute_completeness_from_df_sgn_uc(bkind_df, sgn, uc):
    """
    Computes the completeness of the dataset using CCTBX.

    Args:
    - bkind_df (DataFrame): DataFrame containing the 'asu' column with ASU indices.
    - sgn (int): Space group number.
    - uc (tuple): Unit cell parameters.

    Returns:
    - float: Completeness of the dataset, or 0 if the DataFrame is empty.
    """
    # Extract unit cell parameters
    unit_cell_params = uc

    # Extract Miller indices from the DataFrame
    df = extract_unique_asu_indices(bkind_df)

    if df.empty:
        # print("Warning: No Miller indices found. Returning completeness as 0.")
        return 0  # Or another appropriate default value indicating no completeness

    # Extract space group number from the CIF file
    space_group_number = sgn

    # Define the crystal symmetry
    crystal_symmetry = crystal.symmetry(
        unit_cell=unit_cell(unit_cell_params),
        space_group_info=space_group_info(space_group_number)
    )

    unique_indices = df.drop_duplicates(subset=['h', 'k', 'l'])

    # Create a Miller set from the DataFrame
    miller_set = miller.set(
        crystal_symmetry=crystal_symmetry,
        indices=flex.miller_index(list(zip(unique_indices['h'], unique_indices['k'], unique_indices['l']))),
        anomalous_flag=False
    )

    # Compute the completeness
    completeness = miller_set.completeness()
    return completeness
