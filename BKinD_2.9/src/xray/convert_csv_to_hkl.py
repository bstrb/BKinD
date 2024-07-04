# # convert_csv_to_hkl.py

# # Standard Library Imports
# import os

# # Third-Party Imports
# import pandas as pd

# # File Imports
# from util.file.find_file import find_file

# def convert_csv_to_hkl(folder_path):
#     # Define the input CSV file path
#     # csv_file = os.path.join(folder_path, 'remaining_data.csv')
    
#     csv_file = find_file(folder_path,'.csv')

#     # Check if the file exists
#     if not os.path.exists(csv_file):
#         print(f"No CSV file found at {csv_file}")
#         return
    
#     # Load the CSV file
#     df = pd.read_csv(csv_file)
    
#     # Select the relevant columns and process them
#     df['Miller'] = df['Miller'].str.strip('()"')
#     df[['h', 'k', 'l']] = df['Miller'].str.split(',', expand=True)
#     df = df[['h', 'k', 'l', 'Fo^2', 'Fo^2_sigma']]
    
#     # Convert columns to numeric and ensure they are integers
#     df[['h', 'k', 'l']] = df[['h', 'k', 'l']].astype(int)
#     df[['Fo^2', 'Fo^2_sigma']] = df[['Fo^2', 'Fo^2_sigma']].astype(float)
    
#     # Set the formatting for the output .hkl file
#     def format_row(row):
#         h = int(row['h'])
#         k = int(row['k'])
#         l = int(row['l'])
#         Fo2 = row['Fo^2']
#         Fo2_sigma = row['Fo^2_sigma']
#         return f"{h:>4d}{k:>4d}{l:>4d}{Fo2:>8.2f}{Fo2_sigma:>8.2f}\n"
    
#     # Define the output HKL file path
#     folder_name = os.path.basename(folder_path)
#     hkl_file = os.path.join(folder_path, f"{folder_name}.hkl")
    
#     # Write to the .hkl file
#     with open(hkl_file, 'w') as file:
#         for index, row in df.iterrows():
#             file.write(format_row(row))

# convert_csv_to_hkl.py

# Standard Library Imports
import os

# Third-Party Imports
import pandas as pd

# File Imports
from util.file.find_file import find_file

def convert_csv_to_hkl(folder_path):
    # Define the input CSV file path
    csv_file = find_file(folder_path, '.csv')

    # Check if the file exists
    if not os.path.exists(csv_file):
        print(f"No CSV file found at {csv_file}")
        return
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if the dataframe is empty (i.e., contains only columns)
    if df.empty:
        print("CSV file is empty. No .hkl file will be created.")
        return
    
    # Select the relevant columns and process them
    df['Miller'] = df['Miller'].str.strip('()"')
    df[['h', 'k', 'l']] = df['Miller'].str.split(',', expand=True)
    df = df[['h', 'k', 'l', 'Fo^2', 'Fo^2_sigma']]
    
    # Convert columns to numeric and ensure they are integers
    df[['h', 'k', 'l']] = df[['h', 'k', 'l']].astype(int)
    df[['Fo^2', 'Fo^2_sigma']] = df[['Fo^2', 'Fo^2_sigma']].astype(float)
    
    # Set the formatting for the output .hkl file
    def format_row(row):
        h = int(row['h'])
        k = int(row['k'])
        l = int(row['l'])
        Fo2 = row['Fo^2']
        Fo2_sigma = row['Fo^2_sigma']
        return f"{h:>4d}{k:>4d}{l:>4d}{Fo2:>8.2f}{Fo2_sigma:>8.2f}\n"
    
    # Define the output HKL file path
    folder_name = os.path.basename(folder_path)
    hkl_file = os.path.join(folder_path, f"{folder_name}.hkl")
    
    # Write to the .hkl file
    with open(hkl_file, 'w') as file:
        for index, row in df.iterrows():
            file.write(format_row(row))