import os
import re
import pandas as pd

def extract_final_rfree_from_txt(file_path):
    """
    Extracts the final Rfree value from a text file.
    """
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(r'R free\s+\d+\.\d+\s+(\d+\.\d+)', content)
        if match:
            return float(match.group(1))
    return None

def gather_rfree_values(directory):
    """
    Gathers the final Rfree values from all relevant folders and files.
    """
    rfree_data = []

    # Iterate through each folder in the given directory
    for root, dirs, files in os.walk(directory):
        head_folder = os.path.basename(root)

        # Check if the folder is a head folder and contains subfolders starting with "merge"
        for dir_name in dirs:
            if dir_name.startswith("merge"):
                merge_folder_path = os.path.join(root, dir_name)

                # Look for .txt files in the "merge" folder
                for file_name in os.listdir(merge_folder_path):
                    if file_name.endswith(".txt"):
                        txt_file_path = os.path.join(merge_folder_path, file_name)
                        rfree_value = extract_final_rfree_from_txt(txt_file_path)

                        if rfree_value is not None:
                            rfree_data.append({
                                'Head Folder': head_folder,
                                'Rfree': rfree_value
                            })

    # Convert the gathered data to a DataFrame and sort by 'Rfree'
    rfree_df = pd.DataFrame(rfree_data)
    rfree_df = rfree_df.sort_values(by='Rfree')  # Sort by 'Rfree' value
    print(rfree_df)
    return rfree_df

# Example usage
directory = "/home/buster/UOXm/5x5_0-01"
csv_output = os.path.join(directory, "rfree_values.csv")
rfree_table = gather_rfree_values(directory)
rfree_table.to_csv(csv_output, index=False)  # Save the table to a CSV file
