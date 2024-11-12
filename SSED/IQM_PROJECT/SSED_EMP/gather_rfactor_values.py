import os
import re
import pandas as pd

def extract_final_rfactor_from_txt(file_path):
    """
    Extracts the final Rfactor value from a text file.
    """
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(r'R factor\s+\d+\.\d+\s+(\d+\.\d+)', content)
        if match:
            return float(match.group(1))
    return None

def gather_rfactor_values(directory):
    """
    Gathers the final Rfactor values from all relevant folders and files.
    """
    rfactor_data = []

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
                        rfactor_value = extract_final_rfactor_from_txt(txt_file_path)

                        if rfactor_value is not None:
                            rfactor_data.append({
                                'Head Folder': head_folder,
                                'Rfactor': rfactor_value
                            })

    # Convert the gathered data to a DataFrame and sort by 'Rfactor'
    rfactor_df = pd.DataFrame(rfactor_data)
    rfactor_df = rfactor_df.sort_values(by='Rfactor')  # Sort by 'Rfactor' value
    print(rfactor_df)
    return rfactor_df

# Example usage
directory = "/home/buster/UOX1/different_index_params/5x5_retry"
csv_output = os.path.join(directory, "rfactor_values.csv")
rfactor_table = gather_rfactor_values(directory)
rfactor_table.to_csv(csv_output, index=False)  # Save the table to a CSV file
