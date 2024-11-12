import os
import re

def extract_final_rfactor(input_folder):
    # Iterate through all subfolders in the given input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            # Check if the file is a text file
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Search for the desired section using regex
                    match = re.search(r'\$TEXT:Result: \$\$ Final results \$\$.*?R factor\s+(\S+)\s+(\S+)', content, re.DOTALL)
                    
                    if match:
                        initial_rfactor = float(match.group(1))
                        final_rfactor = float(match.group(2))
                        # print(f"File: {file_path}\nInitial R factor: {initial_rfactor}\nFinal R factor: {final_rfactor}\n")
                        return final_rfactor

# Example usage
# input_folder = '/home/buster/UOX1/different_index_params/3x3_retry/IQM_1_2_3_-1_1_-2_1_1_-1'  # Replace with your actual path
# extract_final_rfactor(input_folder)
