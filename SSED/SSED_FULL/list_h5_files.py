# list_h5_files.py

import os

def list_h5_files(input_path):
    # Function to prepare list file
    listfile_path = os.path.join(input_path, 'list.lst')
    
    with open(listfile_path, 'w') as list_file:
        for file in os.listdir(input_path):
            if file.endswith('.h5'):
                list_file.write(os.path.join(input_path, file) + '\n')