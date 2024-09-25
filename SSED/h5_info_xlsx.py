import os
import pandas as pd
import h5py

def check_nPeaks(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            if '/entry/data/nPeaks' in f:
                nPeaks = f['/entry/data/nPeaks'][()]
                count_ge_10 = sum(nPeaks >= 10)
                count_ge_25 = sum(nPeaks >= 25)
                count_ge_50 = sum(nPeaks >= 50)
                return len(nPeaks), count_ge_10, count_ge_25, count_ge_50
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return 0, 0, 0, 0

def find_h5_files(directory):
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
                nPeaks_total, nPeaks_ge_10, nPeaks_ge_25, nPeaks_ge_50 = check_nPeaks(file_path)
                h5_files.append([root, file, file_size_gb, nPeaks_total, nPeaks_ge_10, nPeaks_ge_25, nPeaks_ge_50])
    return h5_files

def create_excel(h5_files, output_file):
    columns = ['Folder', 'File Name', 'Size (GB)', 'Frames', 'nPeaks >=10', 'nPeaks >=25', 'nPeaks >=50']
    df = pd.DataFrame(h5_files, columns=columns)
    df.to_excel(output_file, index=False)

# Replace 'your_directory_path' with the path to the directory you want to scan
directory_to_scan = '/home/buster'
output_excel_file = '/mnt/c/Users/bubl3932/Desktop/UOX_h5_files_info.xlsx'

h5_files_info = find_h5_files(directory_to_scan)
create_excel(h5_files_info, output_excel_file)