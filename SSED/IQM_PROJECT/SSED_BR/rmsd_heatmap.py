# rmsd_heatmap.py

from rmsd_analysis_frame import rmsd_analysis_frame
from find_stream_files import find_stream_files

def rmsd_heatmap(folder_path, target_serial_number=6186):

    file_paths = find_stream_files(folder_path)

    # Exclude 'best_results.stream' from processing
    file_paths = [file_path for file_path in file_paths if not file_path.endswith('best_results.stream')]
    
    print("Processing the following stream files:", file_paths)

    n = 10  # Number of nearest neighbors to use for RMSD calculation

    # Call the updated rmsd_analysis function with the specific image serial number
    rmsd_analysis_frame(file_paths, n, target_serial_number)