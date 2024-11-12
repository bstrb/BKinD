import os
import csv

from find_stream_files import find_stream_files
from parse_stream_file import parse_stream_file
from find_nearest_neighbours import find_nearest_neighbours

def extract_rmsd_to_csv(stream_dir, n):
    file_paths = find_stream_files(stream_dir)
    output_csv_path = os.path.join(stream_dir, "rmsd_data.csv")

    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(['stream_file', 'x_coord', 'y_coord', 'serial_number', 'rmsd'])

        # Loop through each stream file
        for file_path in file_paths:
            # filename = file_path.split('/')[-1]
            filename = os.path.basename(file_path) 
            
            # Check if the filename matches the expected pattern
            if filename.count('_') < 2 or not filename.endswith('.stream'):
                print(f"Skipping file {filename} as it doesn't match the expected naming pattern.")
                continue
            
            coords = filename.split('_')[-2:]  # Extract the last two parts as coordinates
            coords[1] = coords[1].replace('.stream', '')
            x, y = abs(float(coords[0])), abs(float(coords[1]))
            
            chunks = parse_stream_file(file_path)
            
            # Process each chunk
            for chunk in chunks:
                serial_number = int(chunk['serial'])
                rmsd = find_nearest_neighbours(chunk['peaks'], chunk['reflections'], n)
                
                if rmsd is not None:
                    # Write the data row to the CSV
                    csv_writer.writerow([filename, x, y, serial_number, rmsd])
    
    print(f"RMSD data has been saved to {output_csv_path}")
