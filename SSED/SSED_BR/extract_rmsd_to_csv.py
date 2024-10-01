import csv
from parse_stream_file import parse_stream_file
from find_nearest_neighbours import find_nearest_neighbours

def extract_rmsd_to_csv(file_paths, n, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(['stream_file', 'x_coord', 'y_coord', 'serial_number', 'rmsd'])

        # Loop through each stream file
        for file_path in file_paths:
            filename = file_path.split('/')[-1]
            
            # Check if the filename matches the expected pattern
            if filename.count('_') < 2 or not filename.endswith('.stream'):
                print(f"Skipping file {filename} as it doesn't match the expected naming pattern.")
                continue
            
            coords = filename.split('_')[1:3]
            coords[1] = coords[1].replace('.stream', '')
            x, y = float(coords[0]), float(coords[1])
            
            chunks = parse_stream_file(file_path)
            
            # Process each chunk
            for chunk in chunks:
                serial_number = int(chunk['serial'])
                rmsd = find_nearest_neighbours(chunk['peaks'], chunk['reflections'], n)
                
                if rmsd is not None:
                    # Write the data row to the CSV
                    csv_writer.writerow([filename, x, y, serial_number, rmsd])
    
    print(f"RMSD data has been saved to {output_csv_path}")
