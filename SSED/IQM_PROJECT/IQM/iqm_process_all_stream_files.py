import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock

from process_and_store import process_and_store
from read_stream_write_sol import read_stream_write_sol

# Function to process all stream files in a folder using multiprocessing
def process_all_stream_files(folder_path):
    manager = Manager()
    all_results = manager.list()
    best_results = manager.list()
    header = manager.list()
    lock = manager.Lock()

    # Remove existing best_results stream files in the folder
    for f in os.listdir(folder_path):
        if f.startswith('best_results') and f.endswith('.stream'):
            os.remove(os.path.join(folder_path, f))

    # Iterate over all stream files in the folder
    stream_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.stream')]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_and_store, stream_file, all_results, best_results, header, lock): stream_file for stream_file in stream_files}
        for future in as_completed(futures):
            futures.pop(future)

    # Sort best_results by combined metric value in ascending order
    best_results = list(best_results)
    best_results.sort(key=lambda x: x[2])

    # Create a filename for the output files based on the weight combination
    output_csv_path = os.path.join(folder_path, f'IQM.csv')
    output_stream_path = os.path.join(folder_path, f'best_results_IQM.stream')

    # Write all results to CSV
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['stream_file', 'event_number', 'combined_metric'])
        for result in all_results:
            csv_writer.writerow(result[:3])

    # Write best results to a stream file (keeping all unique event numbers with the lowest combined metric)
    if best_results and header:
        with open(output_stream_path, 'w') as stream_file:
            stream_file.write(header[0])  # Write the header to the output stream file
            for result in best_results:
                stream_file.write("----- Begin chunk -----\n")
                stream_file.write(result[3])
                stream_file.write("----- End chunk -----\n")

        print(f'Combined metrics CSV written to {output_csv_path}')
        print(f'Best results stream file written to {output_stream_path}')
    else:
        print("No valid chunks found in any stream file.")
        
    lines_written = read_stream_write_sol(output_stream_path)

# Example usage
if __name__ == "__main__":
    folder_path = "/home/buster/hMTH1_TH287"
    process_all_stream_files(folder_path)
