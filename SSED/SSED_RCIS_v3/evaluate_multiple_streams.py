import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from find_stream_files import find_stream_files
from parse_stream_file import parse_stream_file
from process_stream_file import process_stream_file

# Function to parse multiple stream files, evaluate indexing, and create a combined output file
def evaluate_multiple_streams(stream_file_folder, wrmsd_exp, cld_exp, cad_exp, np_exp, nr_exp, pr_exp):
    try:
        stream_file_paths = [path for path in find_stream_files(stream_file_folder) if not os.path.basename(path).startswith("best_results")]
        all_metrics = []
        header = None
        output_file_path = os.path.join(stream_file_folder, f'best_results_RCIS_{wrmsd_exp}_{cld_exp}_{cad_exp}_{np_exp}_{nr_exp}_{pr_exp}.stream')

        # Remove existing best_results.stream if it exists
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        # Initialize progress tracking
        manager = Manager()
        progress_queue = manager.Queue()
        total_chunks = sum(len(parse_stream_file(path)[1]) - 1 for path in stream_file_paths)

        # Launch progress bar updater
        with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as progress_bar:
            def progress_updater():
                while True:
                    item = progress_queue.get()
                    if item == "STOP":
                        break
                    progress_bar.update(item)

            from threading import Thread
            progress_thread = Thread(target=progress_updater, daemon=True)
            progress_thread.start()

            # Process stream files in parallel
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(process_stream_file, path, wrmsd_exp, cld_exp, cad_exp, np_exp, nr_exp, pr_exp, progress_queue): path for path in stream_file_paths}
                for future in futures:
                    try:
                        current_header, metrics = future.result()
                        if header is None and current_header is not None:
                            header = current_header
                        all_metrics.extend(metrics)
                    except Exception as e:
                        tqdm.write(f"Error processing stream file {futures[future]}: {e}")

            # Stop the progress updater thread
            progress_queue.put("STOP")
            progress_thread.join()

        # Sort all metrics by combined metric value in ascending order
        all_metrics.sort(key=lambda x: x[1])

        # Write the combined metrics and corresponding chunks with the lowest scores to the output file
        with open(output_file_path, 'w') as output_file:
            if header is not None:
                output_file.write(header + '\n')  # Write the header from any of the stream files
            written_events = set()

            for event_number, _, chunk in all_metrics:
                if event_number not in written_events:
                    output_file.write("----- Begin chunk -----\n")
                    output_file.write(chunk)
                    written_events.add(event_number)

        print(f'Combined metrics and selected chunks written to {output_file_path}')
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")