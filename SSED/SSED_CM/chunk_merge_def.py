import os
import glob
import re
import subprocess

def run_partialator(stream_file, output_dir, num_threads, pointgroup):
    """Run the partialator command to process a stream file."""
    merging_cmd = [
        'partialator',
        stream_file,
        '--model=offset',
        '-j', f'{num_threads}',
        '-o', os.path.join(output_dir, "crystfel.hkl"),
        '-y', pointgroup,
        '--polarisation=none',
        '--min-measurements=2',
        '--max-adu=inf',
        '--min-res=inf',
        '--push-res=inf',
        '--no-Bscale',
        '--no-logs',
        '--iterations=3',
        '--harvest-file=' + os.path.join(output_dir, "parameters.json"),
        '--log-folder=' + os.path.join(output_dir, "pr-logs")
    ]

    try:
        with open(os.path.join(output_dir, "stdout.log"), "a") as stdout, open(os.path.join(output_dir, "stderr.log"), "a") as stderr:
            print(f"Running partialator for stream file: {stream_file}")
            subprocess.run(merging_cmd, stdout=stdout, stderr=stderr, check=True)
            print(f"Partialator completed for stream file: {stream_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during partialator execution for {stream_file}: {e}")
        raise

def convert_hkl_to_mtz(output_dir, cellfile_path):
    """Convert the crystfel.hkl file to output.mtz using get_hkl."""
    hkl2mtz_cmd = [
        'get_hkl',
        '-i', os.path.join(output_dir, "crystfel.hkl"),
        '-o', os.path.join(output_dir, "output.mtz"),
        '-p', f'{cellfile_path}',
        '--output-format=mtz'
    ]

    try:
        with open(os.path.join(output_dir, "stdout.log"), "a") as stdout, open(os.path.join(output_dir, "stderr.log"), "a") as stderr:
            print(f"Converting crystfel.hkl to output.mtz in directory: {output_dir}")
            subprocess.run(hkl2mtz_cmd, stdout=stdout, stderr=stderr, check=True)
            print(f"Conversion to output.mtz completed for directory: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion to MTZ in {output_dir}: {e}")
        raise

import time

def chunk_merge_and_write_mtz(stream_dir, cellfile_path, pointgroup, num_threads):
    """
    Merge chunked stream files and run partialator and conversion processes for each.
    
    Parameters:
    - stream_dir: Directory containing the chunked stream files.
    - cellfile_path: Path to the cell file.s
    - pointgroup: Point group for the partialator command.
    - num_threads: Number of threads to use.
    """
    # stream_files = glob.glob(f'{stream_dir}/*.stream')
    stream_files = sorted(glob.glob(f'{stream_dir}/*.stream'), key=lambda x: int(re.search(r'chunk_(\d+)\.stream', x).group(1)))
    total_files = len(stream_files)

    if total_files == 0:
        print(f"No stream files found in directory: {stream_dir}")
        return

    for index, stream_file in enumerate(stream_files):
        
        # added for time
        start_time = time.time()

        # Extract the chunk size from the file name assuming it contains "chunk_X" where X is the chunk size
        chunk_size_match = re.search(r'chunk_(\d+)\.stream', stream_file)
        if not chunk_size_match:
            print(f"Skipping file {stream_file} as it does not match the expected naming pattern.")
            continue

        chunk_size = int(chunk_size_match.group(1))

        # Create output directory named after the chunk size
        output_dir = os.path.join(stream_dir, f"merge_chunk_{chunk_size}")
        os.makedirs(output_dir, exist_ok=True)

        # Create a directory for partialator logs
        os.makedirs(os.path.join(output_dir, "pr-logs"), exist_ok=True)

        # Execute the merging and conversion processes
        try:
            print(f"[{index + 1}/{total_files}] Running merging for chunk size: {chunk_size}")
            run_partialator(stream_file, output_dir, num_threads, pointgroup)
            convert_hkl_to_mtz(output_dir, cellfile_path)
            print(f"Completed merging for chunk size: {chunk_size}")
        except subprocess.CalledProcessError:
            print(f"Skipping further steps for {stream_file} due to error.")
            continue

        # added for time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Completed merging for chunk size: {chunk_size} in {processing_time:.2f} seconds")


    print("Completed processing all chunked stream files.")

# # Example usage
# stream_dir = "/path/to/chunked_stream_files"
# cellfile_path = "/path/to/cellfile.cell"
# pointgroup = "mmm"
# num_threads = 23

# chunk_merge_and_write_mtz(stream_dir, cellfile_path, pointgroup, num_threads)
