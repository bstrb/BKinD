import os
import glob
import subprocess
import time
from tqdm import tqdm

from find_first_file import find_first_file
from convert_hkl_to_mtz import convert_hkl_to_mtz

def run_partialator(stream_file, output_dir, num_threads, pointgroup, iterations):
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
        f'--iterations={iterations}',
        '--harvest-file=' + os.path.join(output_dir, "parameters.json"),
        '--log-folder=' + os.path.join(output_dir, "pr-logs")
    ]

    stderr_path = os.path.join(output_dir, "stderr.log")
    total_residuals = iterations + 2

    try:
        with open(os.path.join(output_dir, "stdout.log"), "w") as stdout, open(stderr_path, "w") as stderr:
            print(f"Running partialator for stream file: {stream_file}")
            progress = tqdm(total=total_residuals, desc="Partialator Progress", unit="Residuals")
            process = subprocess.Popen(merging_cmd, stdout=stdout, stderr=stderr)
            
            # Track progress based on "Residuals:" in stderr.log
            while process.poll() is None:
                time.sleep(1)  # Wait for a few seconds before checking
                if os.path.exists(stderr_path):
                    with open(stderr_path, "r") as f:
                        residual_count = sum(1 for line in f if line.startswith("Residuals:"))
                        progress.n = min(residual_count, total_residuals)
                        progress.refresh()

            process.communicate()  # Ensure the process has completed
            progress.n = total_residuals
            progress.refresh()
            progress.close()
            print(f"Partialator completed for stream file: {stream_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error during partialator execution for {stream_file}: {e}")
        raise
    finally:
        progress.close()
    
def merge_and_write_mtz(stream_dir, cellfile_path, pointgroup, num_threads, iterations):
    stream_files = glob.glob(f'{stream_dir}/*.stream')
    total_files = len(stream_files)

    if total_files == 0:
        print(f"No stream files found in directory: {stream_dir}")
        return

    for stream_file in stream_files:
        stream_file_name = os.path.basename(stream_file).replace('.stream', '')
        output_dir = os.path.join(stream_dir, f'merge_{iterations}_iter_{stream_file_name}')

        # Create output directory and handle existing directories
        if not os.path.exists(output_dir):
            os.makedirs(os.path.join(output_dir, "pr-logs"), exist_ok=True)
        else:
            print(f"Output directory {output_dir} already exists. Skipping re-processing.")
            continue

        # Execute the merging and conversion commands
        try:
            run_partialator(stream_file, output_dir, num_threads, pointgroup, iterations)
            convert_hkl_to_mtz(output_dir, cellfile_path)
            print(f"Merging Complete for {stream_file_name}")
        except subprocess.CalledProcessError:
            print(f"Skipping further steps for {stream_file_name} due to error.")
            continue

# Example usage
if __name__ == "__main__":
    base_path = "/home/buster/UOXm/5x5_0-01/fast_int_RCIS_3_3_1"  # Replace with the actual stream files directory
    stream_file_folder = "/home/buster/UOXm/5x5_0-01"
    cellfile_path = find_first_file(stream_file_folder, ".cell")
    pointgroup = "mmm"
    num_threads = 23
    iterations = 3

    merge_and_write_mtz(base_path, cellfile_path, pointgroup, num_threads, iterations)