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
    
def merge_and_write_mtz(stream_dir, cellfile_path, pointgroup, num_threads):
    stream_files = glob.glob(f'{stream_dir}/*.stream')
    total_files = len(stream_files)

    if total_files == 0:
        print(f"No stream files found in directory: {stream_dir}")
        return

    for index, stream_file in enumerate(stream_files):
        ring_sizes_match = re.search(r'_rings_([0-9-]+)\.stream', stream_file)
        if not ring_sizes_match:
            continue

        ring_sizes = ring_sizes_match.group(1)
        output_dir = os.path.join(stream_dir, f'merge-{ring_sizes}')

        # Create output directory and handle existing directories
        os.makedirs(os.path.join(output_dir, "pr-logs"), exist_ok=True)

        # Execute the merging and conversion commands
        try:
            run_partialator(stream_file, output_dir, num_threads, pointgroup)
            convert_hkl_to_mtz(output_dir, cellfile_path)
            print(f"Merging Complete")
        except subprocess.CalledProcessError:
            print(f"Skipping further steps for {stream_file} due to error.")
            continue