# full_merge_def.py

import re
import os
import glob
import subprocess

def merge_and_write_mtz(stream_dir, cellfile_path, pointgroup):
    run = 1
    stream_files = glob.glob(f'{stream_dir}*.stream')  

    for stream_file in stream_files:
        ring_sizes_match = re.search(r'_rings_([0-9-]+)\.stream', stream_file)
        if not ring_sizes_match:
            continue

        ring_sizes = ring_sizes_match.group(1)
        output_dir_base = os.path.join(stream_dir, f'merge-{ring_sizes}-run')
        output_dir = f"{output_dir_base}{run}"

        while True:
            try:
                os.makedirs(output_dir)
                break  
            except FileExistsError:
                run += 1  
                output_dir = f"{output_dir_base}{run}"

        os.makedirs(os.path.join(output_dir, "pr-logs"), exist_ok=True)

        merging_cmd = [
            'partialator', 
            stream_file,
            '--model=offset',
            '-j', '23',
            '-o', os.path.join(output_dir, "crystfel.hkl"),
            '-y', pointgroup,
            '--polarisation=none',
            '--min-measurements=2',
            '--max-adu=inf',
            '--min-res=inf',
            '--push-res=inf',
            '--no-Bscale',
            # '--no-pr',
            '--no-logs',
            '--iterations=3',
            '--harvest-file=' + os.path.join(output_dir, "parameters.json"),
            '--log-folder=' + os.path.join(output_dir, "pr-logs")
        ]

        hkl2mtz_cmd = [
            'get_hkl',
            '-i', f'{output_dir}/crystfel.hkl',
            '-o', f'{output_dir}/output.mtz',
            '-p', f'{cellfile_path}',
            '--output-format=mtz'
        ]


        with open(os.path.join(output_dir, "stdout.log"), "w") as stdout, open(os.path.join(output_dir, "stderr.log"), "w") as stderr:
            print(f"Running merging for ring sizes: {ring_sizes}")
            subprocess.run(merging_cmd, stdout=stdout, stderr=stderr)
            subprocess.run(hkl2mtz_cmd, stdout=stdout, stderr=stderr)