# rcis_int_def_v3.py

import subprocess
import os
import time
import threading
import queue
from tqdm import tqdm
# from count_sol_lines import count_sol_lines

def run_script(bash_file_path, num_indexed_frames, updated_output_file):
    try:
        # Determine the directory of the bash script
        bash_dir = os.path.dirname(bash_file_path)

        # Copy the current environment variables
        env = os.environ.copy()

        # Start the script with the working directory set to the bash script's directory
        process = subprocess.Popen(
            ['bash', bash_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=bash_dir,
            env=env  # Pass the environment variables
        )

        # Initialize the progress bar
        progress_bar = tqdm(total=num_indexed_frames, desc="Writing integration output stream", unit="lines")
        
        # Queue to hold process output lines
        output_queue = queue.Queue()

        def enqueue_output(out, queue):
            for line in iter(out.readline, ''):
                queue.put(line)
            out.close()

        # Start threads to read stdout and stderr without blocking
        threading.Thread(target=enqueue_output, args=(process.stdout, output_queue), daemon=True).start()
        threading.Thread(target=enqueue_output, args=(process.stderr, output_queue), daemon=True).start()

        last_size = 0
        while process.poll() is None:
            time.sleep(0.5)  # Reduced sleep time for more frequent checks

            # Check the size of the updated output file
            if os.path.exists(updated_output_file):
                current_size = os.stat(updated_output_file).st_size
                if current_size > last_size:
                    with open(updated_output_file, 'r') as output_file:
                        output_file.seek(last_size)  # Move to the last read position
                        new_lines = sum(1 for line in output_file if 'num_reflections' in line)
                        last_size = output_file.tell()  # Update the last position to the end of the file

                        # Update the progress bar based on new lines found
                        if new_lines > 0:
                            progress_bar.update(new_lines)

                progress_bar.refresh()

        # Capture the remaining output
        output, error = process.communicate(timeout=10)
        progress_bar.close()

        if process.returncode == 0:
            print("Integration executed successfully")
        else:
            print("Script failed with return code", process.returncode)
            if error:
                print("Error:", error)
    except subprocess.TimeoutExpired:
        print("Process timed out while waiting for completion.")
        process.kill()
    except Exception as e:
        print("An exception occurred:", e)


def update_bash_file(file_path, integration="rings", int_radius=(3, 4, 7)):
    if integration == "rings":
        inner, middle, outer = int_radius
        integrationext = f"{integration}_{inner}-{middle}-{outer}"
        int_radius_param = ",".join(map(str, int_radius))  # Convert tuple to comma-separated string
    else:
        integrationext = integration

    new_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if '-o' in line:  # This check is still valid since -o is a separate argument
                parts = line.split()
                # Update the output file name
                output_file_index = parts.index('-o') + 1
                output_file = parts[output_file_index]
                base_name, ext = output_file.split('.')
                updated_output_file = f"{base_name}_{integrationext}.{ext}"
                parts[output_file_index] = updated_output_file

                # Flags to track if `--integration` and `--int-radius` are present
                int_radius_present = False

                for i, part in enumerate(parts):
                    if '--integration' in part:
                        parts[i] = f'--integration={integration}'
                    elif '--int-radius' in part and integration == "rings":
                        parts[i] = f'--int-radius={int_radius_param}'
                        int_radius_present = True

                # If `--int-radius` is not present and integration is "rings", append it
                if integration == "rings" and not int_radius_present:
                    parts.append(f'--int-radius={int_radius_param}')

                line = ' '.join(parts)

            new_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(new_lines)

    return updated_output_file


def reset_bash_path(file_path, output_stream_format):
    new_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if '-o' in line:
                parts = line.split()
                parts[parts.index('-o') + 1] = output_stream_format
                new_lines.append(' '.join(parts) + '\n')
            else:
                new_lines.append(line)
    
    with open(file_path, 'w') as file:
        file.writelines(new_lines)


def fast_integration(bash_file_path, output_stream_format, num_indexed_frames, integration="rings", ring_sizes=None):
    # Verify that the provided bash file exists
    if not os.path.exists(bash_file_path):
        raise FileNotFoundError(f"The bash file at {bash_file_path} does not exist.")

    # Create the output directory if it does not exist
    output_dir = os.path.dirname(output_stream_format)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if integration == "rings" and ring_sizes:
        for int_radius in ring_sizes:
            # Reset and update the bash file for each set of ring sizes
            reset_bash_path(bash_file_path, output_stream_format)
            updated_output_file = update_bash_file(bash_file_path, integration, int_radius)
            run_script(bash_file_path, num_indexed_frames, updated_output_file)
            reset_bash_path(bash_file_path, output_stream_format)
    else:
        # Handle other integration types
        reset_bash_path(bash_file_path, output_stream_format)
        updated_output_file = update_bash_file(bash_file_path, integration)
        run_script(bash_file_path, num_indexed_frames, updated_output_file)
        reset_bash_path(bash_file_path, output_stream_format)

# Example usage
def example_usage():
    bash_file_path = "/home/bubl3932/files/UOX1/UOX1_original_IQM/IQM_SUM_22_12_10_-12_12_-15_10_13_-13.sh"
    output_stream_format = "/home/bubl3932/files/UOX1/UOX1_original_IQM/IQM_SUM_22_12_10_-12_12_-15_10_13_-13/int_rings_2-5-10.stream"
    num_indexed_frames = 5227
    integration = "rings"
    ring_sizes = [(2, 5, 10)]

    fast_integration(bash_file_path, output_stream_format, num_indexed_frames, integration, ring_sizes)

if __name__ == "__main__":
    example_usage()