# fast_int_def.py

import subprocess
import os

def run_script(bash_file_path):
    try:
        # Debug: Print the bash script content before running
        with open(bash_file_path, 'r') as bash_file:
            print("Bash script content:")
            print(bash_file.read())

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

        # Capture the output
        output, error = process.communicate()

        if process.returncode == 0:
            print("Script executed successfully")
            print("Output:", output)
        else:
            print("Script failed with return code", process.returncode)
            print("Error:", error)
    except Exception as e:
        print("An exception occurred:", e)


def update_bash_file(file_path, integration="rings", int_radius=(4, 5, 7)):
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
                new_output_file = f"{base_name}_{integrationext}.{ext}"
                parts[output_file_index] = new_output_file

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

def fast_integration(bash_file_path, output_stream_format, integration="rings", ring_sizes=None):
    
    # Verify that the provided bash file exists
    if not os.path.exists(bash_file_path):
        raise FileNotFoundError(f"The bash file at {bash_file_path} does not exist.")

    if integration == "rings" and ring_sizes:
        for int_radius in ring_sizes:
            # Reset and update the bash file for each set of ring sizes
            reset_bash_path(bash_file_path, output_stream_format)
            update_bash_file(bash_file_path, integration, int_radius)
            run_script(bash_file_path)
            reset_bash_path(bash_file_path, output_stream_format)
    else:
        # Handle other integration types
        reset_bash_path(bash_file_path, output_stream_format)
        update_bash_file(bash_file_path, integration)
        run_script(bash_file_path)
        reset_bash_path(bash_file_path, output_stream_format)