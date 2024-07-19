# run_process.py

# Standard library imports
import os
import subprocess

def run_process(command, directory, input_file=None, suppress_output=False, additional_command=None):
    """
    Runs a specified command in the given directory, optionally on a specified input file,
    and optionally with an additional command.

    Parameters:
    - command: The command to run as a list of arguments (e.g., ["shelxl"]).
    - directory: The directory in which to run the command.
    - input_file: Optional. The input file to find in the directory and include in the command.
    - suppress_output: Optional. If True, suppresses the console output from the subprocess.
    - additional_command: Optional. Additional command or options to append to the main command.

    Returns:
    - None, but prints the output or error messages as applicable.
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        print("The specified directory does not exist.")
        return

    # Change the current working directory to the input directory
    os.chdir(directory)

    # If an input file is specified, find it in the directory
    if input_file:
        file_path = None
        files = os.listdir(directory)
        for file in files:
            if file.endswith(input_file):
                file_path = os.path.join(directory, file)
                command.append(file.replace(input_file, ''))
                break

        if not file_path:
            print(f"No file with extension '{input_file}' found in the directory.")
            return

    # If an additional command is specified, append it to the command list
    if additional_command:
        command.extend(additional_command.split())

    # Run the specified command
    try:
        if suppress_output:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        else:
            result = subprocess.run(command, check=True, text=True)

        if result.returncode == 0 and not suppress_output:
            print(f"{command[0]} ran successfully.")
            print(result.stdout)
        elif result.returncode != 0:
            print(f"{command[0]} encountered an error:")
            print(result.stderr)
    
    except Exception as e:
        print(f"Failed to run {command[0]}: {e}")

# Example usage
# run_process(["shelxl"], "/path/to/directory", input_file=".ins", suppress_output=False, additional_command='-s"Pm-3m"')
