import os
 
def find_first_file(directory, extension):
    # Find the first file in the directory with the specified extension
    for file_name in os.listdir(directory):
        if file_name.endswith(extension):
            return file_name  # Return only the file name, not the full path
    return None

def generate_bash_script(bash_file_name, stream_files_dir,
                        num_threads=None, sol_file=None, 
                        lst_file=None, geom_file=None, 
                        cell_file=None
                        ):
    # Generate the full path for the bash file
    bash_file_path = f"{stream_files_dir}/{bash_file_name}.sh"

    # If no geom_file, lst_file, or cell_file is provided, find the first one in the directory
    num_threads = num_threads or 23
    input_sol_file = sol_file
    lst_file_path = lst_file or find_first_file(stream_files_dir, ".lst")
    geom_file_path = geom_file or find_first_file(stream_files_dir, ".geom")
    cell_file_path = cell_file or find_first_file(stream_files_dir, ".cell")

    # Error if any of the required files are not found
    if not lst_file_path:
        raise FileNotFoundError("No .lst file found in the directory.")
    if not geom_file_path:
        raise FileNotFoundError("No .geom file found in the directory.")
    if not cell_file_path:
        raise FileNotFoundError("No .cell file found in the directory.")
    if not input_sol_file:
        raise FileNotFoundError("No .sol file found in the directory.")

    # Create the content of the bash file
    bash_script_content = f"""indexamajig -i {lst_file_path} -g {geom_file_path} -p {cell_file_path} -j {num_threads} -o int_output.stream --indexing=file --fromfile-input-file={input_sol_file} --no-revalidate --no-retry --integration=rings --no-refine --no-half-pixel-shift --no-check-peaks --no-non-hits-in-stream --no-check-cell --peaks=cxi --min-peaks=15
"""
    
    # Write the content to the bash file
    with open(bash_file_path, 'w') as bash_file:
        bash_file.write(bash_script_content)

    # Make the bash file executable
    os.chmod(bash_file_path, 0o755)

    # Output the full path of the generated bash file
    print(f"Bash script generated: {bash_file_path}")


# Example usage
if __name__ == "__main__":
    # Directory where stream files and other required files (.sol, .lst, .geom, .cell) are stored
    stream_files_dir = ''  # Update this with your actual directory

    # Name for the generated bash script (without the extension)
    bash_file_name = 'bash'

    # Optional parameters (you can specify them or leave them to be auto-found by the script)
    num_threads = 23  # Optional, default is 23 if not specified
    sol_file = None  # Optional, will find the first .sol file in the directory if not specified
    lst_file = None  # Optional, will find the first .lst file in the directory if not specified
    geom_file = None  # Optional, will find the first .geom file in the directory if not specified
    cell_file = None  # Optional, will find the first .cell file in the directory if not specified

    # Generate the bash script
    generate_bash_script(bash_file_name, stream_files_dir, num_threads, sol_file, lst_file, geom_file, cell_file)