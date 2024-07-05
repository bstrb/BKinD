# modify_xds_inp.py

# Standard library imports
import os

def modify_xds_inp(target_directory):
    """
    Modifies the XDS.INP file to ensure that the only active JOB line is 'JOB= CORRECT'.
    All other JOB lines are commented out.
    """
    
    file_path = os.path.join(target_directory, 'xds.inp')
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        stripped_line = line.strip()
        if 'JOB=' in stripped_line:
            job_command = ' '.join(stripped_line.replace('!', '').strip().split()).upper()
            if job_command == 'JOB= CORRECT':
                new_line = 'JOB= CORRECT\n'  # Ensure this is the only uncommented JOB command
            else:
                new_line = '!' + stripped_line.lstrip('!') + '\n'  # Comment out all other JOB commands
            new_lines.append(new_line)
        else:
            new_lines.append(line)  # Return the original line if it doesn't start with 'JOB='

    # Write the modified lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)