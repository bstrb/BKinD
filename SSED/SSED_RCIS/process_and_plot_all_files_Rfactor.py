import os
import re
import numpy as np
import plotly.graph_objects as go

from open_plot import open_plot

def find_last_cycle_line(content):
    """
    Finds the starting position of the last ':Cycle' in the content.
    """
    cycle_pattern = re.compile(r'^:Cycle\s+\d+', re.MULTILINE)
    matches = list(cycle_pattern.finditer(content))
    
    if matches:
        last_cycle_match = matches[-1]  # Get the last ':Cycle' occurrence
        last_cycle_line_position = last_cycle_match.start()
        print(f"Found last cycle at character position: {last_cycle_line_position}.")
        return last_cycle_line_position
    else:
        print("No cycle sections found.")
        return None

def extract_data_from_section(content, last_cycle_line):
    """
    Extracts the numerical data from the section starting at the last cycle line.
    Looks for "Rf_used" line, then the first $$ line, and extracts the table until the next $$ line.
    """
    lines = content[last_cycle_line:].splitlines()
    numerical_section = []
    rf_used_found = False
    in_data_block = False

    # Pattern to match lines containing only '$$'
    delimiter_pattern = re.compile(r'^\$\$')

    for line in lines:
        # Look for the "Rf_used" line
        if not rf_used_found and 'Rf_used' in line:
            rf_used_found = True
            continue  # Skip to the next line after finding "Rf_used"

        # Look for the first '$$' line after finding "Rf_used"
        if rf_used_found and not in_data_block and delimiter_pattern.match(line):
            in_data_block = True  # Enter the data block
            continue

        # Extract numerical data while inside the data block
        if in_data_block:
            if delimiter_pattern.match(line):
                break  # Stop at the second '$$' line
            numerical_section.append(line.strip())

    if numerical_section:
        return numerical_section
    else:
        print("No numerical data found after the 'Rf_used' section.")
        return None

def format_extracted_data(numerical_section):
    """
    Converts the extracted numerical section into a structured NumPy array.
    """
    data = []
    for line in numerical_section:
        # Split the line by spaces and convert to float
        values = list(map(float, re.split(r'\s+', line.strip())))
        data.append(values)
    
    return np.array(data)

def plot_data(fig, data, label=None):
    """
    Plots the 6th column (Rf_used) vs. 1st column (resolution) using Plotly.
    Converts resolution using sqrt(1/resolution) before plotting.
    """
    resolution = data[:, 0]
    rf_used = data[:, 5]

    # Convert resolution: sqrt(1/resolution)
    converted_resolution = np.sqrt(1 / resolution)

    # Add a trace to the global figure
    fig.add_trace(go.Scatter(
        x=converted_resolution,
        y=rf_used,
        mode='lines+markers',
        name=label
    ))

def process_and_plot_all_files(base_path):
    fig = go.Figure()
    folder_labels = []

    # Iterate through all folders in the base directory
    for head_folder in os.listdir(base_path):
        head_folder_path = os.path.join(base_path, head_folder)
        
        if os.path.isdir(head_folder_path):
            # Iterate through subfolders starting with "merge"
            for sub_folder in os.listdir(head_folder_path):
                if sub_folder.startswith("merge"):
                    merge_folder_path = os.path.join(head_folder_path, sub_folder)
                    
                    if os.path.isdir(merge_folder_path):
                        # Find the .txt file in the "merge" subfolder
                        for file in os.listdir(merge_folder_path):
                            if file.endswith(".txt"):
                                file_path = os.path.join(merge_folder_path, file)
                                print(f"Processing file: {file_path}")

                                with open(file_path, 'r') as f:
                                    content = f.read()

                                last_cycle_line = find_last_cycle_line(content)

                                if last_cycle_line:
                                    numerical_section = extract_data_from_section(content, last_cycle_line)

                                    if numerical_section:
                                        data = format_extracted_data(numerical_section)

                                        if data is not None:
                                            # Use the head folder name as label
                                            plot_data(fig, data, label=head_folder)
                                            folder_labels.append(head_folder)
                                        else:
                                            print(f"No data to plot for {file_path}.")
                                    else:
                                        print(f"No numerical section found in {file_path}.")
                                else:
                                    print(f"No last cycle found in {file_path}.")

    # Sort labels by name
    sorted_labels = sorted(folder_labels)
    fig.data = sorted(fig.data, key=lambda trace: sorted_labels.index(trace.name))

    fig.update_layout(
        title=f'Rf_used vs Resolution for data in {base_path}',
        xaxis_title='Resolution (Å)',
        yaxis_title='Rf_used',
        template='plotly_dark',
        showlegend=True,
        xaxis=dict(autorange='reversed')  # This line reverses the x-axis
    )
    
    plotname = "Rf_used_vs_Resolution.html"
    plot_filename = os.path.join(base_path, plotname)
    fig.write_html(plot_filename)

    open_plot(fig, plot_filename)

directory = "/home/buster/UOXm/5x5_0-01"
process_and_plot_all_files(directory)