# plot_chunk_merge_def.py

import re
import os
import numpy as np
import plotly.graph_objects as go

from open_plot import open_plot  # Ensure this function is correctly defined in the open_plot module

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
    Looks for "Rf_free" line, then the first $$ line, and extracts the table until the next $$ line.
    """
    lines = content[last_cycle_line:].splitlines()
    numerical_section = []
    rf_free_found = False
    in_data_block = False

    # Pattern to match lines containing only '$$'
    delimiter_pattern = re.compile(r'^\$\$$')

    for line in lines:
        # Look for the "Rf_free" line
        if not rf_free_found and 'Rf_free' in line:
            rf_free_found = True
            continue  # Skip to the next line after finding "Rf_free"

        # Look for the first '$$' line after finding "Rf_free"
        if rf_free_found and not in_data_block and delimiter_pattern.match(line):
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
        print("No numerical data found after the 'Rf_free' section.")
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

def process_and_plot_all_files(base_path):

    fig = go.Figure()
    traces = []  # List to collect traces

    for root, dirs, files in os.walk(base_path):
        txt_files = [f for f in files if f.endswith('.txt')]
        
        if len(txt_files) == 1:
            file_path = os.path.join(root, txt_files[0])
            print(f"Processing file: {file_path}")

            # Extract just the folder name
            folder_name = os.path.basename(os.path.dirname(file_path))

            # Extract the numeric value from the folder name
            match = re.search(r'\d+', folder_name)
            if match:
                label = match.group(0)  # Extract the number as the label
            else:
                label = folder_name  # Fallback to the original folder name if no number is found

            with open(file_path, 'r') as file:
                content = file.read()

            last_cycle_line = find_last_cycle_line(content)

            if last_cycle_line:
                numerical_section = extract_data_from_section(content, last_cycle_line)

                if numerical_section:
                    data = format_extracted_data(numerical_section)

                    if data is not None:
                        # Instead of plotting directly, store the trace in the list
                        resolution = data[:, 0]
                        rf_free = data[:, 10]
                        converted_resolution = np.sqrt(1 / resolution)

                        trace = go.Scatter(
                            x=converted_resolution,
                            y=rf_free,
                            mode='lines+markers',
                            name=label
                        )
                        traces.append((int(label), trace))  # Store the label as integer for sorting
                    else:
                        print(f"No data to plot for {file_path}.")
                else:
                    print(f"No numerical section found in {file_path}.")
            else:
                print(f"No last cycle found in {file_path}.")

    # Sort traces by label (first element of the tuple)
    traces.sort(key=lambda x: x[0])

    # Add sorted traces to the figure
    for _, trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        title=f'Rf_free vs Resolution for data in {base_path}',
        xaxis_title='Resolution (Å)',
        yaxis_title='Rf_free',
        template='plotly_dark',
        showlegend=True,
        legend_title_text='Indexed Frames',  # Add the legend title
        xaxis=dict(autorange='reversed')  # This line reverses the x-axis
    )
    
    plotname = "Rfree_vs_Resolution.html"
    plot_filename = os.path.join(base_path, plotname)
    fig.write_html(plot_filename)

    open_plot(fig, plot_filename)
