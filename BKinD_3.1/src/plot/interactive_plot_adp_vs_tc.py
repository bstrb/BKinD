# interactive_plot_adp_vs_tc.py

import os
import re
import plotly.graph_objs as go
from collections import defaultdict


# Plot Imports
from plot.open_plot import open_plot

def extract_values_from_res(file_path):
    values = {}
    with open(file_path, 'r') as file:
        for line in file:
            if re.match(r'^[A-Za-z]+\d+', line) and not line.startswith("Q1"):
                parts = line.split()
                atom_name = parts[0]
                last_value = float(parts[-1])
                values[atom_name] = last_value
    return values

def plot_results(data, folder_path):
    fig = go.Figure()
    
    for atom_name, values in data.items():
        fig.add_trace(go.Scatter(
            x=values['completeness'],
            y=values['values'],
            mode='markers',
            name=atom_name
        ))

    fig.update_layout(
        title='Interactive Plot of ADP vs Target Completeness - Mark Desired Area',
        xaxis_title='Target Completeness (%)',
        yaxis_title='ADP (ADP)',
        xaxis=dict(autorange='reversed'),  # Reverse the x-axis
        hovermode='closest'
    )

    plotname="ADP_vs_TC.html"

    # Save the plot as an HTML file
    plot_filename = os.path.join(os.path.dirname(folder_path), plotname)
    fig.write_html(plot_filename)

    open_plot(fig, plot_filename)

def interactive_plot_adp_vs_tc(folder_path):
    data = defaultdict(lambda: {'completeness': [], 'values': []})
    
    for subfolder in os.listdir(folder_path):
        match = re.match(r'filtered_(\d+\.\d+)', subfolder)
        if match:
            completeness = float(match.group(1))
            res_file = os.path.join(folder_path, subfolder, f'removed_data_{completeness:.1f}', f'removed_data_{completeness:.1f}.res')
            
            if os.path.exists(res_file):
                values = extract_values_from_res(res_file)
                for atom_name, value in values.items():
                    data[atom_name]['completeness'].append(completeness)
                    data[atom_name]['values'].append(value)
    
    plot_results(data, folder_path)

