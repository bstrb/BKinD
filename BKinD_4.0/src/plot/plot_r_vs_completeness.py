# plot_R1_Rint_vs_ASU.py

# Standard library imports
import os

# Third-party imports
import pandas as pd
import plotly.graph_objects as go

# Plot Imports
from plot.open_plot import open_plot

#####################################################################
# - Parse and plot R1, Rint, Remaining Data, and Average Multiplicity vs ASU
#####################################################################

def parse_r1_rint_stats_file(output_folder):
    """
    Parses the filtering_stats.txt to create a DataFrame with the necessary columns for plotting.
    """
    file_path = os.path.join(output_folder, "filtering_stats.txt")
    data = {
        'Target': [],
        'R1': [],
        'Rint': [],
        'Remaining_Percentage': [],
        'Average_Multiplicity': []
    }

    if not os.path.exists(file_path):
        print(f"No file found at {file_path}")
        return pd.DataFrame(data)

    with open(file_path, 'r') as file:
        sections = file.read().split('-------------------------')
        for section in sections:
            lines = section.strip().split('\n')
            target_completeness = None
            remaining_percent = None
            avg_multiplicity = None
            for line in lines:
                if 'Target Completeness:' in line:
                    target_completeness = float(line.split(':')[1].strip().strip('%'))
                elif 'Resulting Data Percentage:' in line:
                    remaining_percent = float(line.split(':')[1].strip().strip('%'))
                elif 'Average Multiplicity:' in line:
                    avg_multiplicity = float(line.split(':')[1].strip())
                elif 'R1:' in line:
                    parts = line.split(',')
                    r1 = float(parts[0].split(':')[1].strip())
                    rint = float(parts[1].split(':')[1].strip())
                    if target_completeness and remaining_percent is not None:
                        data['Target'].append(target_completeness)
                        data['R1'].append(r1)
                        data['Rint'].append(rint)
                        data['Remaining_Percentage'].append(remaining_percent)
                        data['Average_Multiplicity'].append(avg_multiplicity)

    return pd.DataFrame(data)

#####################################################################

def plot_R1_Rint_vs_completeness(output_folder):
    """
    Reads data from the main directory's filtering_stats.txt, constructs a DataFrame, and creates a plot as R1 and Rint vs. Target ASU.
    """
    df = parse_r1_rint_stats_file(output_folder)
    if df.empty:
        print("No data available for plotting.")
        return

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add Remaining Percentage as a transparent bar plot
    fig.add_trace(go.Bar(x=df['Target'], y=df['Remaining_Percentage'], name='Remaining Data',
                        yaxis='y2', marker_color='firebrick', opacity=0.5, text=df['Average_Multiplicity'],
                        textposition='outside', texttemplate='Avg Mult: %{text:.2f}'))  # Display Average Multiplicity

    # Add R1 scatter plot
    fig.add_trace(go.Scatter(x=df['Target'], y=df['R1'], mode='markers', name='R1 Value',
                            marker=dict(color='blue', size=10),
                            text=df['R1'],  # Show R1 values on hover
                            hoverinfo='text+x'))

    # Add Rint scatter plot
    fig.add_trace(go.Scatter(x=df['Target'], y=df['Rint'], mode='markers', name='Rint Value',
                            marker=dict(color='green', size=10),
                            text=df['Rint'],  # Show Rint values on hover
                            hoverinfo='text+x'))

    # Set x-axis title
    fig.update_xaxes(title_text='Target Completeness (%)')#, autorange="reversed")

    # Set y-axes titles and apply font sizes
    fig.update_layout(
        title=dict(text='R1, Rint, Remaining Data % and Average Multiplicity vs Target Completeness', font=dict(size=32)),
        xaxis=dict(title=dict(font=dict(size=32)), tickfont=dict(size=24)),
        yaxis=dict(title=dict(text='R1 and Rint Values', font=dict(size=32)), tickfont=dict(size=24)),
        yaxis2=dict(title=dict(text='Remaining (%)', font=dict(size=32)), tickfont=dict(size=24), overlaying='y', side='right'),
        legend=dict(title=dict(text='   Metrics', font=dict(size=32)), font=dict(size=24)),
        template="plotly_white"
    )

    # Save the plot as an HTML file
    plot_filename = os.path.join(output_folder, "R1_Rint_vs_Completeness.html")
    fig.write_html(plot_filename)

    open_plot(fig, plot_filename)
