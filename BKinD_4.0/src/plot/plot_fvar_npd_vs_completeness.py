# plot_FVAR_NPD_vs_ASU.py

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
        'Target Completeness': [],
        'FVAR': [],
        'NPD': []
    }

    if not os.path.exists(file_path):
        print(f"No file found at {file_path}")
        return pd.DataFrame(data)

    with open(file_path, 'r') as file:
        sections = file.read().split('-------------------------')
        for section in sections:
            lines = section.strip().split('\n')
            target_completeness = None
            for line in lines:
                if 'Target Completeness:' in line:
                    target_completeness = float(line.split(':')[1].strip().strip('%'))
                elif 'R1:' in line:
                    parts = line.split(',')
                    FVAR = float(parts[2].split(':')[1].strip())
                    NPD = float(parts[3].split(':')[1].strip())
                    if target_completeness and FVAR and NPD is not None:
                        data['Target Completeness'].append(target_completeness)
                        data['FVAR'].append(FVAR)
                        data['NPD'].append(NPD)

    return pd.DataFrame(data)

#####################################################################

def plot_FVAR_NPD_vs_completeness(output_folder):
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
    fig.add_trace(go.Bar(x=df['Target Completeness'], y=df['NPD'], name='Number of NPDs',
                        yaxis='y2', marker_color='firebrick', opacity=0.5,
                        # text=df['Average_Multiplicity'],
                        # textposition='outside', texttemplate='Avg Mult: %{text:.2f}'
                        ))  # Display Average Multiplicity

    # Add R1 scatter plot
    fig.add_trace(go.Scatter(x=df['Target Completeness'], y=df['FVAR'], mode='markers', name='FVAR Values',
                            marker=dict(color='blue', size=10),
                            text=df['FVAR'],  # Show R1 values on hover
                            hoverinfo='text+x'))


    # Set x-axis title
    fig.update_xaxes(title_text='Completeness')#, autorange="reversed")

    # Set y-axes titles and apply font sizes
    fig.update_layout(
        title=dict(text='FVAR and Number of NPDs vs. Completeness', font=dict(size=32)),
        xaxis=dict(title=dict(font=dict(size=32)), tickfont=dict(size=24)),
        yaxis=dict(title=dict(text='FVAR Values', font=dict(size=32)), tickfont=dict(size=24)),
        yaxis2=dict(title=dict(text='Number of NPDs', font=dict(size=32)), tickfont=dict(size=24), overlaying='y', side='right'),
        legend=dict(title=dict(text='   Metrics', font=dict(size=32)), font=dict(size=24)),
        template="plotly_white"
    )

    # Save the plot as an HTML file
    plot_filename = os.path.join(output_folder, "FVAR_NPD_vs_Completeness.html")
    fig.write_html(plot_filename)

    open_plot(fig, plot_filename)
