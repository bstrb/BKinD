import os
import pandas as pd
import plotly.graph_objects as go

from open_plot import open_plot

# Load the CSV file
csv_path = '/home/buster/UOX1/different_index_params/3x3_retry/rfactor_values.csv'
df = pd.read_csv(csv_path)

# Extract Rfactor and metrics
df['Metrics'] = df['Head Folder'].str.replace("IQM_SUM_", "").str.split("_")
df[['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5', 
    'Metric 6', 'Metric 7', 'Metric 8', 'Metric 9']] = pd.DataFrame(df['Metrics'].tolist(), index=df.index).astype(int)

# Prepare the Rfactor and metrics
rfactor = df['Rfactor']
metrics = df[['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5', 
              'Metric 6', 'Metric 7', 'Metric 8', 'Metric 9']]

# Create the interactive plot
fig = go.Figure()

# Add each metric as a line
for col in metrics.columns:
    fig.add_trace(go.Scatter(
        x=rfactor,
        y=metrics[col],
        mode='lines+markers',
        name=col
    ))

# Update layout
fig.update_layout(
    title="Metrics vs Rfactor",
    xaxis_title="Rfactor",
    yaxis_title="Metric Value",
    legend_title="Metrics",
    hovermode="x unified"
)

# Save the plot as an HTML file and open it
base_path = '/home/buster/UOX1/different_index_params/3x3_retry'
plotname = "metrics_plot.html"
plot_filename = os.path.join(base_path, plotname)
fig.write_html(plot_filename)

# Open the plot using the predefined open_plot function
try:
    open_plot(fig, plot_filename)
except ImportError:
    print(f"Plot saved as: {plot_filename}. Please ensure 'open_plot' is available.")
