# plot_intensities.py

import plotly.graph_objects as go

def plot_intensities(self, inside_intensity_values, outside_intensity_values, total_intensity_values, absolute_difference_values, sigma_level):
    """Plot the selected intensity values."""
    fig = go.Figure()

    # Plot based on selected checkboxes
    if self.plot_inside_var.get():
        fig.add_trace(go.Scatter(x=list(range(1, len(inside_intensity_values) + 1)), y=inside_intensity_values,
                                mode='lines', name='Inside Gaussian Region', line=dict(color='blue')))
    if self.plot_outside_var.get():
        fig.add_trace(go.Scatter(x=list(range(1, len(outside_intensity_values) + 1)), y=outside_intensity_values,
                                mode='lines', name='Outside Gaussian Region', line=dict(color='red')))
    if self.plot_total_var.get():
        fig.add_trace(go.Scatter(x=list(range(1, len(total_intensity_values) + 1)), y=total_intensity_values,
                                mode='lines', name='Total Intensity', line=dict(color='green')))
    if self.plot_difference_var.get():
        fig.add_trace(go.Scatter(x=list(range(1, len(absolute_difference_values) + 1)), y=absolute_difference_values,
                                mode='lines', name='|Inside - Outside|', line=dict(color='magenta')))

    fig.update_layout(
        title=f'Normalized Sum of Intensities vs. Frame (CB Gaussian Sigma = {sigma_level})',
        xaxis_title='Frame Number',
        yaxis_title='Normalized Sum of Intensity',
        legend_title='Intensity Types',
        hovermode='x unified'
    )

    fig.show()