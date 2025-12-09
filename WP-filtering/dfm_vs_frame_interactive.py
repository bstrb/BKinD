#!/usr/bin/env python3
import os
import base64

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update

# ------------------------------------------------------------------
# 1) Load the sample_df you already generated with BKinD script
# ------------------------------------------------------------------

CSV_PATH = "/Users/xiaodong/Desktop/DFM_vs_frame_nem_output/sample_df_no_filter.csv"
FRAME_DIR = "/Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/images/png"
FRAME_PATTERN = "{:05d}.png"


sample_df = pd.read_csv(CSV_PATH)

# Optional: keep a nice 'frame' column for inspection
sample_df["frame"] = sample_df["zobs"].round().astype(int)

# ------------------------------------------------------------------
# 2) Create the Plotly figure (DFM vs frame)
# ------------------------------------------------------------------

fig = px.scatter(
    sample_df,
    x="zobs",
    y="DFM",
    title="DFM vs Frame (click a point to see frame PNG)",
    labels={"zobs": "Frame (zobs)", "DFM": "DFM"},
)

# ------------------------------------------------------------------
# 3) Dash app layout: plot + image
# ------------------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "row", "gap": "20px"},
    children=[
        html.Div(
            style={"flex": "2"},
            children=[
                dcc.Graph(
                    id="dfm-plot",
                    figure=fig,
                    style={"height": "80vh"},
                )
            ],
        ),
        html.Div(
            style={"flex": "1", "display": "flex", "flexDirection": "column", "alignItems": "center"},
            children=[
                html.H3("Selected frame"),
                html.Div(id="frame-info"),
                html.Img(
                    id="frame-image",
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "70vh",
                        "border": "1px solid #ccc",
                    },
                ),
            ],
        ),
    ],
)

# ------------------------------------------------------------------
# 4) Callback: when you click a point, update the image
# ------------------------------------------------------------------

def frame_path_from_zobs(zobs: float) -> str | None:
    """
    Map zobs -> nearest integer frame -> PNG path.
    Adjust this logic to match your actual naming/offset.
    """
    if zobs is None:
        return None
    frame_index = int(round(zobs))

    # Example: if your files are like frame_0001.png, frame_0002.png, ...
    fname = FRAME_PATTERN.format(frame_index)
    path = os.path.join(FRAME_DIR, fname)
    if not os.path.exists(path):
        return None
    return path


@app.callback(
    Output("frame-image", "src"),
    Output("frame-info", "children"),
    Input("dfm-plot", "clickData"),
)
def update_image(clickData):
    if clickData is None:
        return no_update, "Click a point in the plot to view the corresponding frame."

    # We set x = zobs in the figure, so this is the zobs for the clicked reflection
    try:
        zobs = float(clickData["points"][0]["x"])
    except Exception:
        return no_update, "Could not read zobs from click."

    path = frame_path_from_zobs(zobs)
    if path is None:
        return no_update, f"No PNG found for zobs ≈ {zobs:.2f}"

    # Read file and base64-encode it for embedding in HTML
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    src = f"data:image/png;base64,{encoded}"

    info = f"zobs = {zobs:.2f} → frame file: {os.path.basename(path)}"
    return src, info


if __name__ == "__main__":
    app.run(debug=True)

