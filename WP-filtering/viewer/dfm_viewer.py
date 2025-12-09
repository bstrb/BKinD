#!/usr/bin/env python3
import os
import base64
import pandas as pd
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.express as px

# --------------------------------------------------------------
# USER SETTINGS
# --------------------------------------------------------------

CSV_PATH = "sample_df_no_filter.csv"     # your merged reflection table
IMAGE_DIR = "images"                 # folder containing 00001.png, 00002.png, ...
PORT = 8051                               # you can change if needed

# --------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# Add a frame column for convenience
df["frame"] = df["zobs"].round().astype(int)

fig = px.scatter(
    df,
    x="zobs",
    y="DFM",
    title="DFM vs Frame (click a point to load PNG)",
    labels={"zobs": "Frame (zobs)", "DFM": "DFM"},
    height=700
)

# --------------------------------------------------------------
# DASH APP
# --------------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "row", "gap": "20px"},
    children=[
        # Left: plot
        html.Div(
            style={"flex": "2"},
            children=[
                dcc.Graph(
                    id="dfm-plot",
                    figure=fig,
                    style={"height": "90vh"},
                )
            ],
        ),

        # Right: image panel
        html.Div(
            style={
                "flex": "1",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center"
            },
            children=[
                html.H3("Selected Frame"),
                html.Div(id="frame-info", style={"marginBottom": "10px"}),
                html.Img(
                    id="frame-image",
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "80vh",
                        "border": "1px solid #ccc",
                        "background": "#fafafa"
                    }
                ),
            ],
        ),
    ],
)

# --------------------------------------------------------------
# CALLBACK: CLICK → LOAD PNG
# --------------------------------------------------------------

def frame_to_png_path(frame: int):
    filename = f"{frame:05d}.png"
    path = os.path.join(IMAGE_DIR, filename)
    return path if os.path.exists(path) else None


@app.callback(
    Output("frame-image", "src"),
    Output("frame-info", "children"),
    Input("dfm-plot", "clickData"),
)
def update_image(clickData):
    if clickData is None:
        return no_update, "Click a point to display its PNG."

    try:
        zobs = float(clickData["points"][0]["x"])
    except Exception:
        return no_update, "Could not read zobs from click."

    frame = round(zobs)
    path = frame_to_png_path(frame)

    if path is None:
        return no_update, f"No PNG found for frame {frame}."

    # Load image and encode
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return no_update, f"Could not open PNG for frame {frame}."

    src = f"data:image/png;base64,{encoded}"
    info = f"zobs ≈ {zobs:.2f} → frame {frame:05d}.png"

    return src, info


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    print(f"Viewer ready. Open http://127.0.0.1:{PORT} in a browser.")
    app.run(debug=True, port=PORT)
