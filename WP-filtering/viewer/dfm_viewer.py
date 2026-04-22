#!/usr/bin/env python3
import argparse
import base64
import os
import re
from functools import lru_cache
from typing import Optional

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "sample_df_no_filter.csv")
DEFAULT_IMAGE_DIR = os.path.join(BASE_DIR, "images")
DEFAULT_PORT = 8051
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Interactive reflection viewer: scatter plot on the left, frame image on the right."
    )
    ap.add_argument("--csv", default=DEFAULT_CSV_PATH, help="CSV containing the plot data")
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Directory containing prepared frame images")
    ap.add_argument("--x-col", default="zobs", help="Column to plot on the x axis")
    ap.add_argument("--y-col", default=None, help="Column to plot on the y axis")
    ap.add_argument(
        "--frame-col",
        default=None,
        help="Column to convert to the displayed frame number. Defaults to 'frame' if present else round(x-col).",
    )
    ap.add_argument("--title", default=None, help="Custom plot title")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="Dash server port")
    return ap.parse_args()


def die(msg: str) -> None:
    raise SystemExit(msg)


def choose_y_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested is not None:
        if requested not in df.columns:
            die(f"Missing y column '{requested}'. Available columns: {list(df.columns)}")
        return requested

    for candidate in ("SRES", "DFM"):
        if candidate in df.columns:
            return candidate

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        if col != "zobs":
            return col

    die(f"Could not infer a y column. Available columns: {list(df.columns)}")


def build_image_index(image_dir: str) -> dict[int, str]:
    index: dict[int, str] = {}
    if not os.path.isdir(image_dir):
        return index

    for entry in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, entry)
        if not os.path.isfile(path):
            continue
        suffix = os.path.splitext(entry)[1].lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        numbers = re.findall(r"\d+", os.path.splitext(entry)[0])
        if not numbers:
            continue
        frame = int(numbers[-1])
        index.setdefault(frame, path)
    return index


def build_display_df(
    csv_path: str,
    x_col: str,
    y_col: str,
    frame_col: Optional[str],
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        die(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    if x_col not in df.columns:
        die(f"Missing x column '{x_col}'. Available columns: {list(df.columns)}")
    if y_col not in df.columns:
        die(f"Missing y column '{y_col}'. Available columns: {list(df.columns)}")

    frame_source = frame_col or ("frame" if "frame" in df.columns else x_col)
    if frame_source not in df.columns:
        die(f"Missing frame column '{frame_source}'. Available columns: {list(df.columns)}")

    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df["_frame_value"] = pd.to_numeric(df[frame_source], errors="coerce")
    if frame_source == x_col:
        df["_frame_value"] = df["_frame_value"].round()

    mask = df[x_col].notna() & df[y_col].notna() & df["_frame_value"].notna()
    df = df.loc[mask].copy().reset_index(drop=True)
    if df.empty:
        die("No rows left after filtering non-numeric x/y/frame values.")

    df["frame"] = df["_frame_value"].round().astype(int)
    df["_row_id"] = range(len(df))
    return df


@lru_cache(maxsize=256)
def load_image_as_data_url(path: str) -> str:
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".gif":
        mime = "image/gif"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{encoded}"


def format_number(value, decimals: int = 4) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def build_info_panel(row: pd.Series, x_col: str, y_col: str, image_path: Optional[str]):
    lines = [
        f"Frame: {int(row['frame'])}",
        f"{x_col}: {format_number(row[x_col], 2)}",
        f"{y_col}: {format_number(row[y_col], 4)}",
    ]

    for col, decimals in (
        ("Miller", None),
        ("asu", None),
        ("Resolution", 4),
        ("Fo^2_raw", 2),
        ("Fo^2_scaled", 2),
        ("Fo^2_sigma_raw", 2),
        ("Fo^2_sigma_scaled", 2),
        ("Fc^2", 2),
    ):
        if col not in row.index:
            continue
        label = col
        value = row[col] if decimals is None else format_number(row[col], decimals)
        lines.append(f"{label}: {value}")

    if image_path is None:
        lines.append("Image: not found for this frame")
    else:
        lines.append(f"Image: {os.path.basename(image_path)}")

    return html.Div([html.Div(line) for line in lines], style={"lineHeight": "1.5"})


def make_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    hover_columns = [
        col
        for col in (
            "Miller",
            "asu",
            "Resolution",
            "frame",
            "Fo^2_raw",
            "Fo^2_scaled",
            "Fo^2_sigma_scaled",
            "Fc^2",
        )
        if col in df.columns
    ]

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_col, y_col: y_col},
        hover_data=hover_columns,
        custom_data=["_row_id"],
        height=760,
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.75})
    fig.update_layout(hovermode="closest")
    return fig


def main() -> None:
    args = parse_args()
    csv_path = os.path.abspath(args.csv)
    image_dir = os.path.abspath(args.image_dir)

    preview_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    y_col = choose_y_column(preview_df, args.y_col) if preview_df is not None else args.y_col
    if y_col is None:
        die(f"Missing CSV file: {csv_path}")

    df = build_display_df(csv_path=csv_path, x_col=args.x_col, y_col=y_col, frame_col=args.frame_col)
    image_index = build_image_index(image_dir)
    title = args.title or f"{y_col} vs {args.x_col}"
    figure = make_figure(df, args.x_col, y_col, title)

    app = Dash(__name__)

    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "row", "gap": "20px"},
        children=[
            html.Div(
                style={"flex": "2"},
                children=[
                    dcc.Graph(
                        id="reflection-plot",
                        figure=figure,
                        clear_on_unhover=False,
                        style={"height": "92vh"},
                    )
                ],
            ),
            html.Div(
                style={
                    "flex": "1",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "gap": "12px",
                },
                children=[
                    html.H3("Frame Viewer"),
                    html.Div(
                        id="frame-info",
                        children="Hover over a reflection to display its frame.",
                        style={"width": "100%"},
                    ),
                    html.Img(
                        id="frame-image",
                        style={
                            "maxWidth": "100%",
                            "maxHeight": "75vh",
                            "border": "1px solid #ccc",
                            "background": "#fafafa",
                        },
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("frame-image", "src"),
        Output("frame-info", "children"),
        Input("reflection-plot", "hoverData"),
        Input("reflection-plot", "clickData"),
    )
    def update_image(hover_data, click_data):
        point = None
        if hover_data and hover_data.get("points"):
            point = hover_data["points"][0]
        elif click_data and click_data.get("points"):
            point = click_data["points"][0]

        if point is None:
            return None, "Hover over a reflection to display its frame."

        row_id = int(point["customdata"][0])
        row = df.iloc[row_id]
        image_path = image_index.get(int(row["frame"]))
        info = build_info_panel(row, args.x_col, y_col, image_path)

        if image_path is None:
            return None, info

        return load_image_as_data_url(image_path), info

    print(f"Viewer ready. Open http://127.0.0.1:{args.port} in a browser.")
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Images indexed: {len(image_index)} from {image_dir}")
    app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
