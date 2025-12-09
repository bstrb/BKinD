# DFM Interactive Viewer

## How to run
1. Open a terminal and go to this folder:
   cd viewer

2. Make the launcher executable (first time only):
   chmod +x run_viewer.sh

3. Start the viewer:
   ./run_viewer.sh

The first run will create a Conda environment called "dfmviewer"
(this takes a short moment). After that, the viewer starts immediately.

## Usage
- When the script starts, a link is shown (usually http://127.0.0.1:8050).
- Open the link in your browser.
- Click any point in the DFM plot to display the corresponding PNG image.
- Press Ctrl+C in the terminal to stop the viewer.

## Folder structure
viewer/
  environment.yml
  run_viewer.sh
  dfm_viewer.py
  sample_df_no_filter.csv
  images/00001.png ... 00752.png
