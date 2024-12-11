import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensure Tkinter backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CenterPlotGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Beam Center Plotter")

        # Data storage
        self.csv_data = []  # list of (chunk_number, xc, yc)
        self.h5_data = []   # list of (chunk_number, xc, yc)

        # Buttons
        self.btn_csv = tk.Button(master, text="Open CSV", command=self.open_csv)
        self.btn_csv.pack(pady=5)

        self.btn_h5 = tk.Button(master, text="Open H5", command=self.open_h5)
        self.btn_h5.pack(pady=5)

        self.btn_quit = tk.Button(master, text="Quit", command=master.quit)
        self.btn_quit.pack(pady=5)
        
        # Matplotlib figure
        self.fig, (self.ax_x, self.ax_y) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Different markers/colors for CSV and H5 data for distinction
        self.csv_marker = 'o'
        self.csv_color = 'red'
        self.h5_marker = '^'
        self.h5_color = 'blue'

        # Range around median for zooming in
        self.RANGE = 5.0

    def open_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file:\n{e}")
            return

        # Check if required columns exist
        required_cols = ['Chunk_Number', 'Xc_px', 'Yc_px']
        for col in required_cols:
            if col not in df.columns:
                messagebox.showerror("Error", f"CSV missing required column: {col}")
                return

        chunk_number = df['Chunk_Number'].values
        xc = df['Xc_px'].values
        yc = df['Yc_px'].values

        self.csv_data.append((chunk_number, xc, yc))
        self.plot_centers()

    def open_h5(self):
        file_path = filedialog.askopenfilename(
            title="Select H5 file",
            filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            with h5py.File(file_path, 'r') as f:
                if '/entry/data/center_x' not in f or '/entry/data/center_y' not in f:
                    messagebox.showerror("Error", "H5 file missing required datasets: /entry/data/center_x and /entry/data/center_y")
                    return
                center_x = f['/entry/data/center_x'][:]
                center_y = f['/entry/data/center_y'][:]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read H5 file:\n{e}")
            return

        # Assume chunk numbers are sequential
        chunk_number = np.arange(1, len(center_x) + 1)
        self.h5_data.append((chunk_number, center_x, center_y))
        self.plot_centers()

    def plot_centers(self, title="Centers"):
        # Clear axes
        self.ax_x.clear()
        self.ax_y.clear()

        # Plot CSV data
        for (chunk_number, xc, yc) in self.csv_data:
            self.ax_x.plot(chunk_number, xc, marker=self.csv_marker, linestyle='-', color=self.csv_color, label='CSV Data' if self.csv_data.index((chunk_number, xc, yc)) == 0 else "")
            self.ax_y.plot(chunk_number, yc, marker=self.csv_marker, linestyle='-', color=self.csv_color, label='')

        # Plot H5 data
        for (chunk_number, xc, yc) in self.h5_data:
            self.ax_x.plot(chunk_number, xc, marker=self.h5_marker, linestyle='-', color=self.h5_color, label='H5 Data' if self.h5_data.index((chunk_number, xc, yc)) == 0 else "")
            self.ax_y.plot(chunk_number, yc, marker=self.h5_marker, linestyle='-', color=self.h5_color, label='')

        self.ax_x.set_ylabel('Xc_px')
        self.ax_y.set_xlabel('Chunk Number')
        self.ax_y.set_ylabel('Yc_px')

        # Create a combined dataset for scaling
        all_xc = []
        all_yc = []

        for (chunk_number, xc, yc) in self.csv_data:
            all_xc.extend(xc)
            all_yc.extend(yc)

        for (chunk_number, xc, yc) in self.h5_data:
            all_xc.extend(xc)
            all_yc.extend(yc)

        if len(all_xc) > 0 and len(all_yc) > 0:
            median_x = np.median(all_xc)
            median_y = np.median(all_yc)

            # Set y-limits around median ± RANGE
            self.ax_x.set_ylim(median_x - self.RANGE, median_x + self.RANGE)
            self.ax_y.set_ylim(median_y - self.RANGE, median_y + self.RANGE)

        # Add legends if data is present
        handles_x, labels_x = self.ax_x.get_legend_handles_labels()
        handles_y, labels_y = self.ax_y.get_legend_handles_labels()
        handles = handles_x + handles_y
        labels = labels_x + labels_y

        # Remove duplicates from legend if any
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            if l not in unique_labels and l != '':
                unique_labels.append(l)
                unique_handles.append(h)

        if unique_handles:
            self.ax_x.legend(unique_handles, unique_labels, loc='best')

        self.ax_x.set_title(title)
        self.ax_x.grid(True)
        self.ax_y.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CenterPlotGUI(root)
    root.mainloop()
