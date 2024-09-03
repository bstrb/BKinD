# gui.py

import tkinter as tk
from tkinter import filedialog, messagebox
import os

from prepare_data import prepare_data
from train_model import train_model
from predict_dynamical_effects import predict_dynamical_effects
from filter_reflections import filter_reflections
from filter_reflections_randomly import filter_reflections_randomly

# Assume the 'prepare_data', 'train_model', and 'filter_reflections' functions are defined here or imported from another module.

class DynaML:
    def __init__(self, root):
        self.root = root
        self.root.title("DynaML: Machine Learning for Dynamical Effects in 3DED with SHELX Validation")

        # Variables to store file paths and output directory
        self.hkl_file_path = tk.StringVar()
        self.ins_file_path = tk.StringVar()
        self.output_dir = tk.StringVar()

        # Create the GUI components
        self.create_widgets()

    def create_widgets(self):
        # Browse XDS_ASCII_CBI.HKL file
        tk.Label(self.root, text="XDS_ASCII_CBI.HKL File:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.hkl_file_path, width=50).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_hkl_file).grid(row=0, column=2, padx=10, pady=5)

        # Browse .ins file
        tk.Label(self.root, text=".ins File:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.ins_file_path, width=50).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_ins_file).grid(row=1, column=2, padx=10, pady=5)

        # Choose output directory
        tk.Label(self.root, text="Output Directory:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_output_dir).grid(row=2, column=2, padx=10, pady=5)

        # Run button to start the processing
        tk.Button(self.root, text="Run", command=self.run).grid(row=3, column=1, padx=10, pady=20)

    def browse_hkl_file(self):
        file_path = filedialog.askopenfilename(title="Select XDS_ASCII_CBI.HKL File", filetypes=[("HKL Files", "*.HKL")])
        if file_path:
            self.hkl_file_path.set(file_path)

    def browse_ins_file(self):
        file_path = filedialog.askopenfilename(title="Select .ins File", filetypes=[("INS Files", "*.ins")])
        if file_path:
            self.ins_file_path.set(file_path)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)


    def run(self):
        hkl_file = self.hkl_file_path.get()
        ins_file = self.ins_file_path.get()
        output_dir = self.output_dir.get()

        if not hkl_file or not ins_file or not output_dir:
            messagebox.showerror("Error", "Please select all files and output directory.")
            return

        # Create the DynaML_OUTPUT folder with two subfolders
        output_folder = os.path.join(output_dir, "DynaML_OUTPUT")
        os.makedirs(output_folder, exist_ok=True)
        predicted_folder = os.path.join(output_folder, "predicted")
        random_folder = os.path.join(output_folder, "random")

        os.makedirs(predicted_folder, exist_ok=True)
        os.makedirs(random_folder, exist_ok=True)

        # Prepare data
        df, header_lines = prepare_data(hkl_file)

        if df is None:
            messagebox.showerror("Error", "Failed to prepare data.")
            return

        # Train model
        model = train_model(df)

        # Predict dynamical effects and add probabilities to DataFrame
        df = predict_dynamical_effects(model, df)
            
        # Predict and filter reflections
        num_removed = filter_reflections(header_lines, df, model, predicted_folder)

        # Filter random reflections, removing the same amount as predicted
        filter_reflections_randomly(header_lines, df, num_removed, random_folder)
        
        # Additional steps can be added here, such as further analysis or validation
        messagebox.showinfo("Success", "Processing completed!")


if __name__ == "__main__":
    root = tk.Tk()
    app = DynaML(root)
    root.mainloop()
