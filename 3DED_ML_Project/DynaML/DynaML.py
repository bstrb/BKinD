# main.py

import shutil

import tkinter as tk
from tkinter import filedialog, messagebox
import os 

from prepare_data import prepare_data
from train_model import train_model
from predict_dynamical_effects import predict_dynamical_effects
from filter_reflections import filter_reflections
from filter_reflections_randomly import filter_reflections_randomly


from extract_data_from_file import extract_data_from_file
from create_xdsconv import create_xdsconv
from run_process import run_process
from extract_r1_value import extract_r1_value

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
        os.makedirs(predicted_folder, exist_ok=True)

        random_folder = os.path.join(output_folder, "random")
        os.makedirs(random_folder, exist_ok=True)

        # Copy the .ins file to both the predicted and random folders with the name 'dynaml.ins'
        shutil.copy(ins_file, os.path.join(predicted_folder, 'dynaml.ins'))
        shutil.copy(ins_file, os.path.join(random_folder, 'dynaml.ins'))
        
        # Extract data from the .HKL file
        unit_cell_constants, space_group_number = extract_data_from_file(hkl_file)

        if not unit_cell_constants or not space_group_number:
            return  # Exit if data extraction fails
        
        # Create xdsconv.inp in the predicted folder
        create_xdsconv(unit_cell_constants, space_group_number, predicted_folder)   

        # Create xdsconv.inp in the random folder
        create_xdsconv(unit_cell_constants, space_group_number, random_folder)     

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

        # After moving the generated xdsconv.inp to the predicted folder
        run_process(command=["xdsconv"], directory=predicted_folder, suppress_output=True)

        # After running xdsconv in predicted folder, run shelxl
        run_process(command=["shelxl"], directory=predicted_folder, input_file=".ins", suppress_output=True)

        # After moving the generated xdsconv.inp to the random folder
        run_process(command=["xdsconv"], directory=random_folder, suppress_output=True)

        # After running xdsconv in random folder, run shelxl
        run_process(command=["shelxl"], directory=random_folder, input_file=".ins", suppress_output=True)

        # Extract R1 values from both folders
        predicted_r1 = extract_r1_value(os.path.join(predicted_folder, "dynaml.res"))
        random_r1 = extract_r1_value(os.path.join(random_folder, "dynaml.res"))

        if predicted_r1 is not None and random_r1 is not None:
            # Compare the R1 values
            comparison = "lower than" if predicted_r1 < random_r1 else "higher than" if predicted_r1 > random_r1 else "equal to"
            messagebox.showinfo("R1 Comparison", f"Predicted R1 (all data): {predicted_r1}\nRandom R1 (all data): {random_r1}\n\nThe R1 for the predicted data is {comparison} the R1 for the random data.")
        else:
            messagebox.showerror("Error", "Failed to extract R1 values from the .res files.")
        
        if comparison == "lower than":
            print('predicted data points are likely dynamical')

        # Additional steps can be added here, such as further analysis or validation
        messagebox.showinfo("Success", "Processing completed!")


if __name__ == "__main__":
    root = tk.Tk()
    app = DynaML(root)
    root.mainloop()
