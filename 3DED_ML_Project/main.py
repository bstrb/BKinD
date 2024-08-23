# main.py

import tkinter as tk
from tkinter import filedialog, messagebox
from parse_xds_ascii import parse_xds_ascii
from feature_engineering import feature_engineering
from prepare_dataframe import prepare_dataframe
from ML_learn import train_and_evaluate_model

class XDSAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XDS ASCII Analyzer")

        self.file_path = ""

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Button to browse and select the XDS_ASCII.HKL file
        self.browse_button = tk.Button(self.root, text="Browse XDS_ASCII.HKL", command=self.browse_file)
        self.browse_button.pack(pady=10)

        # Button to run the analysis
        self.run_button = tk.Button(self.root, text="Run Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.run_button.pack(pady=10)

    def browse_file(self):
        # Open file dialog to select an XDS_ASCII.HKL file
        self.file_path = filedialog.askopenfilename(filetypes=[("XDS ASCII Files", "*.HKL")], title="Select an XDS_ASCII.HKL file")
        
        if not self.file_path:
            messagebox.showwarning("No file selected", "Please select an XDS_ASCII.HKL file.")
            return

        # Enable the run button after a file is selected
        self.run_button.config(state=tk.NORMAL)

    def run_analysis(self):
        if not self.file_path:
            messagebox.showwarning("No file selected", "Please select an XDS_ASCII.HKL file.")
            return

        # Step 1: Parse the XDS_ASCII file
        try:
            data = parse_xds_ascii(self.file_path)
            messagebox.showinfo("Parsing Completed", "Successfully parsed the XDS_ASCII.HKL file.")
        except Exception as e:
            messagebox.showerror("Parsing Error", f"An error occurred during parsing: {e}")
            return

        # Step 2: Perform feature engineering
        try:
            features = feature_engineering(data)
            messagebox.showinfo("Feature Engineering Completed", "Successfully performed feature engineering.")
        except Exception as e:
            messagebox.showerror("Feature Engineering Error", f"An error occurred during feature engineering: {e}")
            return

        # Step 3: Prepare the DataFrame
        try:
            df = prepare_dataframe(features)
            messagebox.showinfo("Data Preparation Completed", "Successfully prepared the DataFrame.")
        except Exception as e:
            messagebox.showerror("Data Preparation Error", f"An error occurred during DataFrame preparation: {e}")
            return

        # Step 4: Machine Learning
        try:
            train_and_evaluate_model(df)
            messagebox.showinfo("Analysis Completed", "Machine learning analysis completed successfully.")
        except Exception as e:
            messagebox.showerror("ML Error", f"An error occurred during machine learning: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = XDSAnalyzerApp(root)
    root.mainloop()
