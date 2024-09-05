# clean_output_folder.py

import os
import shutil

from tkinter import messagebox

def clean_output_folder(self):
    for filename in os.listdir(self.output_folder):
        file_path = os.path.join(self.output_folder, filename)

        if filename.endswith('.txt') or filename.endswith('.html'):
            continue

        if os.path.isdir(file_path) and filename == 'aggregated_filtered':
            continue

        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    messagebox.showinfo("Cleanup", "Output folder has been cleaned, preserving .txt, .html files and the 'aggregated_filtered' folder.")