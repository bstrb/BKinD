# prepare_progress_bar.py

# Third-Party Imports
from tkinter import Toplevel, ttk
from threading import Thread

# Data Processing Imports
from data_processing.prepare import prepare

def prepare_progress_bar(self, shelx_dir, output_dir, crystal_name, completeness, xds_dir, xray):
    # Create a new window for the progress bar
    progress_window = Toplevel(self.root)
    progress_window.title("Preparing Data")

    # Initialize a single progress bar
    progress = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate", maximum=100)
    progress.pack(pady=20)
    self.root.update_idletasks()

    def update_progress(value):
        progress['value'] = value
        progress_window.update_idletasks()

    def run_tasks():
        prepare(shelx_dir, output_dir, crystal_name, completeness, xds_dir, xray, update_progress)
        # Signal that tasks are done
        progress_window.quit()

    # Run tasks in a separate thread to keep the GUI responsive
    task_thread = Thread(target=run_tasks)
    task_thread.start()

    progress_window.mainloop()
    progress_window.destroy()

    # Ensure the thread is finished before returning
    task_thread.join()
