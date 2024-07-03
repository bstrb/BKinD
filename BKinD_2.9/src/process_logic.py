# process_logic.py

# Standard Library Imports
import tkinter as tk
import threading

# Data Processing Imports
from data_processing.filter import filter
from data_processing.process_filtering_results import process_filtering_results
from data_processing.refine_wght_progress import refine_wght_progress
from data_processing.extract import extract_stats_from_filtering

# GUI Imports
from gui.multi_progress_bar_window import MultiProgressBarWindow

# Test Imports
from util.test.solve_filtered import solve_filtered


def process_data(output_folder, target_percentages, filtering_percentage, run_refine_wght, xds_directory=None, xray=False):
    root = tk.Tk()
    root.title("Progress")

    # Initialize tasks
    tasks = {
        'Filtering': len(target_percentages),
        'Processing filtering results': len(target_percentages),
        'Solving structure for filtered data': len(target_percentages),
    }
    if run_refine_wght:
        tasks['Refining WGHT'] = len(target_percentages)
    tasks['Extracting stats from filtering'] = len(target_percentages)

    progress_window = MultiProgressBarWindow(root, tasks)

    def update_progress(task, value):
        progress_window.update_progress(task, value)
        if progress_window.is_complete():
            root.quit()

    def run_tasks():
        filter(output_folder, target_percentages, filtering_percentage, update_progress=update_progress)
        process_filtering_results(output_folder, target_percentages, xds_directory, xray, update_progress=update_progress)

        solve_filtered(output_folder, target_percentages, xds_directory, xray, update_progress=update_progress)

        if run_refine_wght:
            refine_wght_progress(output_folder, target_percentages, update_progress=update_progress)

        extract_stats_from_filtering(output_folder, target_percentages, update_progress=update_progress)
        # Signal that tasks are done
        root.quit()

    # Run tasks in a separate thread to keep the GUI responsive
    task_thread = threading.Thread(target=run_tasks)
    task_thread.start()

    root.mainloop()
    root.destroy()

    # Ensure the thread is finished before returning
    task_thread.join()

    return True
