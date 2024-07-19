# process_logic.py

# Standard Library Imports
import tkinter as tk
import threading

# Data Processing Imports
from data_processing.filter import filter
from data_processing.process_filtering_results import process_filtering_results
from data_processing.refine_wght_progress import refine_wght_progress
from data_processing.extract import extract_stats_from_filtering
from data_processing.solve_removed import solve_removed
from data_processing.solve_remaining import solve_remaining

# GUI Imports
from gui.multi_progress_bar_window import MultiProgressBarWindow

def process_data(output_folder, target_percentages, filtering_percentage, run_refine_wght, run_solve_removed, run_solve_remaining, xds_directory=None, xray=False):
    root = tk.Tk()
    root.title("Progress")

    # Initialize tasks
    tasks = {
        'Filtering Away Extreme Data': len(target_percentages),
        'Processing Filtering Results': len(target_percentages),
    }
    if run_solve_removed:
        tasks['Solving Structure for Removed Data'] = len(target_percentages)
    if run_solve_remaining:
        tasks['Solving Structure for Remaining Data'] = len(target_percentages)
    if run_refine_wght:
        tasks['Refining WGHT'] = len(target_percentages)
    tasks['Extracting Stats From Filtering'] = len(target_percentages)

    progress_window = MultiProgressBarWindow(root, tasks)

    def update_progress(task, value):
        progress_window.update_progress(task, value)
        if progress_window.is_complete():
            root.quit()

    def run_tasks():
        filter(output_folder, target_percentages, filtering_percentage, update_progress=update_progress)
        process_filtering_results(output_folder, target_percentages, xds_directory, xray, update_progress=update_progress)

        if run_solve_removed:
            solve_removed(output_folder, target_percentages, xds_directory, xray, update_progress=update_progress)

        if run_solve_remaining:
            solve_remaining(output_folder, target_percentages, update_progress=update_progress)

        if run_refine_wght:
            refine_wght_progress(output_folder, target_percentages, update_progress=update_progress)

        extract_stats_from_filtering(output_folder, target_percentages, run_solve_remaining, update_progress=update_progress)
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
