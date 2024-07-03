# multi_progress_bar_window.py

# Third-Party Imports
import tkinter as tk
from tkinter import ttk

class MultiProgressBarWindow:
    def __init__(self, master, tasks):
        self.master = master
        self.tasks = tasks
        self.progress_bars = {}

        # Create and configure the main frame
        self.frame = ttk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Initialize progress bars for each task
        self.init_progress_bars()

    def init_progress_bars(self):
        """Initialize progress bars for all tasks."""
        for task, max_value in self.tasks.items():
            self.add_progress_bar(task, max_value)

    def add_progress_bar(self, task, max_value):
        """Add a progress bar for a specific task."""
        label = ttk.Label(self.frame, text=task)
        label.pack(fill=tk.X, padx=10, pady=5)
        
        progress_bar = ttk.Progressbar(self.frame, maximum=max_value)
        progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_bars[task] = progress_bar

    def update_progress(self, task, value):
        """Update the progress of a specific task."""
        if task in self.progress_bars:
            self.progress_bars[task]['value'] = value
            self.master.update_idletasks()

    def update_tasks(self, tasks):
        """Update the list of tasks and their progress bars."""
        for task, max_value in tasks.items():
            if task in self.progress_bars:
                self.progress_bars[task]['maximum'] = max_value
                self.progress_bars[task]['value'] = 0
            else:
                self.add_progress_bar(task, max_value)

    def is_complete(self):
        """Check if all tasks are complete."""
        for task, progress_bar in self.progress_bars.items():
            if progress_bar['value'] < progress_bar['maximum']:
                return False
        return True

