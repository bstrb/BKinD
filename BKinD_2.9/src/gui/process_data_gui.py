# process_data_gui.py

# Standard Library Imports
import os

# Third-Party Imports
from tkinter import messagebox, Tk

# GUI Module Imports
from gui.get_dir import get_dir
from gui.get_input import get_input
from gui.invalid_completeness_error import InvalidCompletenessError
from gui.option_dialog import show_option_dialog
from gui.post_process_dialog import PostProcessDialog
from gui.prepare_progress_bar import prepare_progress_bar

# Logic Module Imports
from process_logic import process_data

# Utility Module Imports
from util.data.compute_start_completeness import compute_start_completeness
from util.file.get_subfolder import get_subfolder
from util.misc.create_percentage_list import create_percentage_list

def process_data_gui(self, xray=False):
    # Gather user input
    xds_dir, shelx_dir, output_dir = get_dir(self, xray)
    crystal_name = self.crystal_name.get().strip()
    completeness, step_size, filtering_percentage = get_input(self, crystal_name)
    run_refine_wght = self.wght_refinement_var.get()
    run_solve_filtered = self.solve_filtered_var.get()

    output_folder = get_subfolder(output_dir, crystal_name, completeness, xray)

    # Create a hidden root window to prevent the entire application from closing
    hidden_root = Tk()
    hidden_root.withdraw()

    try:
        if os.path.exists(output_folder):
            user_choice = show_option_dialog(self.root, self.style)
            if user_choice == 'redo':
                prepare_progress_bar(self, shelx_dir, output_dir, crystal_name, completeness, xds_dir, xray)

                start_completeness = compute_start_completeness(output_folder)

                if start_completeness <= completeness:
                    raise InvalidCompletenessError(start_completeness, completeness)

                target_percentages = create_percentage_list(start_completeness, completeness, step_size)

                if process_data(output_folder, target_percentages, filtering_percentage, run_refine_wght, run_solve_filtered, xds_dir, xray):
                    dlg = PostProcessDialog(hidden_root, output_folder, self.style, DFM_plot=not xray)
                    hidden_root.wait_window(dlg)

            elif user_choice == 'show':
                dlg = PostProcessDialog(hidden_root, output_folder, self.style, DFM_plot=not xray)
                hidden_root.wait_window(dlg)
        else:
            prepare_progress_bar(self, shelx_dir, output_dir, crystal_name, completeness, xds_dir, xray)

            start_completeness = compute_start_completeness(output_folder)

            if start_completeness <= completeness:
                raise InvalidCompletenessError(start_completeness, completeness)

            target_percentages = create_percentage_list(start_completeness, completeness, step_size)

            if process_data(output_folder, target_percentages, filtering_percentage, run_refine_wght, run_solve_filtered, xds_dir, xray):
                dlg = PostProcessDialog(hidden_root, output_folder, self.style, DFM_plot=not xray)
                hidden_root.wait_window(dlg)

    except InvalidCompletenessError as e:
        messagebox.showerror(
            "Invalid Completeness Error",
            f"Invalid target completeness specified:\n\n"
            f"Target Completeness: {e.target_completeness:.2f}%\n"
            f"Start Completeness: {e.start_completeness:.2f}%\n\n"
            f"Please ensure that the target completeness is less than the start completeness."
        )
    finally:
        hidden_root.destroy()  # Ensure the hidden root is destroyed at the end
