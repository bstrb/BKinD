# process_data_gui.py

# Standard Library Imports
import os
import shutil
import re

# Third-Party Imports
from tkinter import messagebox

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

    # Function to validate the crystal name
    def is_valid_crystal_name(name):
        # Valid if the name contains only alphanumeric characters, is not empty, and has no spaces
        return bool(name) and re.match(r'^[a-zA-Z0-9]+$', name)

    # Gather user input
    dir_ok, xds_dir, shelx_dir, output_dir = get_dir(self, xray)
    crystal_name = self.crystal_name.get().strip()

    if dir_ok:
        # Validate the crystal name
        if is_valid_crystal_name(crystal_name):
            input_ok, completeness, filtering_percentage, step_size, num_steps, step_mode, custom_steps, include_steps = get_input(self, crystal_name)
        else:
            messagebox.showerror("Invalid Input", "Invalid crystal name. Ensure it is not empty and contains only letters and numbers without spaces or special characters.")
            input_ok = False
    else:
        input_ok = False

    run_refine_wght = self.wght_refinement_var.get()
    run_solve_removed = self.solve_filtered_var.get()
    run_solve_remaining = self.solve_remaining_var.get()

    if input_ok:
        output_folder = get_subfolder(output_dir, crystal_name, completeness, xray)

        try:
            if os.path.exists(output_folder):
                user_choice = show_option_dialog(self.root, self.style)
                if user_choice == 'redo':
                    prepare_progress_bar(self, shelx_dir, output_dir, crystal_name, completeness, run_solve_remaining, xds_dir, xray)

                    start_completeness = compute_start_completeness(output_folder)

                    if start_completeness <= completeness:
                        shutil.rmtree(output_folder)
                        raise InvalidCompletenessError(start_completeness, completeness)
                    elif step_mode == 'custom':
                        for step in custom_steps:
                            if start_completeness <= step:
                                shutil.rmtree(output_folder)
                                raise InvalidCompletenessError(start_completeness, step)

                    target_percentages = create_percentage_list(0, completeness, step_size, num_steps, step_mode, custom_steps, include_steps)

                    if process_data(output_folder, target_percentages, filtering_percentage, run_refine_wght, run_solve_removed, run_solve_remaining, xds_dir, xray):
                        dlg = PostProcessDialog(self.root, output_folder, self.style, DFM_plot=not xray)
                        self.root.wait_window(dlg)

                elif user_choice == 'show':
                    dlg = PostProcessDialog(self.root, output_folder, self.style, DFM_plot=not xray)
                    self.root.wait_window(dlg)
                elif user_choice == 'cancel':
                    pass
            else:
                prepare_progress_bar(self, shelx_dir, output_dir, crystal_name, completeness, run_solve_remaining, xds_dir, xray)

                start_completeness = compute_start_completeness(output_folder)

                if start_completeness <= completeness:
                    shutil.rmtree(output_folder)
                    raise InvalidCompletenessError(start_completeness, completeness)
                elif step_mode == 'custom':
                    for step in custom_steps:
                        if start_completeness <= step:
                            shutil.rmtree(output_folder)
                            raise InvalidCompletenessError(start_completeness, step)

                target_percentages = create_percentage_list(0, completeness, step_size, num_steps, step_mode, custom_steps, include_steps)

                if process_data(output_folder, target_percentages, filtering_percentage, run_refine_wght, run_solve_removed, run_solve_remaining, xds_dir, xray):
                    dlg = PostProcessDialog(self.root, output_folder, self.style, DFM_plot=not xray)
                    self.root.wait_window(dlg)

        except InvalidCompletenessError as e:
            messagebox.showerror(
                "Invalid Completeness Error",
                f"Invalid target completeness specified:\n\n"
                f"Target Completeness: {e.target_completeness:.2f}%\n"
                f"Start Completeness: {e.start_completeness:.2f}%\n\n"
                f"Please ensure that the target completeness is less than the start completeness."
            )
    else:
        pass
