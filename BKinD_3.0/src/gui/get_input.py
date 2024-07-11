# get_input.py

# Third-party imports
from tkinter import messagebox

def get_input(self, crystal_name):
    try:
        include_steps = self.include_steps.get()
        step_mode = self.step_mode.get()

        completeness = float(self.completeness.get())
        filtering_percentage = float(self.filtering_percentage.get())

        num_steps = int(self.num_steps.get())
        step_size = float(self.step_size.get())

        message = (
            f"Crystal Name: {crystal_name}\n"
            f"Target Completeness: {completeness}%\n"
            f"Filtering Percentage: {filtering_percentage}%\n"
        )
        if include_steps:
            if step_mode == "size":
                message += f"Intermediate Step Size: {step_size} p. p.\n"
            else:
                message += f"Number Intermediate Steps: {num_steps}"
        
        if include_steps and step_mode == "size" and not (0 < filtering_percentage <= step_size):
            raise ValueError("Filtering percentage must be less than or equal to step size in 'size' mode when steps are included.")
        
        if not (0 < completeness <= 100 and
                0 < step_size <= 100 - completeness and
                isinstance(num_steps, int) and
                num_steps >= 1):
            raise ValueError
        result = messagebox.askokcancel("Filtering Diffraction Data", message)
        if result:
            return result, completeness, filtering_percentage, step_size, num_steps, step_mode, include_steps
        else:
            return False, None, None, None, None, None, None
    except ValueError as e:
        messagebox.showerror("Input Error", f"Please ensure all numerical entries are valid. Error: {str(e)}")
        return False, None, None, None, None, None, None
