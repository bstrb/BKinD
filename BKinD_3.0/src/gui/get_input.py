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
        custom_steps_str = self.custom_intermediate_steps.get()
        
        # Split the string by commas, convert to float, and sort in descending order if in custom mode
        custom_steps = sorted([float(step.strip()) for step in custom_steps_str.split(",")], reverse=True) if step_mode == "custom" else []

        message = (
            f"Crystal Name: {crystal_name}\n"
            f"Target Completeness: {completeness}%\n"
            f"Filtering Percentage: {filtering_percentage}%\n"
        )
        if include_steps:
            if step_mode == "size":
                message += f"Intermediate Step Size: {step_size} p. p.\n"
            elif step_mode == "num":
                message += f"Number of Intermediate Steps: {num_steps}\n"
            elif step_mode == "custom":
                message += f"Custom Intermediate Steps: {custom_steps}\n"
        
        if include_steps and step_mode == "size" and not (0 < filtering_percentage <= step_size):
            raise ValueError("Filtering percentage must be less than or equal to step size in 'size' mode when steps are included.")
        
        if not (0 < completeness <= 100):
            raise ValueError("Completeness must be between 0 and 100.")
        if include_steps and step_mode == "size" and not (0 < step_size <= 100 - completeness):
            raise ValueError("Step size must be between 0 and 100 - completeness.")
        if include_steps and step_mode == "num" and (not isinstance(num_steps, int) or num_steps < 1):
            raise ValueError("Number of steps must be an integer greater than or equal to 1.")
        if include_steps and step_mode == "custom" and not all(step > completeness for step in custom_steps):
            raise ValueError("All custom intermediate steps must be greater than the target completeness.")

        result = messagebox.askokcancel("Filtering Diffraction Data", message)
        if result:
            # print(result, completeness, filtering_percentage, step_size, num_steps, step_mode, custom_steps, include_steps)
            return result, completeness, filtering_percentage, step_size, num_steps, step_mode, custom_steps, include_steps
        else:
            return False, None, None, None, None, None, None, None
    except ValueError as e:
        messagebox.showerror("Input Error", f"Please ensure all numerical entries are valid. Error: {str(e)}")
        return False, None, None, None, None, None, None, None
