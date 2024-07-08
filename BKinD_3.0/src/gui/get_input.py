# get_input.py

# Third-party imports
from tkinter import messagebox

# def no_unnecessary_zeros(number):
·#     return float(f"{number:.10g}")

def get_input(self, crystal_name):
    try:
        completeness = float(self.completeness.get())
        step_size = float(self.step_size.get())
        filtering_percentage = float(self.filtering_percentage.get())
        if not (0 < completeness <= 100 and 0 < step_size <= 100 - completeness and 0 < filtering_percentage <= 100):
            raise ValueError
        result = messagebox.askokcancel(
            "Filtering Diffraction Data",
            f"Crystal Name: {crystal_name}\n"
            f"Target Completeness: {completeness}%\n"
            f"Step Size: {step_size} p. p.\n"
            f"Filtering Percentage: {filtering_percentage}%"
        )
        if result:
            return completeness, step_size, filtering_percentage
        else:
            return None, None, None
    except ValueError:
        messagebox.showerror("Input Error", "Please ensure all numerical entries are valid percentages between 0 and 100. For example:\n- Target Unique Reflections Percentage: 90\n- Filtering Percentage: 1\n- Intermediate Output Step Size: 1")
        return None, None, None
