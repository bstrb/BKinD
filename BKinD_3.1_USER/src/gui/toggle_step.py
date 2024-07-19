# toggle_step.py

import tkinter as tk

def toggle_steps(self):
    state = tk.NORMAL if self.include_steps.get() else tk.DISABLED
    self.step_mode_size.config(state=state)
    self.step_mode_num.config(state=state)
    self.step_mode_custom.config(state=state)
    self.toggle_step_mode()

def toggle_step_mode(self):
    if not self.include_steps.get():
        self.step_size_label.grid_remove()
        self.step_size_entry.grid_remove()
        self.num_steps_label.grid_remove()
        self.num_steps_entry.grid_remove()
        self.custom_steps_label.grid_remove()
        self.custom_steps_entry.grid_remove()
    else:
        if self.step_mode.get() == "size":
            self.step_size_label.grid()
            self.step_size_entry.grid()
            self.num_steps_label.grid_remove()
            self.num_steps_entry.grid_remove()
            self.custom_steps_label.grid_remove()
            self.custom_steps_entry.grid_remove()
        elif self.step_mode.get() == "num":
            self.step_size_label.grid_remove()
            self.step_size_entry.grid_remove()
            self.num_steps_label.grid()
            self.num_steps_entry.grid()
            self.custom_steps_label.grid_remove()
            self.custom_steps_entry.grid_remove()
        elif self.step_mode.get() == "custom":
            self.step_size_label.grid_remove()
            self.step_size_entry.grid_remove()
            self.num_steps_label.grid_remove()
            self.num_steps_entry.grid_remove()
            self.custom_steps_label.grid()
            self.custom_steps_entry.grid()