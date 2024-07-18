# create_button.py

from tkinter import ttk

def create_button(self, text, command, plot_func, tooltip_text):
    """Create a button with a tooltip.

    Parameters:
    text : str
        The text to display on the button.
    command : function
        The function to call when the button is clicked.
    plot_func : function
        The plotting function to call, if applicable.
    tooltip_text : str
        The text to display in the tooltip.
    """
    btn_command = lambda: command(plot_func) if plot_func else command
    button = ttk.Button(self, text=text, command=btn_command, style="TButton")
    button.pack(fill='x', expand=True, padx=20, pady=5)
    self.create_tooltip(button, tooltip_text)