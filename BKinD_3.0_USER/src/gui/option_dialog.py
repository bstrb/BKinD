# option_dialog.py

# Third-party imports
import tkinter as tk
from tkinter import ttk, Toplevel


class OptionDialog(tk.Toplevel):
    """A dialog for selecting an option when the output folder already exists."""

    def __init__(self, parent, style):
        """Initialize the OptionDialog.

        Parameters:
        parent : tk.Widget
            The parent widget.
        style : ttk.Style
            The style used for the dialog's widgets.
        """
        super().__init__(parent)
        self.style = style
        self.user_choice = None

        self.title("Output Folder Exists")
        
        ttk.Label(self, text="The output folder already exists. Choose an option:", style="TLabel").pack(pady=10)

        # Button to redo the filtering
        ttk.Button(self, text="Redo Filtering", command=self.redo_filtering, style="TButton").pack(fill='x', expand=True, padx=20, pady=5)

        # Button to just show the results
        ttk.Button(self, text="Show Filter Results", command=self.show_results, style="TButton").pack(fill='x', expand=True, padx=20, pady=5)
        
        # Button to cancel
        ttk.Button(self, text="Cancel", command=self.cancel, style="TButton").pack(fill='x', expand=True, padx=20, pady=5)

    def redo_filtering(self):
        """Handle the redo filtering action."""
        self.user_choice = 'redo'
        self.destroy()

    def show_results(self):
        """Handle the show results action."""
        self.user_choice = 'show'
        self.destroy()

    def cancel(self):
        """Handle the cancel action."""
        self.user_choice = 'cancel'
        self.destroy() 

def show_option_dialog(root, style):
    """Show the OptionDialog and wait for user input.

    Parameters:
    root : tk.Widget
        The parent widget.
    style : ttk.Style
        The style used for the dialog's widgets.

    Returns:
    str
        The user's choice, either 'redo', 'show', or 'cancel'.
    """
    dlg = OptionDialog(root, style)
    root.wait_window(dlg)  # Wait for the dialog window to close
    return dlg.user_choice
