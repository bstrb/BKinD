# tooltip.py

# Third-party imports
import tkinter as tk

class ToolTip:
    """Class to create a tooltip for a widget."""

    def __init__(self, widget, style, delay=500):
        """
        Initialize the ToolTip object.

        Parameters:
        widget : tk.Widget
            The widget to which the tooltip is attached.
        style : ttk.Style
            The style used for the tooltip's font.
        delay : int
            Delay in milliseconds before the tooltip is shown.
        """
        self.widget = widget
        self.style = style
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.delay = delay
        self.after_id = None

    def showtip(self, text):
        """Display text in tooltip window."""
        if self.tipwindow or not text:
            return

        def create_tooltip():
            """Create the tooltip window."""
            x, y, _, cy = self.widget.bbox("insert")
            x = x + self.widget.winfo_rootx() + 25
            y = y + cy + self.widget.winfo_rooty() + 25
            self.tipwindow = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            label = tk.Label(tw, text=text, justify=tk.LEFT,
                             background="lightyellow", relief=tk.SOLID, borderwidth=1,
                             font=self.style.lookup("TLabel", "font"))
            label.pack(ipadx=1)

        self.after_id = self.widget.after(self.delay, create_tooltip)

    def hidetip(self):
        """Hide the tooltip window."""
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None
