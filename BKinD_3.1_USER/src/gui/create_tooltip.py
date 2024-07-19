# create_tooltip.py

from gui.tooltip import ToolTip

def create_tooltip(self, widget, text):
    """Create a tooltip for a widget.

    Parameters:
    widget : tk.Widget
        The widget to attach the tooltip to.
    text : str
        The text to display in the tooltip.
    """
    tooltip = ToolTip(widget, self.style)
    widget.bind('<Enter>', lambda event: tooltip.showtip(text))
    widget.bind('<Leave>', lambda event: tooltip.hidetip())