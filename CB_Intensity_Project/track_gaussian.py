# track_gaussian.py

import tkinter as tk

from create_widgets import create_widgets

class CenterBeamIntensityApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Center Beam Intensity Calculator")
        
        self.file_path = ""
        self.center = None
        self.sigma_x = None
        self.sigma_y = None

        create_widgets(self)

if __name__ == "__main__":
    root = tk.Tk()
    app = CenterBeamIntensityApp(root)
    root.mainloop()