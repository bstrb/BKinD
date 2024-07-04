# app_gui.py

# Standard library imports
import os

# Third-party imports
from tkinter import filedialog, messagebox, ttk

# Only if start image
from PIL import Image, ImageTk

# Custom module imports
from gui.tooltip import ToolTip
from gui.setup_main_frame import setup_main_frame
from gui.process_data_gui import process_data_gui

class App:
    def __init__(self, root, style, version):
        self.root = root
        self.style = style
        self.root.title(f"BKinD {version} Filter Electron Diffraction Data")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize frames
        self.setup_frames()

        # Setup welcome frame
        self.setup_welcome_frame()

    def setup_frames(self):
        self.welcome_frame = ttk.Frame(self.root, padding=(20, 20))
        self.welcome_frame.pack(padx=10, pady=10)

        self.main_frame = ttk.Frame(self.root, padding=(20, 20))

    def setup_welcome_frame(self):
        welcome_label = ttk.Label(self.welcome_frame, text="Welcome to BKinD!", style="Custom.TLabel")
        welcome_label.pack(pady=20)

        # Load the image
        current_working_directory = os.getcwd()
        image_path = os.path.join(current_working_directory, "assets/bkind_logo.jpg")
        image = Image.open(image_path)
        resized_image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(resized_image)
        
        image_label = ttk.Label(self.welcome_frame, image=photo)
        image_label.image = photo  # Keep a reference to the image
        image_label.pack(pady=10)

        # Add buttons to start filtering
        start_button_3ded = ttk.Button(self.welcome_frame, text="Start Filtering 3DED Data", style="Custom.TButton", command=lambda: self.show_main_frame(xray=False))
        start_button_3ded.pack(pady=10)

        start_button_xray = ttk.Button(self.welcome_frame, text="Start Filtering Merged 3DED or SCXRD Data", style="Custom.TButton", command=lambda: self.show_main_frame(xray=True))
        start_button_xray.pack(pady=10)

    def create_tooltip(self, widget, text):
        tooltip = ToolTip(widget, self.style)
        widget.bind('<Enter>', lambda event: tooltip.showtip(text))
        widget.bind('<Leave>', lambda event: tooltip.hidetip())

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to close the application?"):
            self.root.destroy()

    def show_welcome_frame(self):
        self.main_frame.pack_forget()
        self.welcome_frame.pack(fill='both', expand=True)

    def show_main_frame(self, xray=False):
        self.welcome_frame.pack_forget()
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.setup_main_frame(xray)

    def browse_directory(self, variable, check_function=None, error_message=None):
        directory = filedialog.askdirectory()
        if directory:
            variable.set(directory)
            if check_function and not check_function(directory):
                messagebox.showerror("File Error", error_message)
                variable.set('Enter valid directory')

    def browse_xds(self):
        self.browse_directory(
            self.xds_dir,
            self.check_xds_files,
            "The selected XDS directory must contain the following files: INTEGRATE.HKL, XDS_ASCII.HKL, and xds.inp. Please select a valid directory."
        )

    def browse_shelx(self, xray=False):
        self.browse_directory(
            self.shelx_dir_xray if xray else self.shelx_dir,
            self.check_shelx_files_xray if xray else self.check_shelx_files,
            "The selected SHELX directory must contain at least one .ins file and one .hkl file." if xray else "The selected SHELX directory must contain at least one .ins file. Please select a valid directory."
        )

    def browse_output(self, xray=False):
        self.browse_directory(
            self.output_dir_xray if xray else self.output_dir
        )

    def check_xds_files(self, directory):
        required_files = ['INTEGRATE.HKL', 'XDS_ASCII.HKL', 'xds.inp']
        return all(os.path.exists(os.path.join(directory, f)) for f in required_files)

    def check_shelx_files(self, directory):
        return any(file.endswith(".ins") for file in os.listdir(directory))
    
    def check_shelx_files_xray(self, directory):
        files = os.listdir(directory)
        has_ins = any(file.endswith(".ins") for file in files)
        has_hkl = any(file.endswith(".hkl") for file in files)
        return has_ins and has_hkl

    def process_data_gui(self, xray=False):
        process_data_gui(self, xray)

    def setup_main_frame(self, xray=False):
        setup_main_frame(self, xray)