# setup_main_frame_test.py

# Standard Library Imports
import platform

# Third-party imports
import tkinter as tk
from tkinter import ttk

# Import tooltip descriptions
from gui.tooltip_descriptions import *

def setup_main_frame(self, xray=False):
    os_name = platform.system()

    # Create a frame for inputs
    input_frame = self.main_frame
    input_frame.pack(fill='both', expand=True, padx=10, pady=10)

    # Set directory variables based on the OS and mode (xray or non-xray) for developer
    if os_name == 'Darwin':
        if xray:
            self.shelx_dir_xray = tk.StringVar(value='/Users/xiaodong/Downloads/SCXRD-DATA/SCXRDLTA')
            self.output_dir_xray = tk.StringVar(value='/Users/xiaodong/Desktop')
        else:
            self.xds_dir = tk.StringVar(value="/Users/xiaodong/Downloads/3DED-DATA/LTA/LTA4/xds")
            self.shelx_dir = tk.StringVar(value='/Users/xiaodong/Downloads/3DED-DATA/LTA/LTA4/shelx')
            self.output_dir = tk.StringVar(value='/Users/xiaodong/Desktop')
    elif os_name == 'Linux' and 'microsoft' in platform.uname().release.lower():
        if xray:
            self.shelx_dir_xray = tk.StringVar(value="/mnt/c/Users/bubl3932/Desktop/SCXRD-DATA/SCXRDLTA")
            self.output_dir_xray = tk.StringVar(value="/mnt/c/Users/bubl3932/Desktop")
        else:
            self.xds_dir = tk.StringVar(value="/mnt/c/Users/bubl3932/Desktop/3DED-DATA/LTA/LTA1/xds")
            self.shelx_dir = tk.StringVar(value="/mnt/c/Users/bubl3932/Desktop/3DED-DATA/LTA/LTA1/shelx")
            self.output_dir = tk.StringVar(value="/mnt/c/Users/bubl3932/Desktop")
    else:
        print("Unsupported OS. This script supports only macOS and WSL.")
        return

    # User settings
    # if xray:
    #     self.shelx_dir_xray = tk.StringVar()
    #     self.output_dir_xray = tk.StringVar()
    # else:
    #     self.xds_dir = tk.StringVar()
    #     self.shelx_dir = tk.StringVar()
    #     self.output_dir = tk.StringVar()

    # Common inputs
    self.crystal_name = tk.StringVar()
    ttk.Label(input_frame, text="Crystal Name:").grid(row=0, column=0, sticky="w", padx=5, pady=(5,20))
    crystal_name_entry = ttk.Entry(input_frame, textvariable=self.crystal_name, width=40)
    crystal_name_entry.grid(row=0, column=1, columnspan=2, sticky="w", padx=5, pady=(5,20))
    self.create_tooltip(crystal_name_entry, TOOLTIP_CRYSTAL_NAME)

    if not xray:
        # XDS Directory
        ttk.Label(input_frame, text="XDS Directory:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        xds_entry = ttk.Entry(input_frame, textvariable=self.xds_dir, width=40)
        xds_entry.grid(row=1, column=1, columnspan=2, sticky="w", padx=5, pady=5)
        xds_browse_button = ttk.Button(input_frame, text="Browse...", command=self.browse_xds)
        xds_browse_button.grid(row=1, column=3, padx=5, pady=5)
        self.create_tooltip(xds_entry, TOOLTIP_XDS_DIR)

    # SHELX Directory
    ttk.Label(input_frame, text="SHELX Directory:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    shelx_entry = ttk.Entry(input_frame, textvariable=self.shelx_dir if not xray else self.shelx_dir_xray, width=40)
    shelx_entry.grid(row=2, column=1, columnspan=2, sticky="w", padx=5, pady=5)
    shelx_browse_button = ttk.Button(input_frame, text="Browse...", command=lambda: self.browse_shelx(xray))
    shelx_browse_button.grid(row=2, column=3, padx=5, pady=5)
    self.create_tooltip(shelx_entry, TOOLTIP_SHELX_DIR["xray"] if xray else TOOLTIP_SHELX_DIR["default"])

    # Output Directory
    ttk.Label(input_frame, text="Output Directory:").grid(row=3, column=0, sticky="w", padx=5, pady=(5,30))
    output_entry = ttk.Entry(input_frame, textvariable=self.output_dir if not xray else self.output_dir_xray, width=40)
    output_entry.grid(row=3, column=1, columnspan=2, sticky="w", padx=5, pady=(5,30))
    output_browse_button = ttk.Button(input_frame, text="Browse...", command=lambda: self.browse_output(xray))
    output_browse_button.grid(row=3, column=3, padx=5, pady=(5,30))
    self.create_tooltip(output_entry, TOOLTIP_OUTPUT_DIR)

    # Target Completeness Percentage
    self.completeness = tk.StringVar(value="99")
    ttk.Label(input_frame, text="Target Completeness:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
    completeness_entry = ttk.Entry(input_frame, textvariable=self.completeness, width=10)
    completeness_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
    self.create_tooltip(completeness_entry, TOOLTIP_COMPLETENESS)

    # Intermediate Steps Checkbox and Options
    self.include_steps = tk.BooleanVar(value=False)
    self.step_mode = tk.StringVar()
    self.step_size = tk.StringVar(value="1")
    self.num_steps = tk.StringVar(value="1")
    self.custom_intermediate_steps = tk.StringVar()

    self.include_steps_check = ttk.Checkbutton(
        input_frame, text="Include Intermediate Steps", variable=self.include_steps, command=self.toggle_steps)
    self.include_steps_check.grid(row=5, column=0, sticky="w", padx=5, pady=5)

    self.step_mode_size = ttk.Radiobutton(
        input_frame, text="Step Size", variable=self.step_mode, value="size", command=self.toggle_step_mode)
    self.step_mode_size.grid(row=6, column=0, sticky="w", padx=5, pady=5)

    self.step_mode_num = ttk.Radiobutton(
        input_frame, text="Number of Steps", variable=self.step_mode, value="num", command=self.toggle_step_mode)
    self.step_mode_num.grid(row=7, column=0, sticky="w", padx=5, pady=5)

    self.step_mode_custom = ttk.Radiobutton(
        input_frame, text="Custom Intermediate Steps", variable=self.step_mode, value="custom", command=self.toggle_step_mode)
    self.step_mode_custom.grid(row=8, column=0, sticky="w", padx=5, pady=5)

    self.step_size_label = ttk.Label(input_frame, text="Step Size:")
    self.step_size_label.grid(row=9, column=0, sticky="w", padx=5, pady=5)
    self.step_size_entry = ttk.Entry(input_frame, textvariable=self.step_size, width=10)
    self.step_size_entry.grid(row=9, column=1, sticky="w", padx=5, pady=5)
    self.create_tooltip(self.step_size_entry, TOOLTIP_STEP_SIZE)

    self.num_steps_label = ttk.Label(input_frame, text="Number of Steps:")
    self.num_steps_label.grid(row=9, column=0, sticky="w", padx=5, pady=5)
    self.num_steps_entry = ttk.Entry(input_frame, textvariable=self.num_steps, width=10)
    self.num_steps_entry.grid(row=9, column=1, sticky="w", padx=5, pady=5)
    self.create_tooltip(self.num_steps_entry, TOOLTIP_NUM_STEPS)

    self.custom_steps_label = ttk.Label(input_frame, text="Custom Intermediate Steps:")
    self.custom_steps_label.grid(row=9, column=0, sticky="w", padx=5, pady=5)
    self.custom_steps_entry = ttk.Entry(input_frame, textvariable=self.custom_intermediate_steps, width=40)
    self.custom_steps_entry.grid(row=9, column=1, columnspan=2, sticky="w", padx=5, pady=5)
    self.create_tooltip(self.custom_steps_entry, "Enter the custom intermediate steps separated by commas.")

    # Initialize the widgets
    self.toggle_steps()

    self.filtering_percentage = tk.StringVar(value="0.1")
    ttk.Label(input_frame, text="Filtering Percentage:").grid(row=10, column=0, sticky="w", padx=5, pady=5)
    filtering_percentage_entry = ttk.Entry(input_frame, textvariable=self.filtering_percentage, width=10)
    filtering_percentage_entry.grid(row=10, column=1, sticky="w", padx=5, pady=5)
    self.create_tooltip(filtering_percentage_entry, TOOLTIP_FILTERING_PERCENTAGE)

    # Create a frame to hold the label and checkbox together
    self.wght_refinement_var = tk.BooleanVar(value=False)
    wght_frame = ttk.Frame(input_frame)
    wght_frame.grid(row=6, column=2, sticky="e", padx=5, pady=5)
    wght_label = ttk.Label(wght_frame, text="Refine WGHT     ")
    wght_label.pack(side="left", padx=(0, 10))  # Padding to add space between label and checkbox
    self.wght_refinement_button = ttk.Checkbutton(wght_frame, variable=self.wght_refinement_var)
    self.wght_refinement_button.pack(side="left")
    self.create_tooltip(wght_frame, TOOLTIP_WGHT_REFINEMENT)

    # Create a frame to hold the label and checkbox together
    self.solve_filtered_var = tk.BooleanVar(value=False)
    solve_filtered = ttk.Frame(input_frame)
    solve_filtered.grid(row=7, column=2, sticky="e", padx=5, pady=5)
    wght_label = ttk.Label(solve_filtered, text="Solve Structure for Removed Data ")
    wght_label.pack(side="left", padx=(0, 10))  # Padding to add space between label and checkbox
    self.solve_filtered_var_button = ttk.Checkbutton(solve_filtered, variable=self.solve_filtered_var)
    self.solve_filtered_var_button.pack(side="left")
    self.create_tooltip(solve_filtered, TOOLTIP_SOLVE_FILTERED)

    # Create a frame to hold the label and checkbox together
    self.solve_remaining_var = tk.BooleanVar(value=False)
    solve_remaining = ttk.Frame(input_frame)
    solve_remaining.grid(row=8, column=2, sticky="e", padx=5, pady=5)
    wght_label = ttk.Label(solve_remaining, text="Solve Structure for Remaining Data ")
    wght_label.pack(side="left", padx=(0, 10))  # Padding to add space between label and checkbox
    self.solve_remaining_var_button = ttk.Checkbutton(solve_remaining, variable=self.solve_remaining_var)
    self.solve_remaining_var_button.pack(side="left")
    self.create_tooltip(solve_remaining, TOOLTIP_SOLVE_FILTERED)

    # Filter Data Button
    self.process_btn = ttk.Button(input_frame, text="Filter 3DED Data" if not xray else "Filter Merged 3DED/SCXRD Data", command=lambda: self.process_data_gui(xray))
    self.process_btn.grid(row=12, column=0, columnspan=10, pady=(50,5))

    # Back to Welcome Frame Button
    self.back_btn = ttk.Button(input_frame, text="Back to Start Frame", command=self.show_welcome_frame)
    self.back_btn.grid(row=13, column=0, columnspan=10, pady=(10,5))

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