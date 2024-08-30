# create_widgets.py

import tkinter as tk

def create_widgets(self):
    # Button to browse and select the initial .img file
    self.browse_button = tk.Button(self.root, text="Browse Image", command=self.browse_image)
    self.browse_button.pack(pady=10)

    # Label to display the selected file name (initially empty)
    self.file_name_label = tk.Label(self.root, text="", fg="blue")
    self.file_name_label.pack(pady=5)


    # ANNULAR MODEL
    # Add a dropdown or radio buttons to select the fitting method
    self.fitting_method = tk.StringVar(value="Gaussian")  # Default to Gaussian fitting
    tk.Label(self.root, text="Fitting Method:").pack(pady=5)
    tk.Radiobutton(self.root, text="Gaussian", variable=self.fitting_method, value="Gaussian").pack()
    tk.Radiobutton(self.root, text="Annular", variable=self.fitting_method, value="Annular").pack()
    # ANNULAR MODEL
    
    # Entry to input the sigma level
    self.sigma_level_label = tk.Label(self.root, text="Sigma Level:")
    self.sigma_level_label.pack(pady=5)
    self.sigma_level_entry = tk.Entry(self.root)
    self.sigma_level_entry.pack(pady=10)
    self.sigma_level_entry.insert(0, "2")  # Default value of 2 sigma

    # Button to display the image with the Gaussian area
    self.display_image_button = tk.Button(self.root, text="Display Image with Gaussian Area", command=self.load_and_display_image, state=tk.DISABLED)
    self.display_image_button.pack(pady=10)

    # Dropdown to select the normalization method
    self.normalization_var = tk.StringVar(value="z_score")  # Default value
    self.normalization_label = tk.Label(self.root, text="Normalization Method:")
    self.normalization_label.pack(pady=5)
    self.normalization_menu = tk.OptionMenu(self.root, self.normalization_var, "sum", "min_max", "z_score", "total", "log")
    self.normalization_menu.pack(pady=10)

    # Checkboxes for selecting which intensities to plot
    self.plot_inside_var = tk.BooleanVar(value=True)
    self.plot_inside_checkbox = tk.Checkbutton(self.root, text="Inside Gaussian Region", variable=self.plot_inside_var)
    self.plot_inside_checkbox.pack(pady=5)

    self.plot_outside_var = tk.BooleanVar(value=False)
    self.plot_outside_checkbox = tk.Checkbutton(self.root, text="Outside Gaussian Region", variable=self.plot_outside_var)
    self.plot_outside_checkbox.pack(pady=5)

    self.plot_total_var = tk.BooleanVar(value=False)
    self.plot_total_checkbox = tk.Checkbutton(self.root, text="Total Intensity", variable=self.plot_total_var)
    self.plot_total_checkbox.pack(pady=5)

    self.plot_difference_var = tk.BooleanVar(value=False)
    self.plot_difference_checkbox = tk.Checkbutton(self.root, text="|Inside - Outside|", variable=self.plot_difference_var)
    self.plot_difference_checkbox.pack(pady=5)

    # Checkbox to enable/disable smoothing
    self.smooth_var = tk.BooleanVar(value=True)
    self.smooth_checkbox = tk.Checkbutton(self.root, text="Smoothen Curve", variable=self.smooth_var, command=self.toggle_smoothing)
    self.smooth_checkbox.pack(pady=10)

    # Slider to adjust smoothing window size
    self.smooth_slider = tk.Scale(self.root, from_=3, to=51, orient=tk.HORIZONTAL, label="Smoothing Window Size", state=tk.NORMAL)
    self.smooth_slider.pack(pady=10)

    # Button to calculate the center beam intensity in the folder
    self.calculate_button = tk.Button(self.root, text="Calculate Intensities", command=self.calculate_intensities, state=tk.DISABLED)
    self.calculate_button.pack(pady=10)
