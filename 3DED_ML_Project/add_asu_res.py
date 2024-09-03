import tkinter as tk
from tkinter import filedialog, messagebox
from cctbx import crystal, miller
from cctbx.uctbx import unit_cell
from cctbx.sgtbx import space_group_info
from cctbx.array_family import flex

def parse_xds_ascii(file_path):
    miller_indices = []
    space_group_number = None
    unit_cell_parameters = None
    original_data = []
    lines = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Extract space group number and unit cell parameters from the header
        for line in lines:
            if line.startswith("!SPACE_GROUP_NUMBER="):
                space_group_number = int(line.split('=')[1].strip())
            elif line.startswith("!UNIT_CELL_CONSTANTS="):
                unit_cell_parameters = list(map(float, line.split('=')[1].strip().split()))
            elif line.startswith("!END_OF_HEADER"):
                break

        # Extract original data and Miller indices from the data section
        data_section_started = False
        for line in lines:
            if line.startswith("!END_OF_HEADER"):
                data_section_started = True
                continue
            if data_section_started and not line.startswith('!'):
                items = line.split()
                h, k, l = int(items[0]), int(items[1]), int(items[2])
                miller_indices.append((h, k, l))
                original_data.append(items)

    return miller_indices, space_group_number, unit_cell_parameters, original_data, lines
def calculate_asu_and_resolution(file_path):
    miller_indices, space_group_number, unit_cell_parameters, original_data, lines = parse_xds_ascii(file_path)
    
    # Define the crystal symmetry
    try:
        crystal_symmetry = crystal.symmetry(
            unit_cell=unit_cell(unit_cell_parameters),
            space_group_info=space_group_info(space_group_number)
        )
    except Exception as e:
        print(f"Error in creating crystal symmetry: {e}")
        return
    
    # Create a Miller set from the list of tuples
    try:
        miller_set = miller.set(
            crystal_symmetry=crystal_symmetry,
            indices=flex.miller_index(miller_indices),
            anomalous_flag=False
        )
    except Exception as e:
        print(f"Error in creating Miller set: {e}")
        return

    # Calculate ASU and resolutions
    try:
        asu_miller_array = miller_set.map_to_asu()
        asu_indices = asu_miller_array.indices()
        d_spacings = list(miller_set.d_spacings().data())

        # Update the header to include the new columns
        header_end_index = lines.index("!END_OF_HEADER\n")
        lines.insert(header_end_index, "!ITEM_ASU_H=14\n")
        lines.insert(header_end_index + 1, "!ITEM_ASU_K=15\n")
        lines.insert(header_end_index + 2, "!ITEM_ASU_L=16\n")
        lines.insert(header_end_index + 3, "!ITEM_RES=17\n")

        # Prepare the format string for reconstructing the output, including CBI
        format_str = (
            "{h:4} {k:4} {l:4} {iobs:12.4e} {sigma_iobs:12.4e} {xd:8.1f} {yd:8.1f} {zd:8.1f} "
            "{rlp:8.4f} {peak:4d} {corr:4d} {psi:8.2f} {cbi:8.2f} {asu_h:4d} {asu_k:4d} {asu_l:4d} {resolution:12.6f}\n"
        )

        # Save the new data to a file
        output_file_path = file_path.replace(".HKL", "_ASU_RES.HKL")
        with open(output_file_path, 'w') as file:
            # Write the updated header
            for line in lines:
                file.write(line)
                if line.startswith("!END_OF_HEADER"):
                    break
            
            # Write the data with new columns including CBI
            for i, row in enumerate(original_data):
                updated_line = format_str.format(
                    h=int(row[0]), k=int(row[1]), l=int(row[2]), 
                    iobs=float(row[3]), sigma_iobs=float(row[4]), 
                    xd=float(row[5]), yd=float(row[6]), zd=float(row[7]), 
                    rlp=float(row[8]), peak=int(row[9]), corr=int(row[10]), 
                    psi=float(row[11]), cbi=float(row[12]),  # Include CBI here
                    asu_h=asu_indices[i][0], asu_k=asu_indices[i][1], asu_l=asu_indices[i][2], 
                    resolution=d_spacings[i]
                )
                file.write(updated_line)
            file.write('!END_OF_DATA')

        print(f"Updated file saved as: {output_file_path}")
        messagebox.showinfo("Success", f"File saved as {output_file_path}")

    except Exception as e:
        print(f"Error in calculating ASU or d-spacings: {e}")
        messagebox.showerror("Error", f"Error in calculating ASU or d-spacings: {e}")
        return

class XDSASCIIGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XDS ASCII ASU/Resolution Calculator")

        self.file_path = ""

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Button to browse and select the XDS_ASCII_CBI.HKL file
        self.browse_button = tk.Button(self.root, text="Browse XDS_ASCII_CBI.HKL", command=self.browse_file)
        self.browse_button.pack(pady=10)

        # Button to run the ASU and resolution calculation
        self.run_button = tk.Button(self.root, text="Add ASU/Resolution", command=self.run_calculation, state=tk.DISABLED)
        self.run_button.pack(pady=10)

    def browse_file(self):
        # Open file dialog to select an XDS_ASCII_CBI.HKL file
        self.file_path = filedialog.askopenfilename(filetypes=[("XDS ASCII Files", "*.HKL")], title="Select an XDS_ASCII_CBI.HKL file")
        
        if not self.file_path:
            messagebox.showwarning("No file selected", "Please select an XDS_ASCII_CBI.HKL file.")
            return

        # Enable the run button after a file is selected
        self.run_button.config(state=tk.NORMAL)

    def run_calculation(self):
        if not self.file_path:
            messagebox.showwarning("No file selected", "Please select an XDS_ASCII_CBI.HKL file.")
            return

        # Perform ASU and resolution calculation and save the updated file
        calculate_asu_and_resolution(self.file_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = XDSASCIIGuiApp(root)
    root.mainloop()
