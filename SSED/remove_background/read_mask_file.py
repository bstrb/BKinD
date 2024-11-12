import h5py

# Function to read the mask file
def read_mask_file(mask_file_path):
    with h5py.File(mask_file_path, 'r') as mask_file:
        # Assuming the mask is stored under '/mask'
        if '/mask' not in mask_file:
            raise ValueError("Mask dataset '/mask' not found in the mask file.")
        mask_dataset = mask_file['/mask']
        mask = mask_dataset[:]
    return mask