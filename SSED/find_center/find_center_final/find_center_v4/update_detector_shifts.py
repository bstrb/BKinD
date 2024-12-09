import h5py
import numpy as np

def update_detector_shifts(
    h5_file_path,
    beam_centers_dict,
    selected_positions,
    framesize,
    pixels_per_meter=17857.14285714286
):
    """
    Update detector shift datasets based on computed beam centers.

    Parameters:
    - h5_file_path (str): Path to the HDF5 file containing datasets.
    - beam_centers_dict (dict): Dictionary mapping positions in the dataset to beam centers [x, y].
    - selected_positions (list): List of positions in the dataset that were processed.
    - framesize (int): Size of the image frame (assuming square images).
    - pixels_per_meter (float): Conversion factor from pixels to meters.

    Returns:
    - None
    """
    presumed_center = framesize / 2  # Half of the framesize

    # Open the HDF5 file and prepare to update datasets
    with h5py.File(h5_file_path, 'r+') as h5_file:
        # Access datasets
        center_x_dataset = h5_file['/entry/data/center_x']
        center_y_dataset = h5_file['/entry/data/center_y']
        num_images = center_x_dataset.shape[0]

        # Ensure datasets for detector shifts exist; if not, create them
        if '/entry/data/det_shift_x_mm' not in h5_file['/entry/data']:
            h5_file['/entry/data'].create_dataset('det_shift_x_mm', data=np.zeros(num_images), dtype='float64')
        if '/entry/data/det_shift_y_mm' not in h5_file['/entry/data']:
            h5_file['/entry/data'].create_dataset('det_shift_y_mm', data=np.zeros(num_images), dtype='float64')

        det_shift_x_mm_dataset = h5_file['/entry/data/det_shift_x_mm']
        det_shift_y_mm_dataset = h5_file['/entry/data/det_shift_y_mm']

        for pos in selected_positions:
            if pos in beam_centers_dict:
                beam_center = beam_centers_dict[pos]
                # Calculate detector shifts in millimeters
                det_shift_x = -((beam_center[0] - presumed_center) / pixels_per_meter) * 1000  # Convert meters to mm
                det_shift_y = -((beam_center[1] - presumed_center) / pixels_per_meter) * 1000  # Convert meters to mm
                det_shift_x_mm_dataset[pos] = det_shift_x
                det_shift_y_mm_dataset[pos] = det_shift_y
            else:
                # Set to -1 if processing failed
                det_shift_x_mm_dataset[pos] = -1
                det_shift_y_mm_dataset[pos] = -1

    print('Updated detector shifts written to HDF5 file', flush=True)
