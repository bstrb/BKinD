import h5py
import numpy as np
from process_image_with_ellipses import process_image  # Ensure correct import path

def test_image_1800(h5_file_path, mask_path):
    # Load mask
    with h5py.File(mask_path, 'r') as h5_file:
        mask = h5_file['/mask'][()]

    # Load image 1800
    with h5py.File(h5_file_path, 'r') as h5_file:
        image = h5_file['/entry/data/images'][1800, :, :].astype(np.float32)

    # Process image 1800 with plotting enabled
    beam_center = process_image(
        image=image, 
        image_index=1800, 
        mask=mask, 
        plot=True,  # Enable plotting for visualization
        verbose=True,
        intensity_levels=[0.3, 0.5, 0.7, 0.9],
        bin_widths=[0.1, 0.1, 0.1, 0.1],
        residual_threshold=1.0,
        min_samples=10,
        max_trials=3000,
        eps=20.0,
        dbscan_min_samples=3
    )

    if beam_center:
        print(f"Beam center for image 1800: {beam_center}")
    else:
        print("Beam center for image 1800 could not be determined.")

if __name__ == "__main__":
    h5_file_path = '/home/buster/UOX1/UOX1_min_10/UOX1_min_10.h5'
    mask_path = '/home/buster/mask/pxmask.h5'
    test_image_1800(h5_file_path, mask_path)
