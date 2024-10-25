import numpy as np
import h5py
import cv2
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from create_exclusion_mask import create_exclusion_mask

# Gaussian 2D function for fitting
def gaussian_2d(xy, x0, y0, sigma_x, sigma_y, amplitude, offset):
    x, y = xy
    g = offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))
    return g.ravel()

# Preprocess the image to reduce noise and apply thresholding
def preprocess_image(image):
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Adaptive Thresholding to create a binary mask for the beam stopper and beam spot
    beam_stopper_mask = cv2.adaptiveThreshold(
        (blurred / np.max(blurred) * 255).astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Optionally remove small artifacts from the mask
    kernel = np.ones((5, 5), np.uint8)
    beam_stopper_mask = cv2.morphologyEx(beam_stopper_mask, cv2.MORPH_OPEN, kernel)

    return beam_stopper_mask

# Find the largest contour in the mask
def find_largest_contour(beam_stopper_mask):
    # Find contours of the beam spot
    contours, _ = cv2.findContours(beam_stopper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return None
    if not contours:
        return None

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour

# Find the beam center for a single frame
def find_beam_center_single_frame(h5_file_path, frame_index):
    # Load the HDF5 file and access the specified dataset
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Extract the specified frame
        dataset = h5_file['/entry/data/images']
        
        # Check if the frame_index is within the range
        if frame_index >= len(dataset):
            print(f"Frame index {frame_index} is out of range. The dataset has {len(dataset)} frames.")
            return None

        # Get the specific image/frame
        image = dataset[frame_index]

    # Ensure the image is compatible with OpenCV
    image = np.array(image, dtype=np.float32)

    # Create a mask with a circular exclusion and specific angular exclusions
    mask = create_exclusion_mask(image.shape, circle_radius=50, angle_ranges=[(89, 90), (269, 270)])

    # Apply the mask to the image
    image_masked = cv2.bitwise_and(image, image, mask=mask)

    # Preprocessing the image
    beam_stopper_mask = preprocess_image(image_masked)

    # Find contours of the beam spot
    largest_contour = find_largest_contour(beam_stopper_mask)

    # If no contours found, return None
    if largest_contour is None:
        print("No contours found for frame.")
        return None

    # Create a bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    roi = image[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]  # Apply the mask to the region of interest

    # Prepare data for 2D Gaussian fitting, considering only unmasked regions
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    valid_pixels = roi_mask > 0  # Only consider pixels that are not masked out
    X_valid = X[valid_pixels]
    Y_valid = Y[valid_pixels]
    roi_valid = roi[valid_pixels]

    # Ensure that there are enough valid pixels for fitting
    if len(roi_valid) < 10:  # Arbitrary threshold to ensure enough data points
        print("Not enough valid pixels for Gaussian fitting.")
        return None

    # Set the initial guess to always be (w/2, h/2) within the ROI
    initial_guess = (w / 2, h / 2, w / 4, h / 4, np.max(roi_valid), np.min(roi_valid))

    try:
        # Fit a 2D Gaussian to the region of interest, considering only unmasked data
        popt, _ = curve_fit(gaussian_2d, (X_valid, Y_valid), roi_valid.ravel(), p0=initial_guess)
        x0, y0, sigma_x, sigma_y, amplitude, offset = popt

        # Calculate the beam center in the original image coordinates
        cX = int(x + x0)
        cY = int(y + y0)
    except RuntimeError:
        print("Gaussian fit failed for frame.")
        return None

    # Optional: Draw the detected center for visualization
    output_image = cv2.cvtColor((image / np.max(image) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.circle(output_image, (cX, cY), 5, (0, 255, 0), -1)

    # Overlay the mask on the original image for visualization
    mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_overlay[:, :, 1] = 0  # Set the green channel to zero for better visibility
    mask_overlay[:, :, 2] = 0  # Set the red channel to zero for better visibility
    output_image = cv2.addWeighted(output_image, 0.7, mask_overlay, 0.3, 0)

    # Save the output for visual verification
    plt.figure()
    plt.imshow(output_image)
    plt.title(f"Beam Center for Frame {frame_index} with Mask")
    plt.scatter(cX, cY, color='red', label='Beam Center')
    plt.legend()
    plt.axis('off')
    plt.savefig(f'beam_center_frame_{frame_index}.png')  # Save the output to a file

    print(f"Beam center for frame {frame_index} found at: ({cX}, {cY})")
    return cX, cY

# Example usage
h5_file_path = '/home/buster/leidata/hMTH1_TH287_Serial/0deg/hMTH1_TH287_Serial_spot2_300nm_CL1350_015speed_0degree_20241019_2126.h5'  # Replace with your HDF5 file path
frame_index = 11068  # Index of the frame to process (e.g., the first frame)

# Find the beam center for the specified frame
center = find_beam_center_single_frame(h5_file_path, frame_index)
if center:
    print(f"Beam center coordinates: {center}")
