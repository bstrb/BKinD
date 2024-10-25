import cv2
import numpy as np

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

def find_largest_contour(beam_stopper_mask):
    # Find contours of the beam spot
    contours, _ = cv2.findContours(beam_stopper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return None
    if not contours:
        return None

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour
