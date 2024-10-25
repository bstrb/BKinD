import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a mask with a circular exclusion and specific angular exclusions
def create_exclusion_mask(image_shape, circle_radius=50, angle1=0, angle2=45):
    mask = np.ones(image_shape, dtype=np.uint8) * 255
    center = (image_shape[1] // 2, image_shape[0] // 2)

    # Draw a circle in the center to exclude
    cv2.circle(mask, center, circle_radius, 0, -1)

    # Remove specific angular regions symmetrically across the entire frame
    angles = [(angle1, angle2), (angle1 + 180, angle2 + 180)]
    for angle_range in angles:
        for angle in range(angle_range[0], angle_range[1] + 1):
            theta = np.deg2rad(angle)
            for radius in range(max(image_shape)):
                x = int(center[0] + radius * np.cos(theta))
                y = int(center[1] - radius * np.sin(theta))
                if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                    mask[y, x] = 0

    return mask

# Example usage
if __name__ == "__main__":
    # Define image shape
    image_shape = (1024, 1024)  # Height x Width

    # Create mask (angles 30 to 60 will be excluded, as well as their symmetric counterparts from 210 to 240)
    mask = create_exclusion_mask(image_shape, circle_radius=30, angle1=-20, angle2=20)

    # Plot the mask and save it to a file
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title('Exclusion Mask')
    plt.axis('off')
    plt.savefig('exclusion_mask.png', bbox_inches='tight', dpi=300)  # Save the plot
