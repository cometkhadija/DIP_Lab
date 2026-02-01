import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//7.1.05.tiff'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image loaded correctly
if gray_image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Normalize the grayscale image to [0, 1] for consistency
gray_norm = gray_image / 255.0

# Define double threshold values (normalized)
lower_thresh = 50 / 255.0
upper_thresh = 100 / 255.0

# Apply double thresholding
mask = (gray_norm >= lower_thresh) & (gray_norm <= upper_thresh)
thresholded_image = np.zeros_like(gray_norm)
thresholded_image[mask] = 1.0  # Set pixel to 1.0 if in range

# Display using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(gray_norm, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title('Double Thresholded Image\n[50, 100]')
axes[1].axis('off')

plt.tight_layout()
plt.show()


