import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//4.1.02.tiff'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray_image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Determine dynamic range
min_val, max_val = np.min(gray_image), np.max(gray_image)

# Apply thresholding depending on dynamic range
if max_val <= 255:
    threshold_value = 50
    # Using THRESH_BINARY_INV to replicate "gray_image < threshold" condition
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    display_gray = gray_image.astype(np.float32) / 255.0
else:
    # Normalize image to [0,1]
    gray_image_norm = gray_image.astype(np.float32) / max_val
    threshold_value = 50 / 255.0
    binary_mask = (gray_image_norm < threshold_value).astype(np.uint8) * 255
    display_gray = gray_image_norm

display_binary = binary_mask.astype(np.float32) / 255.0

# Plot original and thresholded images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(display_gray, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(display_binary, cmap='gray')
axes[1].set_title('Single Thresholded Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
