import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//7.1.05.tiff'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray_image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Determine dynamic range
min_val, max_val = np.min(gray_image), np.max(gray_image)

# Set threshold based on image range
if max_val <= 255:
    threshold_value = 50  # uint8 range
else:
    # Normalize to [0,1] float and adjust threshold
    gray_image = gray_image.astype(np.float32) / max_val
    threshold_value = 50 / 255.0

# Apply thresholding
if max_val <= 255:
    # Binary mask: pixels less than threshold_value are foreground (True)
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
else:
    # For float images, create mask manually
    binary_mask = (gray_image < threshold_value).astype(np.uint8) * 255

# Prepare images for display
if max_val <= 255:
    display_gray = gray_image.astype(np.float32) / 255.0
else:
    display_gray = gray_image  # already normalized

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
