import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load image (grayscale)
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//7.1.05.tiff'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found.")

# Convert to float if needed (handle dynamic range > 255)
if image.max() > 255:
    image = (image / image.max() * 255).astype(np.uint8)
else:
    image = image.astype(np.uint8)

# Apply Otsu's thresholding
_, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Plot original and thresholded images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(binary_mask, cmap='gray')
axes[1].set_title('Otsu Thresholded Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
