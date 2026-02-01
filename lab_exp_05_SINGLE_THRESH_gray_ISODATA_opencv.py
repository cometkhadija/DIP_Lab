import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to compute ISODATA threshold manually
def isodata_threshold(image):
    # Flatten image and normalize to [0, 1] if needed
    flat = image.ravel()
    flat = flat[~np.isnan(flat)]  # Remove NaNs if any
    T = flat.mean()

    while True:
        lower = flat[flat < T]
        upper = flat[flat >= T]

        if len(lower) == 0 or len(upper) == 0:
            break

        new_T = 0.5 * (lower.mean() + upper.mean())

        if abs(T - new_T) < 1e-5:
            break
        T = new_T
    return T

# Load image in grayscale
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//7.1.05.tiff'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if gray_image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Convert to float32 and normalize to [0, 1]
gray_image = gray_image.astype(np.float32) / 255.0

# Apply ISODATA threshold
thresh = isodata_threshold(gray_image)
binary_mask = gray_image < thresh  # Similar to skimage behavior

# Prepare for display
thresholded_image = binary_mask.astype(np.float32)

# Plot original and thresholded images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title(f'ISODATA Thresholded Image\nThreshold = {thresh:.4f}')
axes[1].axis('off')

plt.tight_layout()
plt.show()
