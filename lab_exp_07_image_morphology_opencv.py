import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread(cv2.samples.findFile("E://4-1//DIP Lab//Tutorial_2//misc//4.2.01.tiff"), cv2.IMREAD_GRAYSCALE)  # Or use your own path
if image is None:
    raise FileNotFoundError("Image not found.")

# Apply Otsu's thresholding
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Create structuring element (disk-shaped approximation using ellipse)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Disk of radius ~3

# Apply morphological operations
eroded = cv2.erode(binary_image, kernel, iterations=1)
dilated = cv2.dilate(binary_image, kernel, iterations=1)

# Plot the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

images = [
    (image, 'Original Grayscale Image'),
    (binary_image, 'Binary Image (Otsu Threshold)'),
    (eroded, 'Eroded Image'),
    (dilated, 'Dilated Image')
]

for ax, (img, title) in zip(axes.flat, images):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
