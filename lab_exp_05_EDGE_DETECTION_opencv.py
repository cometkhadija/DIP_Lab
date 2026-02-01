import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = 'E://4-1//DIP Lab//Tutorial_2//misc//7.1.05.tiff'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image loaded correctly
if gray_image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Convert to float32 and normalize to [0, 1]
gray_image = gray_image.astype(np.float32) / 255.0

# ----- Prewitt Operator -----
# Prewitt kernels
kernel_prewitt_x = np.array([[ -1, 0, 1],
                             [ -1, 0, 1],
                             [ -1, 0, 1]], dtype=np.float32)

kernel_prewitt_y = np.array([[ 1,  1,  1],
                             [ 0,  0,  0],
                             [-1, -1, -1]], dtype=np.float32)

prewitt_x = cv2.filter2D(gray_image, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(gray_image, -1, kernel_prewitt_y)
edges_prewitt = cv2.magnitude(prewitt_x, prewitt_y)

# ----- Sobel Operator -----
sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobel_x, sobel_y)

# ----- Roberts Operator -----
# Roberts kernels (2x2)
kernel_roberts_x = np.array([[1, 0],
                             [0, -1]], dtype=np.float32)

kernel_roberts_y = np.array([[0, 1],
                             [-1, 0]], dtype=np.float32)

roberts_x = cv2.filter2D(gray_image, -1, kernel_roberts_x)
roberts_y = cv2.filter2D(gray_image, -1, kernel_roberts_y)
edges_roberts = cv2.magnitude(roberts_x, roberts_y)

# ----- Display Results -----
titles = ['Original Grayscale', 'Prewitt Edge', 'Sobel Edge', 'Roberts Edge']
images = [gray_image, edges_prewitt, edges_sobel, edges_roberts]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
#
