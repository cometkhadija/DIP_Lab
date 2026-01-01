import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image using PIL for better TIFF support
image_path = r'D:\DIP_Handout\DIP_LAB2\DIP_LAB\Dataset\4.2.06.tiff'
image_01 = np.array(Image.open(image_path))

# Check if image has 3 channels (RGB)
if image_01.ndim != 3 or image_01.shape[2] < 3:
    raise ValueError("The image must be RGB with at least 3 channels.")

# Extract individual channels as 2D arrays
red_channel = image_01[:, :, 0]
green_channel = image_01[:, :, 1]
blue_channel = image_01[:, :, 2]

# Create RGB-isolated images
r = image_01.copy()
g = image_01.copy()
b = image_01.copy()

r[:, :, [1, 2]] = 0  # Keep Red
g[:, :, [0, 2]] = 0  # Keep Green
b[:, :, [0, 1]] = 0  # Keep Blue

# Prepare visuals
rgb_images = [image_01, r, g, b]
rgb_titles = ['Original Image', 'Red (RGB)', 'Green (RGB)', 'Blue (RGB)']

gray_images = [None, red_channel, green_channel, blue_channel]
gray_titles = ['', 'Red (Grayscale)', 'Green (Grayscale)', 'Blue (Grayscale)']

# Plot all in one window: 2 rows × 4 columns
plt.figure(figsize=(20, 10))

for i in range(4):
    # Top row: Original + RGB-isolated images
    plt.subplot(2, 4, i + 1)
    plt.imshow(rgb_images[i])
    plt.title(rgb_titles[i])
    plt.axis('off')

    # Bottom row: Grayscale channels (skip original image)
    plt.subplot(2, 4, i + 5)
    if gray_images[i] is not None:
        plt.imshow(gray_images[i], cmap='gray')
        plt.title(gray_titles[i])
    else:
        plt.axis('off')
    plt.axis('off')

plt.tight_layout()
plt.show()