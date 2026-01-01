import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image using PIL for better TIFF support
image_path = r'D:\DIP_Handout\DIP_LAB2\DIP_LAB\Dataset\4.2.06.tiff'
image_01 = np.array(Image.open(image_path))

# Check if image has 3 channels (RGB)
if image_01.ndim != 3 or image_01.shape[2] < 3:
    raise ValueError("The image must be RGB with at least 3 channels.")

# Create copies for each color channel
r = image_01.copy()
g = image_01.copy()
b = image_01.copy()

# Zero out other channels
r[:, :, [1, 2]] = 0  # Keep Red
g[:, :, [0, 2]] = 0  # Keep Green
b[:, :, [0, 1]] = 0  # Keep Blue

# Prepare output and titles
output = [image_01, r, g, b]
titles = ['Original Image', 'Red Channel', 'Green Channel', 'Blue Channel']

# Plotting
plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(output[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

