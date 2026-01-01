import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = r'D:\DIP_Handout\DIP_LAB2\DIP_LAB\Dataset\4.2.06.tiff'
rgb_image = Image.open(image_path).convert('RGB')  # Ensure it's RGB

# Convert to grayscale
gray_image = rgb_image.convert('L')  # 'L' mode is for grayscale

# Convert images to numpy arrays for plotting
rgb_array = np.array(rgb_image)
gray_array = np.array(gray_image)

# Plot both images
plt.figure(figsize=(10, 5))

# RGB image
plt.subplot(1, 2, 1)
plt.imshow(rgb_array)
plt.title('RGB Image')
plt.axis('off')

# Grayscale image
plt.subplot(1, 2, 2)
plt.imshow(gray_array, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
plt.show()