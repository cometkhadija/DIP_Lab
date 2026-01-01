import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load image and convert to RGB
image_path = r'D:\DIP_Handout\DIP_LAB2\DIP_LAB\Dataset\4.2.06.tiff'
rgb_image = Image.open(image_path).convert('RGB')
rgb_array = np.array(rgb_image)

# Manual grayscale conversion using weighted sum
gray_array = (0.299 * rgb_array[:, :, 0] +
              0.587 * rgb_array[:, :, 1] +
              0.114 * rgb_array[:, :, 2]).astype(np.uint8)

# Plot both images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb_array)
plt.title('RGB Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray_array, cmap='gray')
plt.title('Grayscale (Manual)')
plt.axis('off')

plt.tight_layout()
plt.show()